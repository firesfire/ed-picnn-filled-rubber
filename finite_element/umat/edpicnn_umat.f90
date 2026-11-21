! =============================================================================
! edpicnn_umat.f90 -- Abaqus/Standard UMAT interface for the frozen ED-PICNN
! neural viscoelastic constitutive model.
!
! Conventions (fixed in stages 1 and 3, see reports 1 and 3):
!   * Hybrid formulation: *User Material, ..., hybrid formulation=INCOMPRESSIBLE
!     The UMAT returns ONLY the deviatoric Cauchy stress; the element pressure
!     field supplies the reaction pressure. No pressure_project anywhere.
!   * STATEV layout:
!       1..63  Fv^1..Fv^7 (3x3 row-major (i,j) each, reference->intermediate
!              deformation gradient, det=1 by the exponential map)
!       64     cumulative dissipation D_cum (MPa)
!       65     initialization flag (0 = uninitialized, 1 = initialized)
!       66     last-step discrete energy balance (diagnostic)
!       67     Newton-failure counter (diagnostic)
!       68     last max subdivision depth (diagnostic)
!   * Fv is stored relative to the original stress-free reference
!     configuration, exactly like DFGRD0/DFGRD1, so no per-step
!     re-referencing transform is needed. The stage-3 objectivity check
!     covers the non-proportional path; no DROT application on Fv.
!   * On Newton failure the routine does NOT commit STATEV(1..64), writes the
!     failure counter, and requests a cutback via PNEWDT. The returned
!     stress/DDSDDE then come from the frozen start state (dt = 0 pass).
!   * Units: MPa - mm - s (consistent with the frozen kernel).
! =============================================================================
subroutine UMAT(STRESS, STATEV, DDSDDE, SSE, SPD, SCD, RPL, &
                DDSDDT, DRPLDE, DRPLDT, &
                STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP, PREDEF, DPRED, &
                CMNAME, NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS, &
                COORDS, DROT, PNEWDT, CELENT, &
                DFGRD0, DFGRD1, NOEL, NPT, LAYER, KSPT, KSTEP, KINC)
  use edpicnn_tangent, only: material_point_tangent
  use edpicnn_integrator, only: branch_diag
  use edpicnn_native, only: kernel_response
  implicit none

  ! ---- Abaqus/Standard UMAT argument list ----------------------------------
  real(8), intent(inout) :: STRESS(NTENS), STATEV(NSTATV)
  real(8), intent(out)   :: DDSDDE(NTENS, NTENS)
  real(8), intent(inout) :: SSE, SPD, SCD
  real(8), intent(out)   :: RPL, DDSDDT(NTENS), DRPLDE(NTENS), DRPLDT
  real(8), intent(in)    :: STRAN(NTENS), DSTRAN(NTENS)
  real(8), intent(in)    :: TIME(2), DTIME, TEMP, DTEMP
  real(8), intent(in)    :: PREDEF(*), DPRED(*)
  character*80, intent(in) :: CMNAME
  integer, intent(in)    :: NDI, NSHR, NTENS, NSTATV, NPROPS
  real(8), intent(in)    :: PROPS(NPROPS)
  real(8), intent(in)    :: COORDS(3)
  real(8), intent(in)    :: DROT(3, 3)
  real(8), intent(inout) :: PNEWDT
  real(8), intent(in)    :: CELENT
  real(8), intent(in)    :: DFGRD0(3, 3), DFGRD1(3, 3)
  integer, intent(in)    :: NOEL, NPT, LAYER, KSPT, KSTEP, KINC

  ! ---- locals ---------------------------------------------------------------
  integer, parameter :: NBR = 7
  real(8) :: Fv_a(3, 3, NBR), Fv_end(3, 3, NBR), sig_dev(3, 3)
  real(8) :: Calg(6, 6)
  real(8) :: abs_tol, rel_tol, cutback
  real(8) :: psi_eq, psi_b(NBR), psi_tot, sig_eq(3, 3)
  real(8) :: sig_branch(3, 3, NBR), X(3, 3, NBR), flow(3, 3, NBR)
  real(8) :: phi(NBR), diss(NBR), diss_tot
  real(8) :: psi_neq_start, psi_neq_end, diss_step, balance
  type(branch_diag) :: diags(NBR)
  logical :: ok
  integer :: b, i, j, mn, maxdepth

  ! tolerances / cutback factor from PROPS (with safe defaults)
  abs_tol = 1.0d-12
  rel_tol = 1.0d-10
  cutback = 0.25d0
  if (NPROPS >= 1) abs_tol = PROPS(1)
  if (NPROPS >= 2) rel_tol = PROPS(2)
  if (NPROPS >= 3) cutback = PROPS(3)

  ! ---- unpack STATEV --------------------------------------------------------
  if (NSTATV < 68) then
    ! layout mismatch: refuse silently is not an option in UMAT; force a
    ! cutback so the analysis does not continue with a corrupt state.
    PNEWDT = min(PNEWDT, cutback)
    return
  end if
  if (STATEV(65) < 0.5d0) then
    ! first touch (SDVINI absent or not applied): identity history
    Fv_a = 0.0d0
    do b = 1, NBR
      do i = 1, 3
        Fv_a(i, i, b) = 1.0d0
      end do
    end do
    STATEV(64) = 0.0d0
    STATEV(66) = 0.0d0
    STATEV(67) = 0.0d0
    STATEV(68) = 0.0d0
    STATEV(65) = 1.0d0
  else
    do b = 1, NBR
      do i = 1, 3
        do j = 1, 3
          mn = 9*(b - 1) + 3*(i - 1) + j
          Fv_a(i, j, b) = STATEV(mn)
        end do
      end do
    end do
  end if

  ! ---- material update + analytic consistent tangent ------------------------
  call material_point_tangent(Fv_a, DFGRD0, DFGRD1, DTIME, abs_tol, rel_tol, &
                              Fv_end, sig_dev, Calg, diags, ok)

  if (.not. ok) then
    ! Newton failure somewhere in the branch updates: do NOT commit state.
    ! Provide a passive (frozen-history) response for this attempt and ask
    ! Abaqus to retry the increment with a smaller time increment.
    STATEV(67) = STATEV(67) + 1.0d0
    PNEWDT = min(PNEWDT, cutback)
    call material_point_tangent(Fv_a, DFGRD0, DFGRD0, 0.0d0, abs_tol, rel_tol, &
                                Fv_end, sig_dev, Calg, diags, ok)
  else
    ! ---- commit converged state ---------------------------------------------
    do b = 1, NBR
      do i = 1, 3
        do j = 1, 3
          mn = 9*(b - 1) + 3*(i - 1) + j
          STATEV(mn) = Fv_end(i, j, b)
        end do
      end do
    end do
    ! energy bookkeeping
    call kernel_response(DFGRD0, Fv_a, psi_eq, psi_b, psi_tot, sig_eq, &
                         sig_branch, sig_dev, X, flow, phi, diss, diss_tot)
    psi_neq_start = sum(psi_b)
    call kernel_response(DFGRD1, Fv_end, psi_eq, psi_b, psi_tot, sig_eq, &
                         sig_branch, sig_dev, X, flow, phi, diss, diss_tot)
    psi_neq_end = sum(psi_b)
    diss_step = 0.0d0
    maxdepth = 0
    do b = 1, NBR
      diss_step = diss_step + diags(b)%diss_increment
      maxdepth = max(maxdepth, diags(b)%subdivisions)
    end do
    ! discrete energy balance: (psi_neq,n - psi_neq,n+1) - dt * X:flow
    balance = (psi_neq_start - psi_neq_end) - diss_step
    STATEV(64) = STATEV(64) + diss_step
    STATEV(66) = balance
    STATEV(68) = dble(maxdepth)
    SSE = psi_tot
    SCD = STATEV(64)
  end if

  ! ---- outputs (deviatoric stress + tangent), Voigt (11,22,33,12,13,23) -----
  STRESS(1) = sig_dev(1, 1)
  STRESS(2) = sig_dev(2, 2)
  STRESS(3) = sig_dev(3, 3)
  STRESS(4) = sig_dev(1, 2)
  STRESS(5) = sig_dev(1, 3)
  STRESS(6) = sig_dev(2, 3)
  DDSDDE = 0.0d0
  if (NTENS >= 6) then
    DDSDDE(1:6, 1:6) = Calg
  else
    DDSDDE(1:NTENS, 1:NTENS) = Calg(1:NTENS, 1:NTENS)
  end if
  RPL = 0.0d0
  DDSDDT = 0.0d0
  DRPLDE = 0.0d0
  DRPLDT = 0.0d0

end subroutine UMAT
