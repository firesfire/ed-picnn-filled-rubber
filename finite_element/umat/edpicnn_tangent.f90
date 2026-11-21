! ============================================================================
! edpicnn_tangent.f90 - Fully analytic consistent algorithmic tangent for the
! frozen ED-PICNN local update (stage 4).
!
! Mirrors the discrete map defined by edpicnn_integrator exactly:
!   residual  r(z; F1, Fv0) = z - dt * V[ flow(F1, expm(Z(z)) Fv0) ]
!   F_end = expm(Z*) Fv0,   dz*/dq = -r_z^{-1} r_q   (implicit function theorem)
! with analytic directional derivatives of the branch response
! (branch_partials, using the flow-coefficient partials from stage 4a) and of
! the matrix exponential (dexpm3_sym, eigendecomposition Fr'echet derivative).
!
! Subdivision recursion composes the half-step tangents with the midpoint
! derivative (determinant-normalised midpoint, same as the integrator).
!
! Voigt order (11,22,33,12,13,23); C_alg = (1/J) d(J sigma_dev)/d eps with
! dF = deps . F1 (engineering shear columns), matching
! code/tangent_reference.py.
! ============================================================================
module edpicnn_tangent
  use frozen_params, only: dp, sigma_ref, branch_scale
  use edpicnn_native, only: kernel_response, branch_response, flow_coefficient, &
       zhang_li_energy, branch_energy
  use edpicnn_integrator, only: branch_diag, newton_single, solve5, &
       vector_to_tensor, tensor_to_vector, jacobi3, BASIS
  implicit none
  private
  public :: material_point_tangent, advance_tangent
  public :: branch_partials, eq_partials, calg_assemble
  public :: VOIGT6

  integer, parameter :: MAX_DEPTH_T = 10

  ! Voigt directions (11,22,33,12,13,23); shear carries the 1/2 factor
  real(dp), parameter :: VOIGT6(3, 3, 6) = reshape((/ &
    1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, &
    0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, &
    0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, &
    0.0_dp, 0.5_dp, 0.0_dp, 0.5_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, &
    0.0_dp, 0.0_dp, 0.5_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.5_dp, 0.0_dp, 0.0_dp, &
    0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.5_dp, 0.0_dp, 0.5_dp, 0.0_dp /), &
    (/3, 3, 6/))

contains

  ! ---------------- small local helpers ----------------
  pure function det3t(A) result(d)
    real(dp), intent(in) :: A(3, 3)
    real(dp) :: d
    d = A(1,1)*(A(2,2)*A(3,3) - A(2,3)*A(3,2)) &
      - A(1,2)*(A(2,1)*A(3,3) - A(2,3)*A(3,1)) &
      + A(1,3)*(A(2,1)*A(3,2) - A(2,2)*A(3,1))
  end function det3t

  pure function inv3t(A) result(B)
    real(dp), intent(in) :: A(3, 3)
    real(dp) :: B(3, 3), d
    d = det3t(A)
    B(1, 1) = (A(2,2)*A(3,3) - A(2,3)*A(3,2))/d
    B(2, 1) = (A(2,3)*A(3,1) - A(2,1)*A(3,3))/d
    B(3, 1) = (A(2,1)*A(3,2) - A(2,2)*A(3,1))/d
    B(1, 2) = (A(1,3)*A(3,2) - A(1,2)*A(3,3))/d
    B(2, 2) = (A(1,1)*A(3,3) - A(1,3)*A(3,1))/d
    B(3, 2) = (A(1,2)*A(3,1) - A(1,1)*A(3,2))/d
    B(1, 3) = (A(1,2)*A(2,3) - A(1,3)*A(2,2))/d
    B(2, 3) = (A(1,3)*A(2,1) - A(1,1)*A(2,3))/d
    B(3, 3) = (A(1,1)*A(2,2) - A(1,2)*A(2,1))/d
  end function inv3t

  pure function dev3t(A) result(D)
    real(dp), intent(in) :: A(3, 3)
    real(dp) :: D(3, 3)
    D = A - (A(1,1) + A(2,2) + A(3,3))/3.0_dp * reshape( &
         (/1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, &
           0.0_dp, 0.0_dp, 1.0_dp/), (/3, 3/))
  end function dev3t

  pure function sym3t(A) result(S)
    real(dp), intent(in) :: A(3, 3)
    real(dp) :: S(3, 3)
    S = 0.5_dp*(A + transpose(A))
  end function sym3t

  pure subroutine invariants3t(C, I1, I2)
    real(dp), intent(in)  :: C(3, 3)
    real(dp), intent(out) :: I1, I2
    I1 = C(1,1) + C(2,2) + C(3,3)
    I2 = 0.5_dp*(I1*I1 - sum(C*C))
  end subroutine invariants3t

  pure function expm1_local(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: y
    if (abs(x) < 1.0e-5_dp) then
      y = x*(1.0_dp + x*(0.5_dp + x*(1.0_dp/6.0_dp + x/24.0_dp)))
    else
      y = exp(x) - 1.0_dp
    end if
  end function expm1_local

  pure function hencky3(Cv) result(h)
    real(dp), intent(in) :: Cv(3, 3)
    real(dp) :: h, w(3), V(3, 3), pr(3), d(3)
    call jacobi3(sym3t(Cv), w, V)
    pr = 0.5_dp*log(max(w, 1.0e-30_dp))
    d = pr - (pr(1) + pr(2) + pr(3))/3.0_dp
    h = sqrt(max(2.0_dp*sum(d*d)/3.0_dp, 0.0_dp))
  end function hencky3

  ! Fr'echet derivative of expm at Z (symmetric, eigendecomp w,V, ew=exp(w))
  ! in direction B:  D = V [ G o (V^T B V) ] V^T,
  ! G_ij = (e^{w_i} - e^{w_j})/(w_i - w_j)  (removable singularity at w_i=w_j)
  pure subroutine dexpm3_sym(w, V, ew, B, D)
    real(dp), intent(in)  :: w(3), V(3, 3), ew(3), B(3, 3)
    real(dp), intent(out) :: D(3, 3)
    real(dp) :: Fk(3, 3), G(3, 3), tmp(3, 3), dw
    integer  :: i, j
    Fk = matmul(transpose(V), matmul(B, V))
    do i = 1, 3
      do j = 1, 3
        dw = w(i) - w(j)
        if (abs(dw) < 1.0e-8_dp) then
          G(i, j) = ew(i)
        else if (dw > 0.0_dp) then
          G(i, j) = ew(j)*expm1_local(dw)/dw
        else
          G(i, j) = ew(i)*expm1_local(-dw)/(-dw)
        end if
      end do
    end do
    tmp = G*Fk
    D = matmul(V, matmul(tmp, transpose(V)))
  end subroutine dexpm3_sym

  ! 4th-order contraction: C(i,j,mn) = sum_{p,q} A(i,j,pq) B(pq,mn)
  pure function t4contract(A, B) result(C)
    real(dp), intent(in) :: A(3, 3, 3, 3), B(3, 3, 9)
    real(dp) :: C(3, 3, 9)
    integer  :: i, j, mn, p, q
    C = 0.0_dp
    do mn = 1, 9
      do p = 1, 3
        do q = 1, 3
          C(:, :, mn) = C(:, :, mn) + A(:, :, p, q)*B(p, q, mn)
        end do
      end do
    end do
  end function t4contract

  ! ---------------- branch response directional derivatives ---------------
  ! Given (F, Fv_b) and directions (dF, dFv), returns the directional
  ! derivatives of the driving force X, the flow and the branch Cauchy stress
  ! (values are recomputed internally; the energy slopes a1,b1,a2,b2 from
  ! stage 4a and the flow-coefficient partials make this fully analytic).
  subroutine branch_partials(F, Fv_b, branch, dF, dFv, dX, dflow, dsig)
    integer, intent(in)   :: branch
    real(dp), intent(in)  :: F(3, 3), Fv_b(3, 3), dF(3, 3), dFv(3, 3)
    real(dp), intent(out) :: dX(3, 3), dflow(3, 3), dsig(3, 3)
    real(dp) :: Fvinv(3, 3), Fe(3, 3), Feinv(3, 3), Je, Jf, Fbe(3, 3)
    real(dp) :: Cbe(3, 3), M(3, 3), N(3, 3), X(3, 3), tau(3, 3)
    real(dp) :: dFe(3, 3), c1, dFbe(3, 3), dCbe(3, 3), dM(3, 3), dN(3, 3)
    real(dp) :: dXv(3, 3), dtau(3, 3)
    real(dp) :: I1, I2, dI1, dI2, w1, w2, a1, b1, a2, b2, dw1, dw2, psib
    real(dp) :: J, Jb, Finv(3, 3), cbar, dFbar(3, 3), Fbar(3, 3)
    real(dp) :: Jv, Jvb, cvv, dFvbar(3, 3), Fvbar(3, 3), Cv(3, 3)
    real(dp) :: Cvfull(3, 3), dCvfull(3, 3), w(3), V(3, 3), pr(3)
    real(dp) :: devp(3), dpr(3), hval, dh, tmp(3, 3)
    real(dp) :: jin, dj, coef, ph, cu, cff(4), feats(4), dfeat(4)
    real(dp) :: scl, u, du, dcoef, eye(3, 3)
    integer  :: i, k

    eye = reshape((/1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, &
                    0.0_dp, 0.0_dp, 1.0_dp/), (/3, 3/))

    ! ---- values (mirror branch_response) ----
    Fvinv = inv3t(Fv_b)
    Fe = matmul(F, Fvinv)
    Je = det3t(Fe)
    Jf = Je**(-1.0_dp/3.0_dp)
    Fbe = Jf*Fe
    Cbe = matmul(transpose(Fbe), Fbe)
    call invariants3t(Cbe, I1, I2)
    call branch_energy(Cbe, branch, psib, w1, w2, a1, b1, a2, b2)
    M = 2.0_dp*w1*Cbe + 2.0_dp*w2*(I1*Cbe - matmul(Cbe, Cbe))
    X = dev3t(M)
    jin = 0.5_dp*sum(X*X)/(sigma_ref*sigma_ref)
    J = det3t(F)
    Jb = J**(-1.0_dp/3.0_dp)
    Fbar = Jb*F
    feats(1) = sum(Fbar*Fbar) - 3.0_dp
    feats(2) = I1 - 3.0_dp
    Jv = det3t(Fv_b)
    Jvb = Jv**(-1.0_dp/3.0_dp)
    Fvbar = Jvb*Fv_b
    Cv = matmul(transpose(Fvbar), Fvbar)
    feats(3) = sum(Fvbar*Fvbar) - 3.0_dp
    Cvfull = matmul(transpose(Fv_b), Fv_b)
    feats(4) = hencky3(Cvfull)
    call flow_coefficient(jin, branch, feats, coef, ph, cu, cff)

    ! ---- directional derivatives ----
    ! Fe = F Fv^{-1}  =>  dFe = dF Fv^{-1} - Fe dFv Fv^{-1}
    dFe = matmul(dF, Fvinv) - matmul(Fe, matmul(dFv, Fvinv))
    Feinv = inv3t(Fe)
    c1 = sum(transpose(Feinv)*dFe)          ! <Fe^{-T}, dFe>
    dFbe = Jf*(dFe - (c1/3.0_dp)*Fe)
    dCbe = matmul(transpose(dFbe), Fbe) + matmul(transpose(Fbe), dFbe)
    dI1 = dCbe(1,1) + dCbe(2,2) + dCbe(3,3)
    dI2 = I1*dI1 - sum(Cbe*dCbe)
    dw1 = a1*dI1 + b1*dI2
    dw2 = a2*dI1 + b2*dI2
    ! M = C N with N = 2 w1 I + 2 w2 (I1 I - C)
    N = 2.0_dp*w1*eye + 2.0_dp*w2*(I1*eye - Cbe)
    dN = 2.0_dp*(dw1*eye + dw2*(I1*eye - Cbe) + w2*(dI1*eye - dCbe))
    dM = matmul(dCbe, N) + matmul(Cbe, dN)
    dX = dev3t(dM)
    dj = sum(X*dX)/(sigma_ref*sigma_ref)

    ! feature directionals
    Finv = inv3t(F)
    cbar = sum(transpose(Finv)*dF)
    dFbar = Jb*(dF - (cbar/3.0_dp)*F)
    dfeat(1) = 2.0_dp*sum(Fbar*dFbar)
    dfeat(2) = dI1
    cvv = sum(transpose(Fvinv)*dFv)
    dFvbar = Jvb*(dFv - (cvv/3.0_dp)*Fv_b)
    dfeat(3) = 2.0_dp*sum(Fvbar*dFvbar)
    dCvfull = sym3t(matmul(transpose(dFv), Fv_b) + matmul(transpose(Fv_b), dFv))
    call jacobi3(sym3t(Cvfull), w, V)
    pr = 0.5_dp*log(max(w, 1.0e-30_dp))
    devp = pr - (pr(1) + pr(2) + pr(3))/3.0_dp
    hval = sqrt(max(2.0_dp*sum(devp*devp)/3.0_dp, 0.0_dp))
    if (hval > 1.0e-14_dp) then
      tmp = matmul(transpose(V), matmul(dCvfull, V))
      do i = 1, 3
        dpr(i) = 0.0_dp
        if (w(i) > 1.0e-14_dp) dpr(i) = 0.5_dp*tmp(i, i)/w(i)
      end do
      dh = (2.0_dp/3.0_dp)*sum(devp*dpr)/hval
    else
      dh = 0.0_dp
    end if
    dfeat(4) = dh

    scl = max(branch_scale(branch), 1.0e-14_dp)
    u = jin/(scl*scl)
    du = dj/(scl*scl)
    dcoef = cu*du + sum(cff*dfeat)
    dflow = coef*dX + dcoef*X

    ! sigma = tau/Je, tau = Fbe N Fbe^T
    tau = matmul(Fbe, matmul(N, transpose(Fbe)))
    dtau = matmul(dFbe, matmul(N, transpose(Fbe))) &
         + matmul(Fbe, matmul(dN, transpose(Fbe))) &
         + matmul(Fbe, matmul(N, transpose(dFbe)))
    ! c1 = <Fe^{-T}, dFe> = dJe/Je (log-derivative), so d(1/Je) = -c1/Je
    dsig = dtau/Je - tau*(c1/Je)
  end subroutine branch_partials

  ! ---------------- equilibrium (Zhang-Li) stress directional derivative ----
  subroutine eq_partials(F, dF, dsig)
    real(dp), intent(in)  :: F(3, 3), dF(3, 3)
    real(dp), intent(out) :: dsig(3, 3)
    real(dp) :: J, Jb, Finv(3, 3), cbv, dFbar(3, 3), Fbar(3, 3)
    real(dp) :: Cbar(3, 3), dCbar(3, 3), N(3, 3), dN(3, 3)
    real(dp) :: tau(3, 3), dtau(3, 3), I1, I2, dI1, dI2
    real(dp) :: w1, w2, a1, b1, a2, b2, dw1, dw2, psi, eye(3, 3), dJ

    eye = reshape((/1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, &
                    0.0_dp, 0.0_dp, 1.0_dp/), (/3, 3/))
    J = det3t(F)
    Jb = J**(-1.0_dp/3.0_dp)
    Fbar = Jb*F
    Finv = inv3t(F)
    cbv = sum(transpose(Finv)*dF)
    dFbar = Jb*(dF - (cbv/3.0_dp)*F)
    Cbar = matmul(transpose(Fbar), Fbar)
    dCbar = matmul(transpose(dFbar), Fbar) + matmul(transpose(Fbar), dFbar)
    call invariants3t(Cbar, I1, I2)
    dI1 = dCbar(1,1) + dCbar(2,2) + dCbar(3,3)
    dI2 = I1*dI1 - sum(Cbar*dCbar)
    call zhang_li_energy(Cbar, psi, w1, w2, a1, b1, a2, b2)
    dw1 = a1*dI1 + b1*dI2
    dw2 = a2*dI1 + b2*dI2
    N = 2.0_dp*w1*eye + 2.0_dp*w2*(I1*eye - Cbar)
    dN = 2.0_dp*(dw1*eye + dw2*(I1*eye - Cbar) + w2*(dI1*eye - dCbar))
    tau = matmul(Fbar, matmul(N, transpose(Fbar)))
    dtau = matmul(dFbar, matmul(N, transpose(Fbar))) &
         + matmul(Fbar, matmul(dN, transpose(Fbar))) &
         + matmul(Fbar, matmul(N, transpose(dFbar)))
    dJ = J*cbv
    dsig = dtau/J - tau*(dJ/(J*J))
  end subroutine eq_partials

  ! ---------------- C_alg assembly ----------------------------------------
  ! dsdF(i,j,pq) = d sigma_dev_ij / d F1_pq (full 9-direction tensor);
  ! C_alg = (1/J) d(J sigma_dev)/d eps, dF = deps . F1, Voigt (11,22,33,12,13,23)
  ! Expanded: d(J sigma)/deps = J dsdF:F1^T + J sigma delta_mn, then divided
  ! by J  =>  C_alg(i,j,m,n) = sum_l dsdF(i,j,m,l) F1(n,l) + sig_ij delta_mn
  subroutine calg_assemble(dsdF, sig, F1, Calg)
    real(dp), intent(in)  :: dsdF(3, 3, 9), sig(3, 3), F1(3, 3)
    real(dp), intent(out) :: Calg(6, 6)
    real(dp) :: D(3, 3, 3, 3), D4(3, 3, 3, 3), T(3, 3)
    integer  :: v, m, n, i, j
    do m = 1, 3
      do n = 1, 3
        D4(:, :, m, n) = dsdF(:, :, 3*(m - 1) + n)
      end do
    end do
    do m = 1, 3
      do n = 1, 3
        do i = 1, 3
          do j = 1, 3
            D(i, j, m, n) = sum(D4(i, j, m, :)*F1(n, :)) &
                           + sig(i, j)*merge(1.0_dp, 0.0_dp, m == n)
          end do
        end do
      end do
    end do
    do v = 1, 6
      T = 0.0_dp
      do m = 1, 3
        do n = 1, 3
          T = T + VOIGT6(m, n, v)*D(:, :, m, n)
        end do
      end do
      Calg(1, v) = T(1, 1)
      Calg(2, v) = T(2, 2)
      Calg(3, v) = T(3, 3)
      Calg(4, v) = T(1, 2)
      Calg(5, v) = T(1, 3)
      Calg(6, v) = T(2, 3)
    end do
  end subroutine calg_assemble

  ! ---------------- one-branch tangent (recursive, mirrors advance_one_branch)
  recursive subroutine advance_tangent(Fv0, F0, F1, dt, branch, abs_tol, rel_tol, &
                             Fv1, dEnd_dFa, dEnd_dF0, dEnd_dF1, diag, level)
    real(dp), intent(in)  :: Fv0(3, 3), F0(3, 3), F1(3, 3), dt
    real(dp), intent(in)  :: abs_tol, rel_tol
    integer, intent(in)   :: branch
    real(dp), intent(out) :: Fv1(3, 3)
    real(dp), intent(out) :: dEnd_dFa(3, 3, 9)   ! d F_end / d Fv0
    real(dp), intent(out) :: dEnd_dF0(3, 3, 9)   ! d F_end / d F0 arg
    real(dp), intent(out) :: dEnd_dF1(3, 3, 9)   ! d F_end / d F1
    type(branch_diag), intent(out) :: diag
    integer, intent(in), optional  :: level
    integer  :: lvl, mn, m, n, k, l
    real(dp) :: z(5), Zt(3, 3), w(3), V(3, 3), ew(3), expmZ(3, 3)
    real(dp) :: Dexp(3, 3, 5), dFvdz(3, 3), dzq(5), tv(5)
    real(dp) :: rz(5, 5), E_mn(3, 3), dFz(3, 3)
    real(dp) :: dX(3, 3), dflow(3, 3), dsig(3, 3)
    real(dp) :: midpoint(3, 3), Sinv(3, 3), detm, fac, gdot
    real(dp) :: half(3, 3), dh_dFa(3, 3, 9), dh_dF0(3, 3, 9)
    real(dp) :: dh_dM(3, 3, 9), d2_dh(3, 3, 9), d2_dM(3, 3, 9)
    real(dp) :: d2_dFc(3, 3, 9), dM_dFc(3, 3, 9)
    type(branch_diag) :: first, second

    lvl = 0
    if (present(level)) lvl = level
    dEnd_dFa = 0.0_dp
    dEnd_dF0 = 0.0_dp
    dEnd_dF1 = 0.0_dp
    call newton_single(Fv0, F1, dt, branch, abs_tol, rel_tol, Fv1, diag, z_out=z)
    if (diag%converged) then
      diag%subdivisions = lvl
      if (dt <= 0.0_dp) then
        do mn = 1, 9
          m = (mn - 1)/3 + 1
          n = mod(mn - 1, 3) + 1
          dEnd_dFa(m, n, mn) = 1.0_dp
        end do
        return
      end if
      ! ---- implicit function theorem at the converged z* ----
      Zt = vector_to_tensor(z)
      call jacobi3(Zt, w, V)
      ew = exp(w)
      expmZ = matmul(V*spread(ew, 1, 3), transpose(V))
      do k = 1, 5
        call dexpm3_sym(w, V, ew, BASIS(:, :, k), Dexp(:, :, k))
      end do
      dFz = 0.0_dp
      ! r_z columns: -dt * dV(flow)/dz_l  (dFv = Dexp_l . Fv0)
      do l = 1, 5
        dFvdz = matmul(Dexp(:, :, l), Fv0)
        call branch_partials(F1, Fv1, branch, dFz, dFvdz, dX, dflow, dsig)
        rz(:, l) = -dt*tensor_to_vector(dflow)
      end do
      do k = 1, 5
        rz(k, k) = rz(k, k) + 1.0_dp
      end do
      ! 9 directions each for F1 and Fv0
      do mn = 1, 9
        m = (mn - 1)/3 + 1
        n = mod(mn - 1, 3) + 1
        E_mn = 0.0_dp
        E_mn(m, n) = 1.0_dp
        ! d F_end / d F1_mn
        call branch_partials(F1, Fv1, branch, E_mn, dFz, dX, dflow, dsig)
        tv = dt*tensor_to_vector(dflow)
        call solve5(rz, tv, dzq)
        do k = 1, 5
          dEnd_dF1(:, :, mn) = dEnd_dF1(:, :, mn) &
               + dzq(k)*matmul(Dexp(:, :, k), Fv0)
        end do
        ! d F_end / d Fv0_mn
        call branch_partials(F1, Fv1, branch, dFz, matmul(expmZ, E_mn), &
                             dX, dflow, dsig)
        tv = dt*tensor_to_vector(dflow)
        call solve5(rz, tv, dzq)
        dEnd_dFa(:, :, mn) = matmul(expmZ, E_mn)
        do k = 1, 5
          dEnd_dFa(:, :, mn) = dEnd_dFa(:, :, mn) &
               + dzq(k)*matmul(Dexp(:, :, k), Fv0)
        end do
      end do
      return
    end if
    ! ---- recursive midpoint subdivision (mirror advance_one_branch) ----
    if (lvl >= MAX_DEPTH_T) then
      diag%converged = .false.
      diag%subdivisions = lvl
      return
    end if
    midpoint = 0.5_dp*(F0 + F1)
    detm = det3t(midpoint)
    fac = abs(detm)**(-1.0_dp/3.0_dp)
    ! dM/dF1 (and dM/dF0, same form): M = fac*S, S = 0.5(F0+F1)
    Sinv = inv3t(midpoint)   ! note: derivative uses S, not normalised M
    do mn = 1, 9
      m = (mn - 1)/3 + 1
      n = mod(mn - 1, 3) + 1
      E_mn = 0.0_dp
      E_mn(m, n) = 1.0_dp
      gdot = 0.5_dp*Sinv(n, m)
      dM_dFc(:, :, mn) = fac*(0.5_dp*E_mn - (gdot/3.0_dp)*midpoint)
    end do
    midpoint = midpoint*fac
    call advance_tangent(Fv0, F0, midpoint, 0.5_dp*dt, branch, abs_tol, &
                         rel_tol, half, dh_dFa, dh_dF0, dh_dM, first, lvl + 1)
    call advance_tangent(half, midpoint, F1, 0.5_dp*dt, branch, abs_tol, &
                         rel_tol, Fv1, d2_dh, d2_dM, d2_dFc, second, lvl + 1)
    diag = second
    diag%iterations = second%iterations + first%iterations
    diag%line_search_reductions = second%line_search_reductions &
                                + first%line_search_reductions
    diag%subdivisions = max(first%subdivisions, second%subdivisions)
    diag%diss_increment = second%diss_increment + first%diss_increment
    if (.not. first%converged) diag%converged = .false.
    ! chain rule through h = H1(Fa, F0, M) and end = H2(h, M, Fc)
    dEnd_dFa = t4contract(d2_dh, dh_dFa)
    dEnd_dF1 = d2_dFc + t4contract(d2_dh, t4contract(dh_dM, dM_dFc)) &
             + t4contract(d2_dM, dM_dFc)
    dEnd_dF0 = t4contract(d2_dh, dh_dF0) &
             + t4contract(d2_dh, t4contract(dh_dM, dM_dFc)) &
             + t4contract(d2_dM, dM_dFc)
  end subroutine advance_tangent

  ! ---------------- full material point -----------------------------------
  subroutine material_point_tangent(Fv_a, F0, F1, dt, abs_tol, rel_tol, &
                                    Fv_end, sig_dev, Calg, diags, ok)
    real(dp), intent(in)  :: Fv_a(3, 3, 7), F0(3, 3), F1(3, 3), dt
    real(dp), intent(in)  :: abs_tol, rel_tol
    real(dp), intent(out) :: Fv_end(3, 3, 7), sig_dev(3, 3), Calg(6, 6)
    type(branch_diag), intent(out) :: diags(7)
    logical, intent(out)  :: ok
    integer  :: b, mn, m, n
    real(dp) :: dEnd_dFa(3, 3, 9), dEnd_dF0(3, 3, 9), dEnd_dF1(3, 3, 9)
    real(dp) :: dsdF(3, 3, 9), dsdF_b(3, 3, 9), dsig_eq(3, 3)
    real(dp) :: dX(3, 3), dflow(3, 3), dsig(3, 3), dFvdir(3, 3), E_mn(3, 3)
    real(dp) :: dFz(3, 3)
    real(dp) :: psi_eq, psi_b(7), psi_tot, sig_eq(3, 3), sig_branch(3, 3, 7)
    real(dp) :: X(3, 3, 7), flow(3, 3, 7), phi(7), diss(7), diss_tot

    ok = .true.
    dsdF = 0.0_dp
    do b = 1, 7
      call advance_tangent(Fv_a(:, :, b), F0, F1, dt, b, abs_tol, rel_tol, &
                           Fv_end(:, :, b), dEnd_dFa, dEnd_dF0, dEnd_dF1, &
                           diags(b))
      if (.not. diags(b)%converged) ok = .false.
      do mn = 1, 9
        m = (mn - 1)/3 + 1
        n = mod(mn - 1, 3) + 1
        E_mn = 0.0_dp
        E_mn(m, n) = 1.0_dp
        dFz = 0.0_dp
        ! d sigma_b / d F1_mn (explicit) + (d sigma_b / d Fv) : dEnd/dF1_mn
        call branch_partials(F1, Fv_end(:, :, b), b, E_mn, dFz, &
                             dX, dflow, dsig)
        dsdF_b(:, :, mn) = dsig
        dFvdir = dEnd_dF1(:, :, mn)
        call branch_partials(F1, Fv_end(:, :, b), b, dFz, dFvdir, &
                             dX, dflow, dsig)
        dsdF_b(:, :, mn) = dsdF_b(:, :, mn) + dsig
      end do
      dsdF = dsdF + dsdF_b
    end do
    ! equilibrium part at the end state
    call kernel_response(F1, Fv_end, psi_eq, psi_b, psi_tot, sig_eq, &
                         sig_branch, sig_dev, X, flow, phi, diss, diss_tot)
    do mn = 1, 9
      m = (mn - 1)/3 + 1
      n = mod(mn - 1, 3) + 1
      E_mn = 0.0_dp
      E_mn(m, n) = 1.0_dp
      call eq_partials(F1, E_mn, dsig_eq)
      dsdF(:, :, mn) = dsdF(:, :, mn) + dsig_eq
      dsdF(:, :, mn) = dev3t(dsdF(:, :, mn))
    end do
    call calg_assemble(dsdF, sig_dev, F1, Calg)
  end subroutine material_point_tangent

end module edpicnn_tangent
