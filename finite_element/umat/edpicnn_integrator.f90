! ============================================================================
! edpicnn_integrator.f90 - Local implicit exponential-map update for the
! frozen ED-PICNN viscous branches (stage 3).
!
! Mirrors dual_potential/integrator.py (source) and code/integrator_numpy.py:
!   residual  r(z) = z - dt * V[ L_v(F1, expm(Z(z)) Fv0) ]   (5-dim)
!   Fv_{n+1}  = expm(Z) Fv_n   (left multiplication, Z symmetric deviatoric)
!   Newton with central-difference Jacobian (h = 1e-7*(1+|z|)), backtracking
!   line search (alpha halving, accept if |r_new| < (1-1e-4*alpha)|r|),
!   initial guess z0 = clip(dt*V[L_v(F1,Fv0)], -0.35, 0.35),
!   recursive midpoint subdivision (determinant-normalised midpoint),
!   tolerances passed in (verification uses abs 1e-12 / rel 1e-10;
!   the source defaults are abs 1e-10 / rel 1e-8).
!
! Determinant preservation: det(expm(Z)) = 1 exactly in exact arithmetic and
! to round-off here, so det(Fv) = 1 is preserved to ~1e-16 per step.
! ============================================================================
module edpicnn_integrator
  use frozen_params, only: dp
  use edpicnn_native, only: branch_response
  implicit none
  private
  public :: advance_branches, advance_one_branch, newton_single
  public :: vector_to_tensor, tensor_to_vector, expm3_sym
  public :: solve5, jacobi3, BASIS

  ! Five orthonormal symmetric deviatoric basis tensors
  ! (same as dual_potential.tensor.symmetric_deviatoric_basis).
  real(dp), parameter :: IR2 = 0.70710678118654752440084436210485_dp  ! 1/sqrt(2)
  real(dp), parameter :: IR6 = 0.40824829046386301636621401245098_dp  ! 1/sqrt(6)
  real(dp), parameter :: BASIS(3, 3, 5) = reshape((/ &
    IR2, 0.0_dp, 0.0_dp, 0.0_dp, -IR2, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, &
    IR6, 0.0_dp, 0.0_dp, 0.0_dp, IR6, 0.0_dp, 0.0_dp, 0.0_dp, -2.0_dp*IR6, &
    0.0_dp, IR2, 0.0_dp, IR2, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, &
    0.0_dp, 0.0_dp, IR2, 0.0_dp, 0.0_dp, 0.0_dp, IR2, 0.0_dp, 0.0_dp, &
    0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, IR2, 0.0_dp, IR2, 0.0_dp /), (/3, 3, 5/))

  integer, parameter :: MAX_ITER = 40
  integer, parameter :: MAX_DEPTH = 10
  integer, parameter :: LS_STEPS = 12

  type, public :: branch_diag
    logical  :: converged = .false.
    integer  :: iterations = 0
    integer  :: subdivisions = 0
    integer  :: line_search_reductions = 0
    real(dp) :: residual_norm = 0.0_dp
    real(dp) :: determinant_error = 0.0_dp
    real(dp) :: diss_increment = 0.0_dp  ! dt * X:flow at the accepted end state
  end type branch_diag

contains

  pure function vector_to_tensor(z) result(Zt)
    real(dp), intent(in) :: z(5)
    real(dp) :: Zt(3, 3)
    integer  :: k
    Zt = 0.0_dp
    do k = 1, 5
      Zt = Zt + z(k)*BASIS(:, :, k)
    end do
  end function vector_to_tensor

  pure function tensor_to_vector(A) result(z)
    real(dp), intent(in) :: A(3, 3)
    real(dp) :: z(5), S(3, 3)
    integer  :: k
    S = 0.5_dp*(A + transpose(A))
    S = S - (S(1,1) + S(2,2) + S(3,3))/3.0_dp * reshape( &
        (/1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, &
          0.0_dp, 0.0_dp, 1.0_dp/), (/3, 3/))
    do k = 1, 5
      z(k) = sum(S*BASIS(:, :, k))
    end do
  end function tensor_to_vector

  ! --- spectral decomposition of a symmetric 3x3 (cyclic Jacobi) ----------
  pure subroutine jacobi3(A, w, V)
    real(dp), intent(in)  :: A(3, 3)
    real(dp), intent(out) :: w(3), V(3, 3)
    real(dp) :: B(3, 3), theta, t, c, s, app, aqq, apq
    real(dp) :: bpj, bqj, vpj, vqj
    integer  :: i, j, p_idx, q_idx, sweep
    B = A
    V = reshape((/1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, &
                  0.0_dp, 0.0_dp, 1.0_dp/), (/3, 3/))
    do sweep = 1, 50
      apq = 0.0_dp
      p_idx = 1
      q_idx = 2
      do i = 1, 2
        do j = i + 1, 3
          if (abs(B(i, j)) > apq) then
            apq = abs(B(i, j))
            p_idx = i
            q_idx = j
          end if
        end do
      end do
      if (apq < 1.0e-300_dp) exit
      app = B(p_idx, p_idx)
      aqq = B(q_idx, q_idx)
      theta = 0.5_dp*(aqq - app)/B(p_idx, q_idx)
      t = sign(1.0_dp, theta)/(abs(theta) + sqrt(1.0_dp + theta*theta))
      c = 1.0_dp/sqrt(1.0_dp + t*t)
      s = t*c
      ! B <- J^T B J and V <- V J simultaneously (read old values first)
      do j = 1, 3
        if (j /= p_idx .and. j /= q_idx) then
          bpj = c*B(j, p_idx) - s*B(j, q_idx)
          bqj = s*B(j, p_idx) + c*B(j, q_idx)
          B(j, p_idx) = bpj
          B(p_idx, j) = bpj
          B(j, q_idx) = bqj
          B(q_idx, j) = bqj
        end if
        vpj = c*V(j, p_idx) - s*V(j, q_idx)
        vqj = s*V(j, p_idx) + c*V(j, q_idx)
        V(j, p_idx) = vpj
        V(j, q_idx) = vqj
      end do
      B(p_idx, p_idx) = c*c*app - 2.0_dp*s*c*B(p_idx, q_idx) + s*s*aqq
      B(q_idx, q_idx) = s*s*app + 2.0_dp*s*c*B(p_idx, q_idx) + c*c*aqq
      B(p_idx, q_idx) = 0.0_dp
      B(q_idx, p_idx) = 0.0_dp
    end do
    w = (/B(1, 1), B(2, 2), B(3, 3)/)
  end subroutine jacobi3

  ! expm of a symmetric 3x3 via spectral decomposition
  pure function expm3_sym(A) result(E)
    real(dp), intent(in) :: A(3, 3)
    real(dp) :: E(3, 3), w(3), V(3, 3)
    call jacobi3(A, w, V)
    E = matmul(V*spread(exp(w), 1, 3), transpose(V))
  end function expm3_sym

  ! ---------------- residual ----------------------------------------------
  subroutine residual_z(z, Fv0, F1, dt, branch, r, Fv, diss)
    real(dp), intent(in)  :: z(5), Fv0(3, 3), F1(3, 3), dt
    integer, intent(in)   :: branch
    real(dp), intent(out) :: r(5), Fv(3, 3), diss
    real(dp) :: Zt(3, 3), X(3, 3), flow(3, 3), phi, psi
    Zt = vector_to_tensor(z)
    Fv = matmul(expm3_sym(Zt), Fv0)
    call branch_response(F1, Fv, branch, X, flow, phi, diss, psi)
    r = z - dt*tensor_to_vector(flow)
  end subroutine residual_z

  ! ---------------- one Newton attempt ------------------------------------
  subroutine newton_single(Fv0, F1, dt, branch, abs_tol, rel_tol, &
                           Fv1, diag, z_out)
    real(dp), intent(in)  :: Fv0(3, 3), F1(3, 3), dt, abs_tol, rel_tol
    integer, intent(in)   :: branch
    real(dp), intent(out) :: Fv1(3, 3)
    type(branch_diag), intent(out) :: diag
    real(dp), intent(out), optional :: z_out(5)
    real(dp) :: z(5), r(5), flow(3, 3), X(3, 3), phi, psi, diss
    real(dp) :: norm, scale, h(5), J(5, 5), dz(5), delta(5)
    real(dp) :: r_plus(5), r_minus(5), candidate(5), cand_norm
    real(dp) :: cand_Fv(3, 3), cand_diss, alpha, znorm
    integer  :: it, col, reduction

    diag = branch_diag(.false., 0, 0.0_dp, 0, 0, 0.0_dp, 0.0_dp)
    if (dt <= 0.0_dp) then
      Fv1 = Fv0
      diag%converged = .true.
      diag%determinant_error = abs(det3l(Fv0) - 1.0_dp)
      if (present(z_out)) z_out = 0.0_dp
      return
    end if
    call branch_response(F1, Fv0, branch, X, flow, phi, diss, psi)
    z = dt*tensor_to_vector(flow)
    z = min(max(z, -0.35_dp), 0.35_dp)
    Fv1 = Fv0
    do it = 1, MAX_ITER
      call residual_z(z, Fv0, F1, dt, branch, r, Fv1, diss)
      norm = sqrt(sum(r*r))
      znorm = sqrt(sum(z*z))
      scale = max(1.0_dp, znorm)
      if (norm <= abs_tol + rel_tol*scale) then
        diag%converged = .true.
        diag%iterations = it
        diag%residual_norm = norm
        diag%determinant_error = abs(det3l(Fv1) - 1.0_dp)
        diag%diss_increment = dt*diss
        if (present(z_out)) z_out = z
        return
      end if
      h = 1.0e-7_dp*(1.0_dp + abs(z))
      do col = 1, 5
        delta = 0.0_dp
        delta(col) = h(col)
        call residual_z(z + delta, Fv0, F1, dt, branch, r_plus, cand_Fv, cand_diss)
        call residual_z(z - delta, Fv0, F1, dt, branch, r_minus, cand_Fv, cand_diss)
        J(:, col) = (r_plus - r_minus)/(2.0_dp*h(col))
      end do
      call solve5(J, -r, dz)
      alpha = 1.0_dp
      diag%iterations = it
      do reduction = 0, LS_STEPS - 1
        candidate = z + alpha*dz
        call residual_z(candidate, Fv0, F1, dt, branch, r_plus, cand_Fv, cand_diss)
        cand_norm = sqrt(sum(r_plus*r_plus))
        if (cand_norm < (1.0_dp - 1.0e-4_dp*alpha)*norm) then
          z = candidate
          Fv1 = cand_Fv
          diss = cand_diss
          diag%line_search_reductions = diag%line_search_reductions + reduction
          exit
        end if
        alpha = alpha*0.5_dp
        if (reduction == LS_STEPS - 1) then
          diag%iterations = MAX_ITER  ! not accepted
          return
        end if
      end do
    end do
    diag%converged = .false.
    diag%residual_norm = norm
  end subroutine newton_single

  ! ---------------- recursive subdivision ---------------------------------
  recursive subroutine advance_one_branch(Fv0, F0, F1, dt, branch, abs_tol, &
                                          rel_tol, state, diag, level)
    real(dp), intent(in)  :: Fv0(3, 3), F0(3, 3), F1(3, 3), dt
    integer, intent(in)   :: branch
    real(dp), intent(in)  :: abs_tol, rel_tol
    real(dp), intent(out) :: state(3, 3)
    type(branch_diag), intent(out) :: diag
    integer, intent(in), optional  :: level
    integer  :: lvl
    real(dp) :: midpoint(3, 3), detm
    real(dp) :: half(3, 3)
    type(branch_diag) :: first, second

    lvl = 0
    if (present(level)) lvl = level
    call newton_single(Fv0, F1, dt, branch, abs_tol, rel_tol, state, diag)
    if (diag%converged) then
      diag%subdivisions = lvl
      return
    end if
    if (lvl >= MAX_DEPTH) then
      diag%converged = .false.
      diag%subdivisions = lvl
      return
    end if
    midpoint = 0.5_dp*(F0 + F1)
    detm = det3l(midpoint)
    midpoint = midpoint*abs(detm) ** (-1.0_dp/3.0_dp)
    call advance_one_branch(Fv0, F0, midpoint, 0.5_dp*dt, branch, abs_tol, &
                            rel_tol, half, first, lvl + 1)
    call advance_one_branch(half, midpoint, F1, 0.5_dp*dt, branch, abs_tol, &
                            rel_tol, state, second, lvl + 1)
    diag = second
    diag%iterations = second%iterations + first%iterations
    diag%line_search_reductions = second%line_search_reductions + first%line_search_reductions
    diag%subdivisions = max(first%subdivisions, second%subdivisions)
    diag%diss_increment = second%diss_increment + first%diss_increment
    if (.not. first%converged) diag%converged = .false.
  end subroutine advance_one_branch

  ! ---------------- all seven branches ------------------------------------
  subroutine advance_branches(Fv, F0, F1, dt, abs_tol, rel_tol, diags, ok)
    real(dp), intent(inout) :: Fv(3, 3, 7)
    real(dp), intent(in)    :: F0(3, 3), F1(3, 3), dt, abs_tol, rel_tol
    type(branch_diag), intent(out) :: diags(7)
    logical, intent(out)    :: ok
    integer  :: b
    real(dp) :: state(3, 3), Fv0(3, 3)
    ok = .true.
    do b = 1, 7
      Fv0 = Fv(:, :, b)
      call advance_one_branch(Fv0, F0, F1, dt, b, abs_tol, rel_tol, state, diags(b))
      Fv(:, :, b) = state
      if (.not. diags(b)%converged) ok = .false.
    end do
  end subroutine advance_branches

  ! ---------------- small dense helpers ------------------------------------
  pure function det3l(A) result(d)
    real(dp), intent(in) :: A(3, 3)
    real(dp) :: d
    d = A(1,1)*(A(2,2)*A(3,3) - A(2,3)*A(3,2)) &
      - A(1,2)*(A(2,1)*A(3,3) - A(2,3)*A(3,1)) &
      + A(1,3)*(A(2,1)*A(3,2) - A(2,2)*A(3,1))
  end function det3l

  subroutine solve5(A, b, x)
    real(dp), intent(in)  :: A(5, 5), b(5)
    real(dp), intent(out) :: x(5)
    real(dp) :: M(5, 6), tmp
    integer  :: i, j, k, piv
    M(:, 1:5) = A
    M(:, 6) = b
    do k = 1, 5
      piv = k
      do i = k + 1, 5
        if (abs(M(i, k)) > abs(M(piv, k))) piv = i
      end do
      if (piv /= k) then
        do j = 1, 6
          tmp = M(k, j)
          M(k, j) = M(piv, j)
          M(piv, j) = tmp
        end do
      end if
      if (abs(M(k, k)) < 1.0e-300_dp) then
        ! singular: leave x = 0 for this direction (caller line-searches)
        x = 0.0_dp
        return
      end if
      do i = k + 1, 5
        M(i, :) = M(i, :) - (M(i, k)/M(k, k))*M(k, :)
      end do
    end do
    do i = 5, 1, -1
      x(i) = (M(i, 6) - sum(M(i, i + 1:5)*x(i + 1:5)))/M(i, i)
    end do
  end subroutine solve5

end module edpicnn_integrator
