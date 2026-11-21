! ============================================================================
! edpicnn_native.f90 - Native (PyTorch-free) double-precision dual-potential
! kernel for the frozen ED-PICNN neural viscoelastic constitutive model.
!
! Mirrors code/native_kernel_numpy.py exactly (which mirrors the frozen
! PyTorch source). All frozen weights come from the read-only module
! frozen_params (umat/includes/frozen_params.f90).
!
! Units: stresses/energies in MPa, rates in 1/s.
! ============================================================================
module edpicnn_native
  use frozen_params, only: dp, zhang_li_D, zhang_li_mu, zhang_li_B, &
       zhang_li_N, zhang_li_beta, tau_s, branch_group, &
       effective_branch_moduli, branch_scale, raw_rates, centers, scales, &
       reference_calibration, state_scales, sigma_ref, strict_convexity, &
       energy_g0_weights, energy_g0_rho_I1, energy_g0_b, energy_g0_p, &
       energy_g0_hardening, energy_g0_hardening_rho, &
       energy_g1_weights, energy_g1_rho_I1, energy_g1_b, energy_g1_p, &
       energy_g1_hardening, energy_g1_hardening_rho, &
       state_gate, nonlinear_coeff, enc0_w, enc0_b, enc2_w, enc2_b, &
       coef_w, coef_b, bias_w, bias_b, gain_w, gain_b, &
       snet_w1, snet_b1, snet_w2, snet_b2
  implicit none
  private
  public :: kernel_response, branch_response, flow_coefficient, &
       zhang_li_energy, branch_energy

  real(dp), parameter :: PI = 3.1415926535897932384626433832795_dp
  real(dp), parameter :: NBASIS = 12.0_dp

contains

  ! ---------------- small tensor helpers ----------------
  pure function det3(A) result(d)
    real(dp), intent(in) :: A(3, 3)
    real(dp) :: d
    d = A(1,1)*(A(2,2)*A(3,3) - A(2,3)*A(3,2)) &
      - A(1,2)*(A(2,1)*A(3,3) - A(2,3)*A(3,1)) &
      + A(1,3)*(A(2,1)*A(3,2) - A(2,2)*A(3,1))
  end function det3

  pure function sym3(A) result(S)
    real(dp), intent(in) :: A(3, 3)
    real(dp) :: S(3, 3)
    S = 0.5_dp*(A + transpose(A))
  end function sym3

  pure function dev3(A) result(D)
    real(dp), intent(in) :: A(3, 3)
    real(dp) :: D(3, 3)
    D = A - (A(1,1) + A(2,2) + A(3,3))/3.0_dp * reshape( &
         (/1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, &
           0.0_dp, 0.0_dp, 1.0_dp/), (/3, 3/))
  end function dev3

  pure subroutine invariants3(C, I1, I2)
    real(dp), intent(in)  :: C(3, 3)
    real(dp), intent(out) :: I1, I2
    I1 = C(1,1) + C(2,2) + C(3,3)
    I2 = 0.5_dp*(I1*I1 - sum(C*C))
  end subroutine invariants3

  elemental function softplus_f(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: y
    y = log(1.0_dp + exp(-abs(x))) + max(x, 0.0_dp)
  end function softplus_f

  elemental function sigmoid_f(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: y
    if (x >= 0.0_dp) then
      y = 1.0_dp/(1.0_dp + exp(-x))
    else
      y = exp(x)/(1.0_dp + exp(x))
    end if
  end function sigmoid_f

  ! Eigenvalues of a symmetric 3x3 matrix, ascending (only values needed).
  pure function eigvalsh3(A) result(w)
    real(dp), intent(in) :: A(3, 3)
    real(dp) :: w(3), B(3, 3), q, p2, p, r, phi
    integer :: i
    q = (A(1,1) + A(2,2) + A(3,3))/3.0_dp
    B = A
    do i = 1, 3
      B(i, i) = B(i, i) - q
    end do
    p2 = B(1,1)**2 + B(2,2)**2 + B(3,3)**2 &
       + 2.0_dp*(B(1,2)**2 + B(1,3)**2 + B(2,3)**2)
    p = sqrt(p2/6.0_dp)
    if (p < 1.0e-300_dp) then
      w = q
      return
    end if
    B = B/p
    r = 0.5_dp*det3(B)
    r = min(1.0_dp, max(-1.0_dp, r))
    phi = acos(r)/3.0_dp
    w(1) = q + 2.0_dp*p*cos(phi + 2.0_dp*PI/3.0_dp)
    w(2) = q + 2.0_dp*p*cos(phi - 2.0_dp*PI/3.0_dp)
    w(3) = q + 2.0_dp*p*cos(phi)
  end function eigvalsh3

  ! sqrt(2/3 * sum dev(ln sqrt(eig Cv))^2) for symmetric Cv
  pure function hencky_dev(Cv) result(h)
    real(dp), intent(in) :: Cv(3, 3)
    real(dp) :: h, w(3), pr(3), m, d(3)
    integer :: i
    w = eigvalsh3(sym3(Cv))
    do i = 1, 3
      pr(i) = 0.5_dp*log(max(w(i), 1.0e-30_dp))
    end do
    m = (pr(1) + pr(2) + pr(3))/3.0_dp
    d = pr - m
    h = sqrt(max(2.0_dp*sum(d*d)/3.0_dp, 0.0_dp))
  end function hencky_dev

  ! ---------------- equilibrium energy (Zhang-Li) ----------------
  ! Also returns invariant derivatives (a1=dw1/dI1, b1=dw1/dI2,
  ! a2=dw2/dI1, b2=dw2/dI2) for the consistent tangent.
  subroutine zhang_li_energy(Cbar, psi, w1, w2, a1, b1, a2, b2)
    real(dp), intent(in)  :: Cbar(3, 3)
    real(dp), intent(out) :: psi, w1, w2, a1, b1, a2, b2
    real(dp) :: I1, I2, x
    call invariants3(Cbar, I1, I2)
    x = I2 - 3.0_dp + zhang_li_N
    if (abs(zhang_li_mu) < 1.0e-12_dp) then
      w1 = zhang_li_D
      psi = zhang_li_D*(I1 - 3.0_dp)
      a1 = 0.0_dp
    else
      w1 = zhang_li_D*exp(-zhang_li_mu*(I1 - 3.0_dp))
      psi = zhang_li_D/zhang_li_mu*(1.0_dp - exp(-zhang_li_mu*(I1 - 3.0_dp)))
      a1 = -zhang_li_mu*w1
    end if
    w2 = zhang_li_B*zhang_li_beta*x**(zhang_li_beta - 1.0_dp)
    psi = psi + zhang_li_B*(x**zhang_li_beta - zhang_li_N**zhang_li_beta)
    b1 = 0.0_dp
    a2 = 0.0_dp
    b2 = zhang_li_B*zhang_li_beta*(zhang_li_beta - 1.0_dp)*x**(zhang_li_beta - 2.0_dp)
  end subroutine zhang_li_energy

  ! ---------------- branch energy (TangentSecant + hardening) ----------------
  ! Also returns invariant derivatives (a1,b1,a2,b2) for the tangent.
  subroutine branch_energy(Cbe, branch, psi, w1, w2, a1, b1, a2, b2)
    integer, intent(in)   :: branch
    real(dp), intent(in)  :: Cbe(3, 3)
    real(dp), intent(out) :: psi, w1, w2, a1, b1, a2, b2
    real(dp) :: I1, I2, x1, x2, z, slope, sat, modulus, hardening, zh
    real(dp) :: wsum(3), rho(3), bb(3), pp(3), hrho, gsum
    real(dp) :: ds, s1, s2
    integer  :: g, k
    call invariants3(Cbe, I1, I2)
    x1 = max(I1 - 3.0_dp, 0.0_dp)
    x2 = max(I2 - 3.0_dp, 0.0_dp)
    g = int(branch_group(branch)) + 1
    modulus = effective_branch_moduli(branch)
    gsum = 0.0_dp
    do k = 1, 7
      if (int(branch_group(k)) + 1 == g) gsum = gsum + effective_branch_moduli(k)
    end do
    hardening = merge(energy_g0_hardening, energy_g1_hardening, g == 1) &
              * modulus/gsum
    hrho = merge(energy_g0_hardening_rho, energy_g1_hardening_rho, g == 1)
    do k = 1, 3
      if (g == 1) then
        wsum(k) = energy_g0_weights(k)
        rho(k)  = energy_g0_rho_I1(k)
        bb(k)   = energy_g0_b(k)
        pp(k)   = energy_g0_p(k)
      else
        wsum(k) = energy_g1_weights(k)
        rho(k)  = energy_g1_rho_I1(k)
        bb(k)   = energy_g1_b(k)
        pp(k)   = energy_g1_p(k)
      end if
    end do
    psi = 0.0_dp
    w1 = 0.0_dp
    w2 = 0.0_dp
    a1 = 0.0_dp
    b1 = 0.0_dp
    a2 = 0.0_dp
    b2 = 0.0_dp
    s1 = 0.0_dp
    s2 = 0.0_dp
    if (I1 > 3.0_dp) s1 = 1.0_dp
    if (I2 > 3.0_dp) s2 = 1.0_dp
    do k = 1, 3
      z = rho(k)*x1 + (1.0_dp - rho(k))*x2
      slope = (1.0_dp + bb(k)*z)**(pp(k) - 1.0_dp)
      sat = ((1.0_dp + bb(k)*z)**pp(k) - 1.0_dp)/(bb(k)*pp(k))
      ds = bb(k)*(pp(k) - 1.0_dp)*(1.0_dp + bb(k)*z)**(pp(k) - 2.0_dp)
      psi = psi + wsum(k)*sat
      w1 = w1 + wsum(k)*slope*rho(k)
      w2 = w2 + wsum(k)*slope*(1.0_dp - rho(k))
      a1 = a1 + wsum(k)*ds*rho(k)*rho(k)*s1
      b1 = b1 + wsum(k)*ds*rho(k)*(1.0_dp - rho(k))*s2
      a2 = a2 + wsum(k)*ds*rho(k)*(1.0_dp - rho(k))*s1
      b2 = b2 + wsum(k)*ds*(1.0_dp - rho(k))*(1.0_dp - rho(k))*s2
    end do
    psi = 0.5_dp*modulus*psi
    w1 = 0.5_dp*modulus*w1
    w2 = 0.5_dp*modulus*w2
    a1 = 0.5_dp*modulus*a1
    b1 = 0.5_dp*modulus*b1
    a2 = 0.5_dp*modulus*a2
    b2 = 0.5_dp*modulus*b2
    zh = hrho*x1 + (1.0_dp - hrho)*x2
    psi = psi + 0.25_dp*hardening*zh*zh
    w1 = w1 + 0.5_dp*hardening*zh*hrho
    w2 = w2 + 0.5_dp*hardening*zh*(1.0_dp - hrho)
    a1 = a1 + 0.5_dp*hardening*hrho*hrho*s1
    b1 = b1 + 0.5_dp*hardening*hrho*(1.0_dp - hrho)*s2
    a2 = a2 + 0.5_dp*hardening*hrho*(1.0_dp - hrho)*s1
    b2 = b2 + 0.5_dp*hardening*(1.0_dp - hrho)*(1.0_dp - hrho)*s2
  end subroutine branch_energy

  ! M = 2 w1 C + 2 w2 (I1 C - C C)
  pure function mandel_f(C, w1, w2) result(M)
    real(dp), intent(in) :: C(3, 3), w1, w2
    real(dp) :: M(3, 3), I1, I2
    call invariants3(C, I1, I2)
    M = 2.0_dp*w1*C + 2.0_dp*w2*(I1*C - matmul(C, C))
  end function mandel_f

  ! ---------------- gate state net: tanh MLP 5 -> 12 -> 1 ----------------
  pure function state_net(x) result(y)
    real(dp), intent(in) :: x(5)
    real(dp) :: y, h(12)
    integer  :: k, j
    do k = 1, 12
      h(k) = snet_b1(k)
      do j = 1, 5
        h(k) = h(k) + snet_w1(k, j)*x(j)
      end do
      h(k) = tanh(h(k))
    end do
    y = snet_b2(1)
    do k = 1, 12
      y = y + snet_w2(1, k)*h(k)
    end do
  end function state_net

  ! ---------------- dissipation: flow coefficient and dual potential -------
  ! feats: 1=total, 2=elastic, 3=viscous, 4=hardening (isochoric measures)
  ! Extended (stage 4): also returns analytic partial derivatives
  !   coef_u = d(coef_flow)/du            (u = j/scale^2)
  !   coef_f = d(coef_flow)/d(feats(1:4))
  ! Derivatives account for softplus/sigmoid/tanh/exp clip activity, the
  ! max(feats,0) gate clipping, and the multiplier exponent clipping.
  subroutine flow_coefficient(j, branch, feats, coef_flow, phi, coef_u, coef_f)
    real(dp), intent(in)  :: j
    integer, intent(in)   :: branch
    real(dp), intent(in)  :: feats(4)
    real(dp), intent(out) :: coef_flow, phi, coef_u, coef_f(4)
    real(dp) :: scl, u, logtau, xnorm(4), h(24), coef(3), bias(12)
    real(dp) :: gain(12), rates(12), args(12), sig(12), A, base_phi
    real(dp) :: values(4), norm(4), ns(5), rs(5), delta, exponent, mult
    real(dp) :: corr_phi, corr_slope, g
    real(dp) :: pre1(24), sp1(24), sp2(24)
    real(dp) :: hs(12), warg, sech2
    real(dp) :: dAdu, nl1, nl2
    real(dp) :: dxn(4), dh1(24), dh2(24), dcoef(3), dbias(12), dgain(12)
    real(dp) :: dA_f(4), dnrm(4), ddelta(4), dmult(4), s2col(4), acc
    logical  :: cclip(4), exclip, ccoef(3), cgain(12), uclip
    integer  :: k, m, gg, f

    scl = max(branch_scale(branch), 1.0e-14_dp)
    u = j/(scl*scl)
    ! clamp u identically to the NumPy kernel: prevents u**3 overflow
    ! during Newton exploration of extreme (non-physical) states
    uclip = abs(u) > 1.0e50_dp
    u = min(max(u, -1.0e50_dp), 1.0e50_dp)
    logtau = log10(tau_s(branch))

    ! Stage-2 PICNN encoder (preactivations saved for backprop)
    xnorm(1) = logtau
    xnorm(2) = feats(2)
    xnorm(3) = feats(3)
    xnorm(4) = feats(4)
    do k = 1, 4
      xnorm(k) = (xnorm(k) - centers(k))/scales(k)
    end do
    do k = 1, 24
      pre1(k) = enc0_b(k)
      do m = 1, 4
        pre1(k) = pre1(k) + enc0_w(k, m)*xnorm(m)
      end do
      sp1(k) = sigmoid_f(pre1(k))
      h(k) = softplus_f(pre1(k))
    end do
    block
      real(dp) :: h2(24)
      do k = 1, 24
        h2(k) = enc2_b(k)
        do m = 1, 24
          h2(k) = h2(k) + enc2_w(k, m)*h(m)
        end do
        sp2(k) = sigmoid_f(h2(k))
        h2(k) = softplus_f(h2(k))
      end do
      h = h2
    end block
    ! heads
    do k = 1, 3
      g = coef_b(k)
      do m = 1, 24
        g = g + coef_w(k, m)*h(m)
      end do
      ccoef(k) = (g > -20.0_dp) .and. (g < 25.0_dp)
      coef(k) = exp(min(max(g, -20.0_dp), 25.0_dp))
    end do
    do k = 1, 12
      bias(k) = bias_b(k)
      do m = 1, 24
        bias(k) = bias(k) + bias_w(k, m)*h(m)
      end do
    end do
    do k = 1, 12
      g = gain_b(k)
      do m = 1, 24
        g = g + gain_w(k, m)*h(m)
      end do
      cgain(k) = (g > -20.0_dp) .and. (g < 18.0_dp)
      gain(k) = exp(min(max(g, -20.0_dp), 18.0_dp))/NBASIS
    end do
    do k = 1, 12
      rates(k) = softplus_f(raw_rates(k)) + 1.0e-8_dp
      args(k) = u*rates(k) + bias(k)
      sig(k) = sigmoid_f(args(k))
    end do
    A = coef(1) + coef(2)*u + coef(3)*u*u + strict_convexity*u &
      + sum(gain*rates*sig)
    base_phi = sigma_ref*scl*(coef(1)*u + 0.5_dp*coef(2)*u*u &
             + coef(3)*u**3/3.0_dp + sum(gain*softplus_f(args)) &
             - sum(gain*softplus_f(bias)) + 0.5_dp*strict_convexity*u*u)

    ! Stage-6 multiplier
    gg = int(branch_group(branch)) + 1
    do k = 1, 4
      values(k) = max(feats(k), 0.0_dp)
      cclip(k) = feats(k) > 0.0_dp
      norm(k) = values(k)/state_scales(k)
    end do
    ns(1) = logtau/3.0_dp
    rs(1) = logtau/3.0_dp
    do k = 1, 4
      ns(k + 1) = log(1.0_dp + norm(k))
      rs(k + 1) = 0.0_dp
    end do
    do k = 1, 12
      hs(k) = tanh(snet_b1(k) + sum(snet_w1(k, :)*ns))
    end do
    warg = (state_net(ns) - state_net(rs))/3.0_dp
    delta = 3.0_dp*tanh(warg)
    sech2 = 1.0_dp - tanh(warg)**2
    exponent = 0.0_dp
    do k = 1, 4
      exponent = exponent - state_gate(gg, k)*norm(k)
    end do
    exponent = exponent + delta
    exclip = (exponent > -30.0_dp) .and. (exponent < 0.0_dp)
    mult = reference_calibration(branch) &
         * exp(min(max(exponent, -30.0_dp), 0.0_dp))

    ! Stage-6 positive nonlinear correction
    nl1 = nonlinear_coeff(gg, 1)
    nl2 = nonlinear_coeff(gg, 2)
    corr_phi = sigma_ref*scl*(0.5_dp*nl1*u*u + nl2*u**3/3.0_dp)
    corr_slope = nl1*u + nl2*u*u
    phi = mult*base_phi + corr_phi
    coef_flow = (mult*A + corr_slope)/(scl*sigma_ref)

    ! ---------------- analytic partial derivatives (stage 4) ----------------
    ! dA/du: polynomial part + basis part dsig/du = sig(1-sig)*rates
    dAdu = coef(2) + 2.0_dp*u*coef(3) + strict_convexity
    do k = 1, 12
      dAdu = dAdu + gain(k)*rates(k)*rates(k)*sig(k)*(1.0_dp - sig(k))
    end do
    coef_u = (mult*dAdu + nl1 + 2.0_dp*u*nl2)/(scl*sigma_ref)
    if (uclip) coef_u = 0.0_dp   ! derivative of the clamped function

    ! snet output column weights for the delta backprop:
    ! s2col(f) = sum_k snet_w2(1,k) (1-hs_k^2) snet_w1(k,f+1)
    do f = 1, 4
      acc = 0.0_dp
      do k = 1, 12
        acc = acc + snet_w2(1, k)*(1.0_dp - hs(k)*hs(k))*snet_w1(k, f + 1)
      end do
      s2col(f) = acc
    end do

    do f = 1, 4
      ! encoder backprop: feats(1) never enters the encoder
      dxn = 0.0_dp
      if (f >= 2) dxn(f) = 1.0_dp/scales(f)
      do k = 1, 24
        dh1(k) = sp1(k)*sum(enc0_w(k, :)*dxn)
      end do
      do k = 1, 24
        dh2(k) = sp2(k)*sum(enc2_w(k, :)*dh1)
      end do
      do k = 1, 3
        dcoef(k) = 0.0_dp
        if (ccoef(k)) dcoef(k) = coef(k)*sum(coef_w(k, :)*dh2)
      end do
      do k = 1, 12
        dbias(k) = sum(bias_w(k, :)*dh2)
        dgain(k) = 0.0_dp
        if (cgain(k)) dgain(k) = gain(k)*sum(gain_w(k, :)*dh2)
      end do
      dA_f(f) = dcoef(1) + dcoef(2)*u + dcoef(3)*u*u
      do k = 1, 12
        dA_f(f) = dA_f(f) + rates(k)*(dgain(k)*sig(k) &
                + gain(k)*sig(k)*(1.0_dp - sig(k))*dbias(k))
      end do
      ! multiplier path
      dnrm(f) = 0.0_dp
      if (cclip(f)) dnrm(f) = 1.0_dp/state_scales(f)
      ddelta(f) = sech2*s2col(f)*dnrm(f)/(1.0_dp + norm(f))
      dmult(f) = 0.0_dp
      if (exclip) dmult(f) = mult*(-state_gate(gg, f)*dnrm(f) + ddelta(f))
      coef_f(f) = (dmult(f)*A + mult*dA_f(f))/(scl*sigma_ref)
    end do
  end subroutine flow_coefficient

  ! ---------------- full 3D response ----------------
  subroutine kernel_response(F, Fv, psi_eq, psi_b, psi_tot, sig_eq, &
                             sig_branch, sigma_dev, X, flow, phi, diss, diss_tot)
    real(dp), intent(in)  :: F(3, 3), Fv(3, 3, 7)
    real(dp), intent(out) :: psi_eq, psi_b(7), psi_tot
    real(dp), intent(out) :: sig_eq(3, 3), sig_branch(3, 3, 7), sigma_dev(3, 3)
    real(dp), intent(out) :: X(3, 3, 7), flow(3, 3, 7), phi(7), diss(7)
    real(dp), intent(out) :: diss_tot
    real(dp) :: J, Jf, Fe(3, 3), Fbe(3, 3), Cbe(3, 3), Bbe(3, 3)
    real(dp) :: Fbar(3, 3), Cbar(3, 3), Be(3, 3), I1, I2, I1e
    real(dp) :: w1, w2, psi, M(3, 3), tau(3, 3), sig(3, 3)
    real(dp) :: Fvbar(3, 3), Cv(3, 3), feats(4), jin, coef, ph
    real(dp) :: cfu, cff(4)
    real(dp) :: ea1, eb1, ea2, eb2
    real(dp) :: sig_tot(3, 3)
    integer  :: b

    ! equilibrium part (isochoric extension of the Zhang-Li potential)
    J = det3(F)
    Jf = J**(-1.0_dp/3.0_dp)
    Fbar = Jf*F
    Cbar = matmul(transpose(Fbar), Fbar)
    call zhang_li_energy(Cbar, psi_eq, w1, w2, ea1, eb1, ea2, eb2)
    Be = matmul(Fbar, transpose(Fbar))
    call invariants3(Cbar, I1, I2)
    tau = 2.0_dp*w1*Be + 2.0_dp*w2*(I1*Be - matmul(Be, Be))
    sig_eq = tau/J

    sig_tot = sig_eq
    psi_tot = psi_eq
    diss_tot = 0.0_dp
    do b = 1, 7
      Fe = matmul(F, inv3(Fv(:, :, b)))
      J = det3(Fe)
      Jf = J**(-1.0_dp/3.0_dp)
      Fbe = Jf*Fe
      Cbe = matmul(transpose(Fbe), Fbe)
      call branch_energy(Cbe, b, psi, w1, w2, ea1, eb1, ea2, eb2)
      M = mandel_f(Cbe, w1, w2)
      X(:, :, b) = dev3(M)
      Bbe = matmul(Fbe, transpose(Fbe))
      call invariants3(Cbe, I1e, I2)
      tau = 2.0_dp*w1*Bbe + 2.0_dp*w2*(I1e*Bbe - matmul(Bbe, Bbe))
      sig = tau/J
      sig_branch(:, :, b) = sig
      sig_tot = sig_tot + sig
      psi_b(b) = psi
      psi_tot = psi_tot + psi

      ! dissipation features (isochoric measures)
      Fvbar = det3(Fv(:, :, b))**(-1.0_dp/3.0_dp)*Fv(:, :, b)
      Cv = matmul(transpose(Fvbar), Fvbar)
      feats(1) = I1 - 3.0_dp
      feats(2) = I1e - 3.0_dp
      feats(3) = Cv(1,1) + Cv(2,2) + Cv(3,3) - 3.0_dp
      feats(4) = hencky_dev(matmul(transpose(Fv(:, :, b)), Fv(:, :, b)))
      jin = 0.5_dp*sum(X(:, :, b)*X(:, :, b))/(sigma_ref*sigma_ref)
      call flow_coefficient(jin, b, feats, coef, ph, cfu, cff)
      flow(:, :, b) = coef*X(:, :, b)
      phi(b) = ph
      diss(b) = sum(X(:, :, b)*flow(:, :, b))
      diss_tot = diss_tot + diss(b)
    end do
    sigma_dev = dev3(sig_tot)
  end subroutine kernel_response

  ! ---------------- single-branch response (for the integrator) ----------------
  subroutine branch_response(F, Fv_b, branch, X, flow, phi, diss, psi_b)
    integer, intent(in)   :: branch
    real(dp), intent(in)  :: F(3, 3), Fv_b(3, 3)
    real(dp), intent(out) :: X(3, 3), flow(3, 3), phi, diss, psi_b
    real(dp) :: Fe(3, 3), J_e, Jf, Fbe(3, 3), Cbe(3, 3)
    real(dp) :: w1, w2, M(3, 3), Fbar(3, 3), Cbar(3, 3)
    real(dp) :: Fvbar(3, 3), Cv(3, 3), feats(4), jin, coef
    real(dp) :: cfu, cff(4)
    real(dp) :: ea1, eb1, ea2, eb2

    Fbar = det3(F) ** (-1.0_dp/3.0_dp) * F
    Cbar = matmul(transpose(Fbar), Fbar)
    Fe = matmul(F, inv3(Fv_b))
    J_e = det3(Fe)
    Jf = J_e ** (-1.0_dp/3.0_dp)
    Fbe = Jf * Fe
    Cbe = matmul(transpose(Fbe), Fbe)
    call branch_energy(Cbe, branch, psi_b, w1, w2, ea1, eb1, ea2, eb2)
    M = mandel_f(Cbe, w1, w2)
    X = dev3(M)
    Fvbar = det3(Fv_b) ** (-1.0_dp/3.0_dp) * Fv_b
    Cv = matmul(transpose(Fvbar), Fvbar)
    feats(1) = Cbar(1,1) + Cbar(2,2) + Cbar(3,3) - 3.0_dp
    feats(2) = Cbe(1,1) + Cbe(2,2) + Cbe(3,3) - 3.0_dp
    feats(3) = Cv(1,1) + Cv(2,2) + Cv(3,3) - 3.0_dp
    feats(4) = hencky_dev(matmul(transpose(Fv_b), Fv_b))
    jin = 0.5_dp*sum(X*X)/(sigma_ref*sigma_ref)
    call flow_coefficient(jin, branch, feats, coef, phi, cfu, cff)
    flow = coef*X
    diss = sum(X*flow)
  end subroutine branch_response

  pure function inv3(A) result(B)
    real(dp), intent(in) :: A(3, 3)
    real(dp) :: B(3, 3), d
    d = det3(A)
    B(1, 1) = (A(2,2)*A(3,3) - A(2,3)*A(3,2))/d
    B(2, 1) = (A(2,3)*A(3,1) - A(2,1)*A(3,3))/d
    B(3, 1) = (A(2,1)*A(3,2) - A(2,2)*A(3,1))/d
    B(1, 2) = (A(1,3)*A(3,2) - A(1,2)*A(3,3))/d
    B(2, 2) = (A(1,1)*A(3,3) - A(1,3)*A(3,1))/d
    B(3, 2) = (A(1,2)*A(3,1) - A(1,1)*A(3,2))/d
    B(1, 3) = (A(1,2)*A(2,3) - A(1,3)*A(2,2))/d
    B(2, 3) = (A(1,3)*A(2,1) - A(1,1)*A(2,3))/d
    B(3, 3) = (A(1,1)*A(2,2) - A(1,2)*A(2,1))/d
  end function inv3

end module edpicnn_native
