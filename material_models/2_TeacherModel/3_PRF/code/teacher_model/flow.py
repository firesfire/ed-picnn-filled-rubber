import numpy as np
from scipy.special import expit


def dev(A):
    return A - np.trace(A) / 3 * np.eye(3)


def equivalent_force(X):
    return float(np.sqrt(max(0, 1.5 * np.tensordot(X, X))))


def equivalent_hencky(A):
    C = 0.5 * (A.T @ A + (A.T @ A).T)
    lam, Q = np.linalg.eigh(C)
    lam = np.maximum(lam, 1e-30)
    E = (Q * (0.5 * np.log(lam))) @ Q.T
    E = dev(E)
    return float(np.sqrt(max(0, 2 * np.tensordot(E, E) / 3)))


def viscous_hardening(Fv, p):
    h = equivalent_hencky(Fv)
    return h, float((1 + h / p.h0) ** p.c)


def flow_rate(X, Fv, eta, p, activation=1.0, drive_scale=1.0,
              modulus_scale=1.0, branch_tau_s=1.0):
    X = dev(0.5 * (X + X.T))
    q = equivalent_force(X)
    h, a = viscous_hardening(Fv, p)
    A = max(float(activation), 1.0)
    linear = 1.0 / (eta * A)
    if p.flow_form in ("timescale_preserving", "timescale_hill", "bounded_additive"):
        rho = np.sqrt(q * q + (p.q0_MPa * drive_scale) ** 2)
        normalized_force = rho / max(float(drive_scale), 1.0e-14)
        if p.flow_form in ("timescale_hill", "bounded_additive"):
            log_ratio = np.log(max(normalized_force, 1e-300) / (p.s0_MPa * A))
            z = float(expit((p.m - 1.0) * log_ratio))
        else:
            z = (normalized_force / (p.s0_MPa * A)) ** (p.m - 1.0)
        if p.flow_form == "bounded_additive":
            log_tau_ratio = np.log(max(branch_tau_s, 1e-300) / p.tau_transition_s)
            tau_gate = float(expit(p.tau_gate_exponent * log_tau_ratio))
            additive = p.nonlinear_multiplier * tau_gate * a * z / max(
                3.0 * drive_scale * modulus_scale, 1e-14)
            coef = linear + additive
            multiplier = coef / linear
            power = additive
        else:
            multiplier = 1.0 + p.nonlinear_multiplier * a * z
            power = multiplier - 1.0
            coef = linear * multiplier
    else:
        rho = np.sqrt(q * q + p.q0_MPa**2)
        power = p.gamma0_per_s * a * rho ** (p.m - 1) / (p.s0_MPa**p.m * A**p.m)
        multiplier = 1.0 + power / linear
        coef = linear + power
    Dv = 1.5 * coef * X
    return Dv, {
        "q_MPa": q,
        "h": h,
        "hardening": a,
        "flow_multiplier": multiplier,
        "tau_gate": locals().get("tau_gate", 1.0),
        "activation": A,
        "gamma_dot_per_s": 1.5 * coef * q,
        "dissipation_MPa_per_s": float(np.tensordot(X, Dv)),
    }
