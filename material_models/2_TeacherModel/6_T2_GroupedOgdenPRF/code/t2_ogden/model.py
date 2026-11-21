from __future__ import annotations

from ogden_prf_core import OgdenPRFModel, build_t2_ogden_parameters


def build_model(zhang_li: dict, fast_flow: dict, slow_flow: dict, tau_s,
                old_weights, Gneq_MPa: float, a_ogden: float,
                transition_tau_s: float = 1.0) -> OgdenPRFModel:
    return OgdenPRFModel(build_t2_ogden_parameters(
        zhang_li, fast_flow, slow_flow, tau_s, old_weights,
        Gneq_MPa, a_ogden, transition_tau_s))
