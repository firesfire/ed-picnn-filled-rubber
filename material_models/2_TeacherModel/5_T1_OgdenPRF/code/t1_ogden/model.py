from __future__ import annotations

from ogden_prf_core import OgdenPRFModel, build_t1_ogden_parameters


def build_model(zhang_li: dict, flow: dict, tau_s, old_weights,
                Gneq_MPa: float, a_ogden: float) -> OgdenPRFModel:
    return OgdenPRFModel(build_t1_ogden_parameters(
        zhang_li, flow, tau_s, old_weights, Gneq_MPa, a_ogden))
