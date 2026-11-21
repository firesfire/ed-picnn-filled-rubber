from __future__ import annotations

import numpy as np

from comparison_core import (CurveTarget, ZhangLi, dma_design,
                             equilibrium_shear_modulus, equilibrium_stress,
                             qlv_piecewise_linear_basis)


class EquilibriumAnchoredQLV:
    """Seven-branch scalar QLV baseline with the measured loading ramp retained."""

    def __init__(self, zhang_li: ZhangLi, tau_s, weights):
        self.zhang_li = zhang_li
        self.tau_s = np.asarray(tau_s, float)
        self.weights = np.asarray(weights, float)
        if len(self.tau_s) != 7 or self.weights.shape != self.tau_s.shape:
            raise ValueError("T0 requires seven matching time scales and weights")
        if np.any(self.tau_s <= 0) or np.any(self.weights < 0):
            raise ValueError("T0 requires positive times and nonnegative weights")

    def history(self, time_s, strain, mode):
        equilibrium = equilibrium_stress(mode, strain, self.zhang_li)
        basis = qlv_piecewise_linear_basis(time_s, equilibrium, self.tau_s)
        return np.asarray(equilibrium) + basis @ self.weights

    def target_basis(self, target: CurveTarget):
        equilibrium = equilibrium_stress(target.mode, target.history_strain, self.zhang_li)
        full = qlv_piecewise_linear_basis(target.history_time_s, equilibrium, self.tau_s)
        return full[target.peak_history_index:]

    def complex_modulus(self, frequency_hz):
        storage_basis, loss_basis = dma_design(frequency_hz, self.tau_s)
        g0 = equilibrium_shear_modulus(self.zhang_li)
        return g0 * (1.0 + storage_basis @ self.weights), g0 * (loss_basis @ self.weights)

    def relaxation_kernel(self, time_s):
        time_s = np.asarray(time_s, float)
        return 1.0 + np.exp(-time_s[:, None] / self.tau_s[None, :]) @ self.weights

