"""Deterministic, confidence-labelled material-point loading histories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.35, "low": 0.08, "diagnostic": 0.0}
DIRECT_UNIAXIAL_LEVELS = np.array([0.25, 0.50, 0.75, 1.00])
DIRECT_SHEAR_LEVELS = np.array([0.20, 0.40, 0.60, 0.80])
DIAGNOSTIC_FAMILIES = {
    "small_amplitude_dma", "large_amplitude_cycle", "preloaded_cycle", "creep",
}


@dataclass
class PathSpec:
    trajectory_id: str
    family: str
    mode: str
    time_s: np.ndarray
    primary: np.ndarray
    secondary: np.ndarray
    control: str = "strain"
    target_stress_MPa: np.ndarray | None = None
    frequency_hz: float = 0.0
    amplitude: float = 0.0
    preload: float = 0.0
    rate_per_s: float = 0.2
    hold_duration_s: float = 0.0
    points_per_cycle: int = 0
    preconditioned: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def confidence(self) -> str:
        if self.family in DIAGNOSTIC_FAMILIES:
            return "diagnostic"
        if self.family.startswith("direct_relaxation_"):
            levels = DIRECT_UNIAXIAL_LEVELS if self.mode == "uniaxial" else DIRECT_SHEAR_LEVELS
            if np.min(np.abs(levels - np.max(self.primary))) < 1e-10 and abs(self.rate_per_s - 0.2) < 1e-10:
                return "high"
        if (
            self.family == "small_amplitude_dma"
            and self.mode == "shear"
            and self.amplitude <= 0.001 + 1e-12
            and self.preload == 0.0
            and 0.1 <= self.frequency_hz <= 100.0
        ):
            return "high"
        if (
            self.mode in ("uniaxial", "shear")
            and (
                self.family.startswith("direct_relaxation_")
                or self.family in ("interpolated_relaxation", "unload_reload", "multistage", "random_segment")
            )
            and not self.preconditioned
            and self.preload == 0.0
            and np.min(self.primary) >= -1e-12
            and np.max(self.primary) <= (1.0 if self.mode == "uniaxial" else 0.8) + 1e-12
            and self.hold_duration_s <= 1200.0
        ):
            return "medium"
        return "low"

    @property
    def confidence_weight(self) -> float:
        return CONFIDENCE_WEIGHTS[self.confidence]

    @property
    def diagnostic_only(self) -> bool:
        return self.family in DIAGNOSTIC_FAMILIES

    @property
    def supervision_masks(self) -> dict[str, bool]:
        if self.diagnostic_only:
            return {
                "flow": False, "rollout": False, "stress": False, "energy": False,
            }
        return {
            "flow": True,
            "rollout": True,
            "stress": self.confidence in ("high", "medium"),
            # The active T2 energy is a known low-fidelity prior and must not
            # hard-supervise a future learnable nonequilibrium free energy.
            "energy": False,
        }


class HistoryBuilder:
    def __init__(self, primary: float = 0.0, secondary: float = 0.0):
        self.t = [0.0]
        self.x = [float(primary)]
        self.y = [float(secondary)]

    def ramp(self, target: float, rate: float, secondary: float | None = None) -> "HistoryBuilder":
        start, start_y = self.x[-1], self.y[-1]
        stop_y = start_y if secondary is None else float(secondary)
        distance = max(abs(float(target) - start), abs(stop_y - start_y))
        if distance < 1e-14:
            return self
        duration = distance / max(float(rate), 1e-12)
        points = max(2, int(np.ceil(distance / 0.025)))
        t0 = self.t[-1]
        for index in range(1, points + 1):
            ratio = index / points
            self.t.append(t0 + ratio * duration)
            self.x.append(start + ratio * (float(target) - start))
            self.y.append(start_y + ratio * (stop_y - start_y))
        return self

    def hold(self, duration: float, points: int = 15) -> "HistoryBuilder":
        duration = float(duration)
        if duration <= 0.0:
            return self
        start = self.t[-1]
        early = min(0.025, duration / max(points, 2))
        offsets = np.unique(np.geomspace(early, duration, max(points, 2)))
        for offset in offsets:
            self.t.append(start + float(offset))
            self.x.append(self.x[-1])
            self.y.append(self.y[-1])
        return self

    def sinusoid(
        self,
        amplitude: float,
        frequency: float,
        cycles: int,
        points_per_cycle: int,
        offset: float = 0.0,
    ) -> "HistoryBuilder":
        start = self.t[-1]
        for step in range(1, cycles * points_per_cycle + 1):
            theta = 2.0 * np.pi * step / points_per_cycle
            self.t.append(start + step / (frequency * points_per_cycle))
            self.x.append(float(offset + amplitude * np.sin(theta)))
            self.y.append(self.y[-1])
        return self

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.asarray(self.t, float), np.asarray(self.x, float), np.asarray(self.y, float)


def _spec(index: int, family: str, mode: str, builder: HistoryBuilder, **kwargs) -> PathSpec:
    t, x, y = builder.arrays()
    return PathSpec(f"traj_{index:04d}", family, mode, t, x, y, **kwargs)


def build_formal_specs(seed: int = 20260825) -> list[PathSpec]:
    rng = np.random.default_rng(seed)
    specs: list[PathSpec] = []

    def add(family: str, mode: str, builder: HistoryBuilder, **kwargs) -> None:
        specs.append(_spec(len(specs) + 1, family, mode, builder, **kwargs))

    for mode, levels in (("uniaxial", DIRECT_UNIAXIAL_LEVELS), ("shear", DIRECT_SHEAR_LEVELS)):
        for level in levels:
            for rate in (0.15, 0.20, 0.25):
                for hold in (120.0, 1200.0):
                    add(
                        f"direct_relaxation_{mode}", mode,
                        HistoryBuilder().ramp(float(level), rate).hold(hold, 19),
                        rate_per_s=rate, hold_duration_s=hold,
                    )

    for mode, levels in (
        ("uniaxial", [0.15, 0.35, 0.45, 0.65, 0.85, 0.95]),
        ("shear", [0.10, 0.30, 0.50, 0.70]),
    ):
        for level in levels:
            for rate in (0.10, 0.15, 0.25, 0.30):
                hold = 90.0 if rate < 0.2 else 600.0
                add(
                    "interpolated_relaxation", mode,
                    HistoryBuilder().ramp(level, rate).hold(hold, 15),
                    rate_per_s=rate, hold_duration_s=hold,
                )

    for mode, peaks in (("uniaxial", [.35, .55, .75, .95]), ("shear", [.20, .35, .55, .75])):
        for peak in peaks:
            for rate in (.15, .25):
                history = (
                    HistoryBuilder().ramp(peak, rate).hold(2.0, 5)
                    .ramp(.25 * peak, rate).hold(1.5, 4)
                    .ramp(.85 * peak, rate).hold(20.0, 8)
                )
                add("unload_reload", mode, history, rate_per_s=rate, hold_duration_s=20.0)

    for mode, ceiling in (("uniaxial", 0.9), ("shear", 0.72)):
        for _ in range(10):
            levels = rng.uniform(.15, .95, 4) * ceiling
            history = HistoryBuilder()
            for level in levels:
                history.ramp(float(level), .20).hold(float(rng.choice([.3, 2., 12.])), 5)
            history.hold(40.0, 8)
            add("multistage", mode, history, hold_duration_s=40.0)

    for mode, ceiling in (("uniaxial", 0.92), ("shear", 0.76)):
        for _ in range(8):
            history = HistoryBuilder()
            for level in rng.uniform(.05, ceiling, 6):
                history.ramp(float(level), float(rng.uniform(.10, .30))).hold(.3, 3)
            add("random_segment", mode, history, rate_per_s=.20, hold_duration_s=.3)

    for mode, levels in (("planar", [.15, .30, .45, .60]), ("equibiaxial", [.10, .20, .35, .50])):
        for level in levels:
            for hold in (30.0, 300.0):
                add("multiaxial_relaxation", mode, HistoryBuilder().ramp(level, .15).hold(hold, 14),
                    rate_per_s=.15, hold_duration_s=hold)

    for compression in (-.05, -.10, -.20, -.30):
        for rate in (.10, .20):
            add("compression", "uniaxial", HistoryBuilder().ramp(compression, rate).hold(50.0, 12),
                rate_per_s=rate, hold_duration_s=50.0)

    for _ in range(12):
        axial = float(rng.uniform(.10, .45))
        shear = float(rng.uniform(.10, .50))
        history = (
            HistoryBuilder().ramp(axial, .15).hold(1., 4)
            .ramp(axial, .15, secondary=shear).hold(15., 7)
            .ramp(.4 * axial, .15, secondary=.2 * shear).hold(5., 5)
        )
        add("nonproportional_multiaxial", "mixed", history,
            rate_per_s=.15, hold_duration_s=15.0)

    for mode, peak in (("uniaxial", .60), ("shear", .45)):
        for scale in (.45, .60, .70, .85, 1., 1.2):
            history = HistoryBuilder().ramp(float(peak * scale), .18).hold(60., 12)
            add("continued_internal_state", mode, history,
                rate_per_s=.18, hold_duration_s=60., preconditioned=True)

    for level in (-.10, -.20, -.40, -.60, -.80):
        for rate in (.15, .25):
            add("symmetric_shear_prior", "shear",
                HistoryBuilder().ramp(level, rate).hold(60.0, 13),
                rate_per_s=rate, hold_duration_s=60.0)

    return specs


def build_diagnostic_specs(seed: int = 20260825) -> list[PathSpec]:
    """Generate known low-confidence domains without using them for supervision."""
    del seed
    specs: list[PathSpec] = []

    def add(family: str, mode: str, builder: HistoryBuilder, **kwargs) -> None:
        t, x, y = builder.arrays()
        specs.append(PathSpec(f"diag_{len(specs)+1:04d}", family, mode, t, x, y, **kwargs))

    for frequency in (.1, .3, 1., 3., 10., 30., 100.):
        for amplitude in (.0005, .001):
            add("small_amplitude_dma", "shear",
                HistoryBuilder().sinusoid(amplitude, frequency, 9, 32),
                frequency_hz=frequency, amplitude=amplitude, points_per_cycle=32)

    for frequency in (.5, 1., 5.):
        for amplitude in (.03, .05, .08):
            add("large_amplitude_cycle", "shear",
                HistoryBuilder().sinusoid(amplitude, frequency, 7, 28),
                frequency_hz=frequency, amplitude=amplitude, points_per_cycle=28)

    for preload in (.10, .30):
        for frequency in (1., 10.):
            for amplitude in (.005, .01):
                history = HistoryBuilder().ramp(preload, .2).hold(1.0, 5)
                history.sinusoid(amplitude, frequency, 7, 28, offset=preload)
                add("preloaded_cycle", "shear", history, frequency_hz=frequency,
                    amplitude=amplitude, preload=preload, points_per_cycle=28)

    for target in (.10, .20, .35, .50):
        builder = HistoryBuilder().ramp(target, .12).hold(180.0, 14)
        times, targets, secondary = builder.arrays()
        specs.append(PathSpec(
            f"diag_{len(specs)+1:04d}", "creep", "uniaxial", times,
            np.zeros_like(targets), secondary, control="stress",
            target_stress_MPa=targets, rate_per_s=.12, hold_duration_s=180.0,
        ))
    return specs


def select_pilot_specs(specs: list[PathSpec], maximum: int = 18) -> list[PathSpec]:
    selected: list[PathSpec] = []
    seen: set[tuple[str, str]] = set()
    for spec in specs:
        group = (spec.family, spec.mode)
        if group not in seen:
            selected.append(spec)
            seen.add(group)
    selected = selected[:maximum]
    selected_ids = {item.trajectory_id for item in selected}
    if len(selected) < maximum:
        remaining = [item for item in specs if item.trajectory_id not in selected_ids]
        if remaining:
            indices = np.linspace(0, len(remaining) - 1, maximum - len(selected), dtype=int)
            selected.extend(remaining[index] for index in indices)
    return selected


def classify_point(spec: PathSpec, primary: float, secondary: float) -> tuple[bool, float]:
    if spec.mode not in ("uniaxial", "shear"):
        return False, 1.0 + abs(float(secondary))
    if spec.mode == "uniaxial":
        distance = max(0.0, -float(primary)) + max(0.0, float(primary) - 1.0)
    else:
        distance = max(0.0, abs(float(primary)) - .8)
    if spec.preload:
        distance += .50 * abs(spec.preload)
    if spec.family == "large_amplitude_cycle":
        distance += max(0.0, spec.amplitude - .01) / .10
    if spec.family in ("creep", "continued_internal_state"):
        distance += .25
    return bool(distance <= 1e-12 and spec.confidence != "low"), float(distance)
