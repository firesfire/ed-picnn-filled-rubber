"""Extract creep displacement increments or fifth-cycle stiffness with Abaqus Python."""
from __future__ import print_function

import argparse
import csv
import re
import sys

import numpy as np


def reference_history(step, channels):
    matches = [region.historyOutputs for region in step.historyRegions.values()
               if all(key in region.historyOutputs for key in channels)]
    if len(matches) != 1:
        raise ValueError("Expected one reference-point history in " + step.name)
    return [np.asarray(matches[0][key].data, dtype=float) for key in channels]


def creep_rows(steps):
    ramp, = reference_history(steps["Creep-ramp"], ("U3",))
    hold, = reference_history(steps["Creep-hold"], ("U3",))
    return ("hold_time_s", "displacement_increment_mm"), zip(hold[:, 0], hold[:, 1] - ramp[-1, 1])


def stiffness_row(step, frequency):
    disp, force = reference_history(step, ("U3", "RF3"))
    if not np.array_equal(disp[:, 0], force[:, 0]):
        raise ValueError("U3 and RF3 must share the same time samples")
    phase = disp[:, 0] * frequency
    tolerance = 1e-6
    idx = (phase >= 4 - tolerance) & (phase <= 5 + tolerance)
    phase = phase[idx]
    if len(phase) < 4 or abs(phase[0] - 4) > tolerance or abs(phase[-1] - 5) > tolerance:
        raise ValueError("Both endpoints of the fifth cycle are required")
    design = np.column_stack((np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)))
    signals = np.column_stack((disp[idx, 1], force[idx, 1]))
    a, b = np.linalg.lstsq(design, signals - signals.mean(axis=0), rcond=-1)[0]
    u1, f1 = a + 1j*b
    if abs(u1) == 0:
        raise ValueError("Zero displacement amplitude")
    k = f1/u1
    return ("frequency_Hz", "K_storage_N_per_mm", "K_loss_N_per_mm"), [(frequency, k.real, k.imag)]


if __name__ == "__main__":
    from odbAccess import openOdb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("odb")
    parser.add_argument("output")
    args = parser.parse_args()
    odb = openOdb(path=args.odb, readOnly=True)
    try:
        dynamic = [(name, re.match(r"^Dynamic-z-(\d+(?:p\d+)?)Hz$", name, re.I))
                   for name in odb.steps.keys()]
        dynamic = [(name, float(match.group(1).replace("p", ".")))
                   for name, match in dynamic if match]
        if len(dynamic) == 1:
            name, frequency = dynamic[0]
            header, rows = stiffness_row(odb.steps[name], frequency)
        elif not dynamic and "Creep-hold" in odb.steps:
            header, rows = creep_rows(odb.steps)
        else:
            raise ValueError("Expected a creep job or a single-frequency restart job")
        options = {"mode": "wb"} if sys.version_info[0] < 3 else {"mode": "w", "newline": ""}
        with open(args.output, **options) as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    finally:
        odb.close()
