"""Build the Abaqus user subroutine from an ED-PICNN checkpoint or parameter archive."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np


def parameter_names():
    for name in ("D", "mu", "B", "N", "beta"):
        yield "zhang_li_" + name, "zhang_li." + name
    yield "tau_s", "spectrum.tau_s"
    yield "branch_group", "spectrum.branch_group"
    for prefix in ("effective", "base"):
        name = prefix + "_branch_moduli"
        yield name, "spectrum." + name + "_MPa"
    for name in ("branch_scale", "raw_rates", "centers", "scales"):
        yield name, "base." + name
    for name in ("reference_calibration", "target_reference_slope", "state_scales"):
        yield name, "gate." + name
    yield "sigma_ref", "base.sigma_ref_MPa"
    yield "strict_convexity", "base.strict_convexity"
    for group in (0, 1):
        for name, key in (("weights", "weights"), ("rho_I1", "rho_I1"),
                          ("b", "b"), ("p", "p"), ("hardening", "hardening_MPa"),
                          ("hardening_rho", "hardening_rho_I1")):
            yield f"energy_g{group}_{name}", f"energy.group{group}.{key}"
    yield "state_gate", "gate.state_gate"
    yield "nonlinear_coeff", "gate.nonlinear_coefficients"
    layers = (("enc0_w", "enc0_b", "base.encoder0"), ("enc2_w", "enc2_b", "base.encoder2"),
              ("coef_w", "coef_b", "base.coefficient_head"), ("bias_w", "bias_b", "base.bias_head"),
              ("gain_w", "gain_b", "base.gain_head"), ("snet_w1", "snet_b1", "gate.state_net_fc1"),
              ("snet_w2", "snet_b2", "gate.state_net_fc2"))
    for weight, bias, layer in layers:
        yield weight, layer + ".weight"
        yield bias, layer + ".bias"


def declaration(name, array):
    values = [format(x, ".17e") + "_dp" for x in array.ravel(order="F")]
    prefix = "  real(dp), parameter :: " + name
    if array.ndim == 0:
        return prefix + " = " + values[0]
    dims = ",".join(map(str, array.shape))
    lines = [prefix + "(" + dims + ") = reshape((/ &"]
    for i in range(0, len(values), 3):
        lines.append("    " + ", ".join(values[i:i+3]))
    return lines[0] + "\n" + ", &\n".join(lines[1:]) + " /), (/" + dims + "/))"


def checkpoint_arrays(checkpoint, teacher, model_code):
    import torch
    if not (model_code / "frozen_material.py").is_file():
        raise FileNotFoundError(f"Missing frozen_material.py in {model_code}")
    sys.path.insert(0, str(model_code.resolve()))
    from frozen_material import load_frozen

    material = load_frozen(checkpoint, teacher)
    energy = material.nonequilibrium_energy
    flow = material.dissipation_potential
    base = flow.base
    arrays = {"zhang_li." + key: getattr(material.equilibrium_energy, key)
              for key in ("D", "mu", "B", "N", "beta")}
    arrays.update({
        "spectrum.tau_s": energy.tau_s, "spectrum.branch_group": energy.branch_group,
        "spectrum.base_branch_moduli_MPa": energy.branch_moduli_MPa,
        "spectrum.effective_branch_moduli_MPa": energy.effective_branch_moduli_MPa,
        "base.sigma_ref_MPa": base.sigma_ref, "base.strict_convexity": base.strict_convexity,
        "gate.state_gate": torch.nn.functional.softplus(flow.raw_state_gate),
        "gate.nonlinear_coefficients": torch.nn.functional.softplus(flow.raw_nonlinear),
    })
    for key in ("branch_scale", "raw_rates", "centers", "scales"):
        arrays["base." + key] = getattr(base, key)
    for key in ("reference_calibration", "target_reference_slope", "state_scales"):
        arrays["gate." + key] = getattr(flow, key)
    for group in (0, 1):
        for key, value in energy.group_physical(group).items():
            arrays[f"energy.group{group}.{key}"] = value
    layers = (("base.encoder0", base.encoder[0]), ("base.encoder2", base.encoder[2]),
              ("base.coefficient_head", base.coefficient_head), ("base.bias_head", base.bias_head),
              ("base.gain_head", base.gain_head), ("gate.state_net_fc1", flow.state_network[0]),
              ("gate.state_net_fc2", flow.state_network[2]))
    for key, layer in layers:
        arrays[key + ".weight"] = layer.weight
        arrays[key + ".bias"] = layer.bias
    return arrays


def build(archive, teacher=None, model_code=None):
    if archive.suffix.lower() == ".pt":
        if teacher is None:
            raise ValueError("A .pt checkpoint also requires --teacher with the T2 parameter JSON")
        code = model_code or Path(__file__).resolve().parents[1] / "material_models" / "6_final_Model" / "code"
        inputs = nullcontext(checkpoint_arrays(archive, teacher, code))
    else:
        inputs = np.load(archive, allow_pickle=False)
    lines = ["module frozen_params", "  implicit none", "  integer, parameter :: dp = kind(1.0d0)"]
    with inputs as data:
        for name, key in parameter_names():
            values = np.asarray(data[key], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite parameter {key}")
            lines.append(declaration(name, values))
    lines.append("end module frozen_params")
    modules = Path(__file__).resolve().parent / "umat"
    for name in ("edpicnn_native", "edpicnn_integrator", "edpicnn_tangent", "edpicnn_umat"):
        lines.append((modules / (name + ".f90")).read_text(encoding="utf-8-sig"))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="Final .pt checkpoint or frozen_weights.npz")
    parser.add_argument("output", type=Path, help="Output Fortran source (.f90)")
    parser.add_argument("--teacher", type=Path, help="T2 parameter JSON for the shared equilibrium energy")
    parser.add_argument("--model-code", type=Path, help="Directory containing frozen_material.py")
    args = parser.parse_args()
    source = build(args.weights, args.teacher, args.model_code)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(source)
