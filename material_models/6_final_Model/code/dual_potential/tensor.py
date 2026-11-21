"""Tensor utilities and incompressible loading charts."""

from __future__ import annotations

import torch


def eye_like(A: torch.Tensor) -> torch.Tensor:
    return torch.eye(3, dtype=A.dtype, device=A.device)


def sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def dev(A: torch.Tensor) -> torch.Tensor:
    return A - torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)[..., None, None] * eye_like(A) / 3.0


def isochoric_metric(C: torch.Tensor) -> torch.Tensor:
    C = sym(C)
    determinant = torch.linalg.det(C)
    if bool(torch.any(determinant <= 0).detach().cpu()):
        raise ValueError("metric must be positive definite")
    return C * determinant.pow(-1.0 / 3.0)[..., None, None]


def invariants(C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    I1 = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
    I2 = 0.5 * (I1.square() - torch.einsum("...ij,...ji->...", C, C))
    I3 = torch.linalg.det(C)
    return I1, I2, I3


def spectral_power(A: torch.Tensor, exponent: torch.Tensor | float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(sym(A))
    values = values.clamp_min(1.0e-14)
    power = torch.as_tensor(exponent, dtype=A.dtype, device=A.device)
    return (vectors * values.pow(power)[..., None, :]) @ vectors.transpose(-1, -2)


def deformation_gradient(value: torch.Tensor | float, mode: str,
                         secondary: torch.Tensor | float = 0.0, *,
                         dtype: torch.dtype = torch.float64,
                         device: torch.device | str = "cpu") -> torch.Tensor:
    x = torch.as_tensor(value, dtype=dtype, device=device)
    y = torch.as_tensor(secondary, dtype=dtype, device=device)
    x, y = torch.broadcast_tensors(x, y)
    z, o = torch.zeros_like(x), torch.ones_like(x)
    if mode == "uniaxial":
        lam = 1.0 + x
        if bool(torch.any(lam <= 0).detach().cpu()):
            raise ValueError("uniaxial stretch must be positive")
        return torch.diag_embed(torch.stack((lam, lam.pow(-0.5), lam.pow(-0.5)), -1))
    if mode == "shear":
        return torch.stack((torch.stack((o, x, z), -1),
                            torch.stack((z, o, z), -1),
                            torch.stack((z, z, o), -1)), -2)
    if mode == "planar":
        lam = 1.0 + x
        return torch.diag_embed(torch.stack((lam, o, lam.reciprocal()), -1))
    if mode == "equibiaxial":
        lam = 1.0 + x
        return torch.diag_embed(torch.stack((lam, lam, lam.pow(-2.0)), -1))
    if mode == "mixed":
        return deformation_gradient(y, "shear", dtype=dtype, device=device) @ deformation_gradient(
            x, "uniaxial", dtype=dtype, device=device)
    raise ValueError(f"unsupported loading mode: {mode}")


def rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
    a = axis / torch.linalg.vector_norm(axis)
    x, y, z = a.unbind()
    O = torch.zeros((), dtype=a.dtype, device=a.device)
    K = torch.stack((torch.stack((O, -z, y)), torch.stack((z, O, -x)), torch.stack((-y, x, O))))
    I = torch.eye(3, dtype=a.dtype, device=a.device)
    theta = torch.as_tensor(angle, dtype=a.dtype, device=a.device)
    return I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def pressure_project(cauchy: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "uniaxial":
        pressure = 0.5 * (cauchy[..., 1, 1] + cauchy[..., 2, 2])
    else:
        pressure = cauchy[..., 2, 2]
    return cauchy - pressure[..., None, None] * eye_like(cauchy)


def first_piola(cauchy: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return torch.linalg.det(F)[..., None, None] * cauchy @ torch.linalg.inv(F).transpose(-1, -2)


def symmetric_deviatoric_basis(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    r2, r6 = 2.0**0.5, 6.0**0.5
    return torch.tensor([
        [[1/r2, 0, 0], [0, -1/r2, 0], [0, 0, 0]],
        [[1/r6, 0, 0], [0, 1/r6, 0], [0, 0, -2/r6]],
        [[0, 1/r2, 0], [1/r2, 0, 0], [0, 0, 0]],
        [[0, 0, 1/r2], [0, 0, 0], [1/r2, 0, 0]],
        [[0, 0, 0], [0, 0, 1/r2], [0, 1/r2, 0]],
    ], dtype=dtype, device=device)
