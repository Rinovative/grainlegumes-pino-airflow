"""
Physics sampling utilities for PINO losses.

Provides a unified interface to evaluate physics residuals in three modes:

1) "grid"
   - Use the native training grid (status quo).
   - No interpolation, no resampling.

2) "upsampled_grid"
   - Evaluate physics on a uniformly refined grid (integer upsampling factor).
   - Fields are interpolated to the finer grid.
   - Still grid-based (NOT fully functional PINO), but numerically stronger.

3) "random"
   - Sample physics residuals at arbitrary points in the domain.
   - Requires interpolation + autograd for derivatives.
   - This is the *functional* PINO mode.

This module is intentionally independent of the concrete PDE
(Brinkman, Navier-Stokes, etc.) and can be reused by both
spectral and physical-space PINO losses.
"""

from __future__ import annotations

from typing import Literal

import torch


# ================================================================
# Helpers
# ================================================================
def _make_uniform_grid(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract grid spacings and bounds from coordinate fields.

    Parameters
    ----------
    x, y : torch.Tensor
        Coordinate fields of shape (B, H, W)

    Returns
    -------
    xmin, xmax, ymin, ymax : torch.Tensor
        Scalars (per batch) defining domain bounds

    """
    xmin = x.amin(dim=(-2, -1))
    xmax = x.amax(dim=(-2, -1))
    ymin = y.amin(dim=(-2, -1))
    ymax = y.amax(dim=(-2, -1))
    return xmin, xmax, ymin, ymax


def _interp_fields_bilinear(
    f: torch.Tensor,
    H_new: int,
    W_new: int,
) -> torch.Tensor:
    """
    Bilinearly interpolate a scalar field.

    Parameters
    ----------
    f : torch.Tensor
        Tensor of shape (B, H, W)
    H_new, W_new : int
        Target resolution

    Returns
    -------
    torch.Tensor
        Interpolated tensor of shape (B, H_new, W_new)

    """
    f4 = f.unsqueeze(1)  # (B,1,H,W)
    f4 = torch.nn.functional.interpolate(f4, size=(H_new, W_new), mode="bilinear", align_corners=True)
    return f4[:, 0]


def _interp_fields_bilinear_multi(
    *fields: torch.Tensor,
    H_new: int,
    W_new: int,
) -> tuple[torch.Tensor, ...]:
    """
    Bilinearly interpolate multiple scalar fields.

    Parameters
    ----------
    *fields : torch.Tensor
        Tensors of shape (B, H, W)
    H_new, W_new : int
        Target resolution

    Returns
    -------
    tuple(torch.Tensor, ...)
        Interpolated tensors of shape (B, H_new, W_new)

    """
    return tuple(_interp_fields_bilinear(f, H_new, W_new) for f in fields)


# ================================================================
# Main API
# ================================================================
def sample_physics(
    *,
    p: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    x_coord: torch.Tensor,
    y_coord: torch.Tensor,
    mode: Literal["grid", "upsampled_grid", "random"],
    upsample_factor: int = 1,
    n_points: int | None = None,
) -> dict[str, torch.Tensor | bool | None]:
    """
    Prepare fields and coordinates for physics evaluation.

    Parameters
    ----------
    p, u, v : torch.Tensor
        Physical fields of shape (B, H, W)
    x_coord, y_coord : torch.Tensor
        Coordinate fields of shape (B, H, W)
    mode : {"grid", "upsampled_grid", "random"}
        Physics sampling mode
    upsample_factor : int, optional
        Upsampling factor for "upsampled_grid" mode (default: 1)
    n_points : int, optional
        Number of random points per batch for "random" mode

    Returns
    -------
    dict
        Dictionary with keys:
            - p, u, v : sampled fields
            - x, y    : corresponding coordinates
            - dx, dy  : effective spacings (grid modes only)
            - requires_autograd : bool

    """
    B, H, W = p.shape
    device = p.device
    dtype = p.dtype

    # ------------------------------------------------------------
    # Mode 1: Native grid
    # ------------------------------------------------------------
    if mode == "grid":
        dx = (x_coord[:, 0, 1] - x_coord[:, 0, 0]).abs().mean()
        dy = (y_coord[:, 1, 0] - y_coord[:, 0, 0]).abs().mean()

        return {
            "p": p,
            "u": u,
            "v": v,
            "x": x_coord,
            "y": y_coord,
            "dx": dx,
            "dy": dy,
            "requires_autograd": False,
        }

    # ------------------------------------------------------------
    # Mode 2: Upsampled grid
    # ------------------------------------------------------------
    if mode == "upsampled_grid":
        if upsample_factor <= 1:
            msg = "upsampled_grid requires upsample_factor > 1"
            raise ValueError(msg)

        Hn = H * upsample_factor
        Wn = W * upsample_factor

        p_u, u_u, v_u = _interp_fields_bilinear_multi(p, u, v, H_new=Hn, W_new=Wn)

        xmin, xmax, ymin, ymax = _make_uniform_grid(x_coord, y_coord)

        xs = torch.linspace(0.0, 1.0, Wn, device=device, dtype=dtype)
        ys = torch.linspace(0.0, 1.0, Hn, device=device, dtype=dtype)

        xs = xmin[:, None, None] + (xmax - xmin)[:, None, None] * xs[None, None, :]
        ys = ymin[:, None, None] + (ymax - ymin)[:, None, None] * ys[None, :, None]

        x_u = xs.expand(B, Hn, Wn)
        y_u = ys.expand(B, Hn, Wn)

        dx = (xmax - xmin).mean() / (Wn - 1)
        dy = (ymax - ymin).mean() / (Hn - 1)

        return {
            "p": p_u,
            "u": u_u,
            "v": v_u,
            "x": x_u,
            "y": y_u,
            "dx": dx,
            "dy": dy,
            "requires_autograd": False,
        }

    # ------------------------------------------------------------
    # Mode 3: Random point sampling (functional PINO)
    # ------------------------------------------------------------
    if mode == "random":
        if n_points is None:
            msg = "random mode requires n_points"
            raise ValueError(msg)

        xmin, xmax, ymin, ymax = _make_uniform_grid(x_coord, y_coord)

        rx = torch.rand(B, n_points, device=device, dtype=dtype)
        ry = torch.rand(B, n_points, device=device, dtype=dtype)

        xs = xmin[:, None] + (xmax - xmin)[:, None] * rx
        ys = ymin[:, None] + (ymax - ymin)[:, None] * ry

        # Normalised grid coordinates for grid_sample
        gx = 2.0 * (xs - xmin[:, None]) / (xmax - xmin)[:, None] - 1.0
        gy = 2.0 * (ys - ymin[:, None]) / (ymax - ymin)[:, None] - 1.0

        grid = torch.stack((gx, gy), dim=-1).unsqueeze(2)  # (B,N,1,2)

        def _sample(f: torch.Tensor) -> torch.Tensor:
            f4 = f.unsqueeze(1)  # (B,1,H,W)
            out = torch.nn.functional.grid_sample(
                f4,
                grid,
                mode="bilinear",
                align_corners=True,
            )
            return out[:, 0, :, 0]  # (B,N)

        p_s = _sample(p)
        u_s = _sample(u)
        v_s = _sample(v)

        xs.requires_grad_(True)
        ys.requires_grad_(True)

        return {
            "p": p_s,
            "u": u_s,
            "v": v_s,
            "x": xs,
            "y": ys,
            "dx": None,
            "dy": None,
            "requires_autograd": True,
        }

    msg = f"Unknown physics sampling mode: {mode}"
    raise ValueError(msg)
