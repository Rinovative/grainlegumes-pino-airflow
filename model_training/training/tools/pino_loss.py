"""
Physics-Informed Loss for Stationary Incompressible Brinkman Flow.

Implements the PINO loss function consistent with COMSOL's formulation
for simulating flow through porous media.
"""

from collections.abc import Callable
from typing import Any

import torch
import wandb
from src.schema.schema_training import DEFAULT_INPUTS_2D, DEFAULT_OUTPUTS_2D
from torch import nn

# ================================================================
# Constants (from COMSOL)
# ================================================================
MU_AIR = 1.8139e-5  # Pa*s, air at T = 293.15 K (COMSOL)


# ================================================================
# Differential operators (uniform grid assumed)
# ================================================================
def grad_x(f: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of tensor `f` along the x-direction.

    Assumes uniform spacing `dx`.

    Parameters
    ----------
    f : torch.Tensor
        Input tensor of shape (B, H, W) or similar.
    dx : torch.Tensor
        Spacing in the x-direction (scalar).

    Returns
    -------
    torch.Tensor
        Gradient of `f` along the x-direction.

    """
    return torch.gradient(f, dim=-1)[0] / dx


def grad_y(f: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of tensor `f` along the y-direction.

    Assumes uniform spacing `dy`.

    Parameters
    ----------
    f : torch.Tensor
        Input tensor of shape (B, H, W) or similar.
    dy : torch.Tensor
        Spacing in the y-direction (scalar).

    Returns
    -------
    torch.Tensor
        Gradient of `f` along the y-direction.

    """
    return torch.gradient(f, dim=-2)[0] / dy


# ================================================================
# PINO Loss
# ================================================================
class PINOLoss(nn.Module):
    """
    Physics-Informed Loss for stationary incompressible Brinkman flow.

    Total loss:
        L = L_data
          + lambda_phys * L_phys
          + lambda_p    * L_pressure_bc

    where
        L_data         : Data loss (e.g., L2 or H1) between predicted and true outputs.
        L_phys         : Physics loss enforcing Brinkman equations.
        L_pressure_bc  : Loss enforcing pressure boundary conditions at inlet/outlet.

    Parameters
    ----------
    data_loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Function to compute data loss between predictions and targets.
    lambda_phys : float
        Weight for the physics loss term.
    lambda_p : float
        Weight for the pressure boundary loss term.
    in_normalizer : Any
        Normalizer for input data (must have `inverse_transform` method).
    out_normalizer : Any
        Normalizer for output data (must have `inverse_transform` method).
    log_every : int, optional
        Frequency of logging to wandb.

    Returns
    -------
    torch.Tensor
        Computed total loss.

    """

    def __init__(
        self,
        data_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lambda_phys: float,
        lambda_p: float,
        in_normalizer: Any,
        out_normalizer: Any,
        log_every: int = 10,
    ) -> None:
        """
        Initialize the PINOLoss.

        Parameters
        ----------
        data_loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            Function to compute data loss between predictions and targets.
        lambda_phys : float
            Weight for the physics loss term.
        lambda_p : float
            Weight for the pressure boundary loss term.
        in_normalizer : Any
            Normalizer for input data (must have `inverse_transform` method).
        out_normalizer : Any
            Normalizer for output data (must have `inverse_transform` method).
        log_every : int, optional
            Frequency of logging to wandb.

        """
        super().__init__()

        self.input_fields = DEFAULT_INPUTS_2D
        self.output_fields = DEFAULT_OUTPUTS_2D
        self.data_loss = data_loss
        self.lambda_phys = lambda_phys
        self.lambda_p = lambda_p
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.log_every = log_every
        self._step = 0

        # precompute indices once
        self._in_idx = {name: i for i, name in enumerate(self.input_fields)}
        self._out_idx = {name: i for i, name in enumerate(self.output_fields)}

    def forward(
        self,
        pred: torch.Tensor,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the PINO loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted outputs from the model (normalized).
        x : torch.Tensor
            Input tensor (normalized).
        y : torch.Tensor
            True output tensor (normalized).
        **_kwargs : Any
            Additional keyword arguments (not used).

        Returns
        -------
        torch.Tensor
            Computed total loss.

        """
        # ------------------------------------------------------------
        # 0) Denormalize
        # ------------------------------------------------------------
        x_phys = self.in_normalizer.inverse_transform(x)
        pred_phys = self.out_normalizer.inverse_transform(pred)
        y_phys = self.out_normalizer.inverse_transform(y)

        # ------------------------------------------------------------
        # 1) Outputs (physical)
        # ------------------------------------------------------------
        p = pred_phys[:, self._out_idx["p"]]
        u = pred_phys[:, self._out_idx["u"]]
        v = pred_phys[:, self._out_idx["v"]]

        # ------------------------------------------------------------
        # 2) Inputs (physical)
        # ------------------------------------------------------------
        x_coord = x_phys[:, self._in_idx["x"]]
        y_coord = x_phys[:, self._in_idx["y"]]

        kxx_log = x_phys[:, self._in_idx["kxx"]]
        kyy_log = x_phys[:, self._in_idx["kyy"]]
        kxy_rel = x_phys[:, self._in_idx["kxy"]]

        phi = x_phys[:, self._in_idx["phi"]].clamp_min(1e-6)
        p_bc = x_phys[:, self._in_idx["p_bc"]]

        kxx = torch.pow(10.0, kxx_log)
        kyy = torch.pow(10.0, kyy_log)

        # local scalar scale (same unit as kappa)
        K0 = torch.sqrt((kxx * kyy).clamp_min(1e-30))

        # dimensionless tensor components
        kxx_hat = kxx / K0  # = sqrt(kxx/kyy)
        kyy_hat = kyy / K0  # = sqrt(kyy/kxx)

        # dimensionless off-diagonal with physical clamp
        kxy_hat = kxy_rel.clamp(-0.99, 0.99)

        # dimensionless determinant (stable)
        det_hat = kxx_hat * kyy_hat - kxy_hat * kxy_hat  # = 1 - kxy_rel^2

        dx = (x_coord[:, 0, 1] - x_coord[:, 0, 0]).abs().mean()
        dy = (y_coord[:, 1, 0] - y_coord[:, 0, 0]).abs().mean()

        # ------------------------------------------------------------
        # 3) Gradients
        # ------------------------------------------------------------
        dpdx = grad_x(p, dx)
        dpdy = grad_y(p, dy)

        dudx = grad_x(u, dx)
        dudy = grad_y(u, dy)
        dvdx = grad_x(v, dx)
        dvdy = grad_y(v, dy)

        div_phi_u = grad_x(phi * u, dx) + grad_y(phi * v, dy)

        # ------------------------------------------------------------
        # 4) Brinkman operator
        # ------------------------------------------------------------
        coef = MU_AIR / phi

        Kxx = coef * (2.0 * dudx) - (2.0 / 3.0) * coef * (dudx + dvdy)
        Kyy = coef * (2.0 * dvdy) - (2.0 / 3.0) * coef * (dudx + dvdy)
        Kxy = coef * (dudy + dvdx)

        divKx = grad_x(Kxx, dx) + grad_y(Kxy, dy)
        divKy = grad_x(Kxy, dx) + grad_y(Kyy, dy)

        # invert only the dimensionless tensor, then rescale
        eps_det = 1e-4  # clamp on dimensionless det, not raw det
        det_hat_safe = det_hat.clamp_min(eps_det)

        invhat_xx = kyy_hat / det_hat_safe
        invhat_xy = -kxy_hat / det_hat_safe
        invhat_yy = kxx_hat / det_hat_safe

        # K^{-1} = (1/K0) * Khat^{-1}
        inv_xx = invhat_xx / K0
        inv_xy = invhat_xy / K0
        inv_yy = invhat_yy / K0

        drag_x = MU_AIR * (inv_xx * u + inv_xy * v)
        drag_y = MU_AIR * (inv_xy * u + inv_yy * v)

        Rx = -dpdx + divKx - drag_x
        Ry = -dpdy + divKy - drag_y
        Rc = div_phi_u

        phys_loss = Rx.pow(2).mean() + Ry.pow(2).mean() + Rc.pow(2).mean()

        # ------------------------------------------------------------
        # 5) Pressure boundary loss
        # ------------------------------------------------------------
        y_min = y_coord.amin(dim=(-2, -1), keepdim=True)
        y_max = y_coord.amax(dim=(-2, -1), keepdim=True)

        inlet_mask = (y_coord - y_min).abs() <= 0.5 * dy
        outlet_mask = (y_coord - y_max).abs() <= 0.5 * dy

        p_inlet_loss = (p[inlet_mask] - p_bc[inlet_mask]).pow(2).mean()
        p_outlet_loss = (p[outlet_mask].mean() - y_phys[:, 0][outlet_mask].mean()).pow(2)

        p_bc_loss = p_inlet_loss + p_outlet_loss

        # ------------------------------------------------------------
        # 6) Data loss (normalized)
        # ------------------------------------------------------------
        data_loss = self.data_loss(pred, y)

        # ------------------------------------------------------------
        # 7) Total loss
        # ------------------------------------------------------------
        total_loss = data_loss + self.lambda_phys * phys_loss + self.lambda_p * p_bc_loss

        # ------------------------------------------------------------
        # 8) Logging (diagnostic, non-intrusive)
        # ------------------------------------------------------------
        if wandb.run is not None and self._step % self.log_every == 0:
            total_loss_val = total_loss.item()

            # # keep a raw det only for logging
            # det_raw = (kxx * kyy) * det_hat

            wandb.log(
                {
                    # --- Total & loss components ---
                    "loss/total": total_loss_val,
                    "loss/data": data_loss.item(),
                    "loss/phys": phys_loss.item(),
                    "loss/p_bc": p_bc_loss.item(),
                    # --- Relative contributions ---
                    "loss/frac_loss_data": data_loss.item() / total_loss_val,
                    "loss/frac_loss_phys": (self.lambda_phys * phys_loss.item()) / total_loss_val,
                    "loss/frac_loss_p_bc": (self.lambda_p * p_bc_loss.item()) / total_loss_val,
                    # --- Physics diagnostics ---
                    "physics/Rx_l2": Rx.pow(2).mean().sqrt().item(),
                    "physics/Ry_l2": Ry.pow(2).mean().sqrt().item(),
                    "physics/div_phi_u_l2": Rc.pow(2).mean().sqrt().item(),
                    # # --- Kappa diagnostics ---
                    # "Kappa/kxx_log_min": kxx_log.min().item(),
                    # "Kappa/kxx_log_mean": kxx_log.mean().item(),
                    # "Kappa/kxx_log_max": kxx_log.max().item(),
                    # "Kappa/kyy_log_min": kyy_log.min().item(),
                    # "Kappa/kyy_log_mean": kyy_log.mean().item(),
                    # "Kappa/kyy_log_max": kyy_log.max().item(),
                    # "Kappa/kxy_rel_min": kxy_rel.min().item(),
                    # "Kappa/kxy_rel_mean": kxy_rel.mean().item(),
                    # "Kappa/kxy_rel_max": kxy_rel.max().item(),
                    # "Kappa/det_raw_phys_min": det_raw.min().item(),
                    # "Kappa/det_raw_phys_mean": det_raw.mean().item(),
                    # "Kappa/det_hat_min": det_hat.min().item(),
                    # "Kappa/det_hat_mean": det_hat.mean().item(),
                    # "Kappa/det_hat_frac_small": (det_hat < 1e-4).float().mean().item(),
                    # "Kappa/K0_log10_mean": torch.log10(K0).mean().item(),
                },
                commit=False,
            )

        self._step += 1
        return total_loss

    def set_normalizers(
        self,
        *,
        in_normalizer: Any,
        out_normalizer: Any,
    ) -> None:
        """
        Attach input and output normalizers after loss construction.

        This method is called by the training pipeline once the dataset
        and its associated normalizers are available. It enables defining
        the PINO loss object independently of the data loading stage.

        Parameters
        ----------
        in_normalizer : Any
            Input normalizer with an `inverse_transform` method.
        out_normalizer : Any
            Output normalizer with an `inverse_transform` method.

        """
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
