"""
Physics-Informed Loss for Stationary Incompressible Brinkman Flow.

Implements the PINO loss function consistent with COMSOL's formulation
for simulating flow through porous media.
"""

from collections.abc import Callable

import torch
import wandb
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


def invert_kappa_2x2(kxx: torch.Tensor, kxy: torch.Tensor, kyy: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Invert 2x2 permeability tensor components.

    Parameters
    ----------
    kxx : torch.Tensor
        xx component of permeability tensor.
    kxy : torch.Tensor
        xy component of permeability tensor.
    kyy : torch.Tensor
        yy component of permeability tensor.
    eps : float, optional
        Small value to prevent division by zero.

    Returns
    -------
    inv_xx : torch.Tensor
        xx component of inverted tensor.
    inv_xy : torch.Tensor
        xy component of inverted tensor.
    inv_yy : torch.Tensor
        yy component of inverted tensor.

    """
    det = (kxx * kyy - kxy * kxy).clamp_min(eps)
    return kyy / det, -kxy / det, kxx / det


# ================================================================
# Physics-Informed Loss (fixed, COMSOL-consistent)
# ================================================================
class PINOLoss(nn.Module):
    """
    Physics-Informed Loss for Stationary Incompressible Brinkman Flow.

    Combines data loss with physics residuals based on the Brinkman equations.

    Parameters
    ----------
    data_loss : nn.Module
        Loss function for data fidelity (e.g., H1Loss, LpLoss).
    lambda_phys : float
        Weighting factor for the physics loss term.
    log_every : int, optional
        Frequency of logging physics loss components to W&B.

    Methods
    -------
    forward(pred, target, inputs)
        Compute the combined PINO loss.
    ------------------------------------------------------------------------

    Notes
    -----
    The physics residuals are computed based on the following equations:
        - Momentum equations in x and y directions
        - Continuity equation
    The loss encourages the model predictions to satisfy these equations
    in addition to fitting the training data.
    ------------------------------------------------------------------------
    MU_AIR : float
        Dynamic viscosity of air used in the Brinkman equations.

    """

    def __init__(
        self,
        data_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lambda_phys: float,
        log_every: int = 10,
    ) -> None:
        """
        Initialize the PINOLoss.

        Parameters
        ----------
        data_loss : nn.Module
            Loss function for data fidelity (e.g., H1Loss, LpLoss).
        lambda_phys : float
            Weighting factor for the physics loss term.
        log_every : int, optional
            Frequency of logging physics loss components to W&B.

        """
        super().__init__()
        self.data_loss = data_loss
        self.lambda_phys = lambda_phys
        self.log_every = log_every
        self._step = 0

    def forward(self, pred: torch.Tensor, target: torch.Tensor, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the combined PINO loss.

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions of shape (B, 3, H, W) corresponding to (p, u, v).
        target : torch.Tensor
            Ground truth tensor (not used in physics loss).
        inputs : dict[str, torch.Tensor]
            Dictionary containing input tensors:
                - "x": x-coordinates (B, H, W)
                - "y": y-coordinates (B, H, W)
                - "kappaxx": xx component of permeability tensor (B, H, W)
                - "kappaxy": xy component of permeability tensor (B, H, W)
                - "kappayy": yy component of permeability tensor (B, H, W)
                - "phi": porosity field (B, H, W)

        Returns
        -------
        torch.Tensor
            Combined PINO loss value.

        """
        # ------------------------------------------------------------
        # Outputs
        # ------------------------------------------------------------
        p = pred[:, 0]
        u = pred[:, 1]
        v = pred[:, 2]

        # ------------------------------------------------------------
        # Inputs
        # ------------------------------------------------------------
        x = inputs["x"]
        y = inputs["y"]
        kxx = inputs["kappaxx"]
        kxy = inputs["kappaxy"]
        kyy = inputs["kappayy"]
        phi = inputs["phi"].clamp_min(1e-6)

        dx = (x[:, 0, 1] - x[:, 0, 0]).abs().mean()
        dy = (y[:, 1, 0] - y[:, 0, 0]).abs().mean()

        # ------------------------------------------------------------
        # Gradients
        # ------------------------------------------------------------
        dpdx = grad_x(p, dx)
        dpdy = grad_y(p, dy)

        dudx = grad_x(u, dx)
        dudy = grad_y(u, dy)
        dvdx = grad_x(v, dx)
        dvdy = grad_y(v, dy)

        div_u = dudx + dvdy

        # ------------------------------------------------------------
        # Brinkman stress tensor
        # ------------------------------------------------------------
        coef = MU_AIR / phi

        Kxx = coef * (2.0 * dudx) - (2.0 / 3.0) * coef * div_u
        Kyy = coef * (2.0 * dvdy) - (2.0 / 3.0) * coef * div_u
        Kxy = coef * (dudy + dvdx)

        divKx = grad_x(Kxx, dx) + grad_y(Kxy, dy)
        divKy = grad_x(Kxy, dx) + grad_y(Kyy, dy)

        # ------------------------------------------------------------
        # Darcy drag
        # ------------------------------------------------------------
        inv_xx, inv_xy, inv_yy = invert_kappa_2x2(kxx, kxy, kyy)

        drag_x = MU_AIR * (inv_xx * u + inv_xy * v)
        drag_y = MU_AIR * (inv_xy * u + inv_yy * v)

        # ------------------------------------------------------------
        # Residuals
        # ------------------------------------------------------------
        Rx = -dpdx + divKx - drag_x
        Ry = -dpdy + divKy - drag_y
        Rc = div_u

        phys_loss = Rx.pow(2).mean() + Ry.pow(2).mean() + Rc.pow(2).mean()

        # ------------------------------------------------------------
        # W&B logging
        # ------------------------------------------------------------
        if wandb.run is not None and self._step % self.log_every == 0:
            wandb.log(
                {
                    "physics/Rx_l2": Rx.pow(2).mean().sqrt().item(),
                    "physics/Ry_l2": Ry.pow(2).mean().sqrt().item(),
                    "physics/div_u_l2": Rc.pow(2).mean().sqrt().item(),
                    "physics/phys_loss": phys_loss.item(),
                    "physics/lambda_phys": self.lambda_phys,
                },
                commit=False,
            )

        self._step += 1

        # ------------------------------------------------------------
        # Total loss
        # ------------------------------------------------------------
        data_loss = self.data_loss(pred, target)
        return data_loss + self.lambda_phys * phys_loss
