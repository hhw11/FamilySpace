"""Periodic-boundary-aware VAE reconstruction loss."""

import torch

from src.models.vae_module import VariationalAutoencoderLitModule


def wrapped_fractional_delta(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the minimum-image displacement in fractional coordinates."""
    return torch.remainder(left - right + 0.5, 1.0) - 0.5


class PBCVariationalAutoencoderLitModule(VariationalAutoencoderLitModule):
    """VAE whose fractional-coordinate loss respects periodic boundaries."""

    def reconstruction_criterion(self, batch, out):
        losses = super().reconstruction_criterion(batch, out)
        losses["loss_frac_coords"] = wrapped_fractional_delta(
            out["frac_coords"], batch.frac_coords
        ).square().mean(dim=-1)
        return losses
