"""Transformer variational autoencoder for periodic crystal structures."""

from typing import Any, Dict

import torch
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from src.models.components.kabsch_utils import random_rotation_matrix


DATASET_TO_IDX = {"mp20": 0}


class DiagonalGaussianDistribution:
    """Diagonal Gaussian parameterized by concatenated mean and log variance."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None) -> torch.Tensor:
        if self.deterministic:
            return self.mean.new_zeros(self.mean.shape[0])
        if other is None:
            return 0.5 * torch.sum(
                self.mean.square() + self.var - 1.0 - self.logvar, dim=1
            )
        return 0.5 * torch.sum(
            (self.mean - other.mean).square() / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=1,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class VariationalAutoencoderLitModule(LightningModule):
    """Per-atom Transformer VAE used as the first stage of latent flow matching."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_dim: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: int,
        loss_weights: Dict,
        augmentations: DictConfig,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = torch.nn.Linear(self.encoder.d_model, 2 * latent_dim, bias=False)
        self.post_quant_conv = torch.nn.Linear(latent_dim, self.decoder.d_model, bias=False)

        self.loss_weights_atom_types = self._weight_tensor(loss_weights, "loss_atom_types")
        self.loss_weights_lengths = self._weight_tensor(loss_weights, "loss_lengths")
        self.loss_weights_angles = self._weight_tensor(loss_weights, "loss_angles")
        self.loss_weights_frac_coords = self._weight_tensor(loss_weights, "loss_frac_coords")
        self.loss_weights_pos = self._weight_tensor(loss_weights, "loss_pos")
        self.loss_weights_kl = self._weight_tensor(loss_weights, "loss_kl")

    @staticmethod
    def _weight_tensor(loss_weights: Dict, name: str) -> torch.nn.Parameter:
        values = torch.tensor(list(loss_weights[name].values()), dtype=torch.float32)
        return torch.nn.Parameter(values, requires_grad=False)

    def encode(self, batch: Data) -> Dict[str, torch.Tensor]:
        encoded_batch = self.encoder(batch)
        encoded_batch["moments"] = self.quant_conv(encoded_batch["x"])
        encoded_batch["posterior"] = DiagonalGaussianDistribution(
            encoded_batch["moments"]
        )
        return encoded_batch

    def decode(self, encoded_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        decoder_batch = dict(encoded_batch)
        decoder_batch["x"] = self.post_quant_conv(decoder_batch["x"])
        return self.decoder(decoder_batch)

    def forward(self, batch: Data, sample_posterior: bool = True):
        encoded_batch = self.encode(batch)
        posterior = encoded_batch["posterior"]
        encoded_batch["x"] = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(encoded_batch), encoded_batch

    def reconstruction_criterion(
        self, batch: Data, out: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        loss_atom_types = F.cross_entropy(
            out["atom_types"], batch.atom_types, reduction="none"
        )
        loss_lengths = F.mse_loss(
            out["lengths"], batch.lengths_scaled, reduction="none"
        ).mean(dim=1)
        loss_angles = F.mse_loss(
            out["angles"], batch.angles_radians, reduction="none"
        ).mean(dim=1)
        loss_frac_coords = F.mse_loss(
            out["frac_coords"], batch.frac_coords, reduction="none"
        ).mean(dim=1)

        pos_true = batch.pos / 10.0
        pos_mean_pred = scatter(out["pos"], batch.batch, dim=0, reduce="mean")[batch.batch]
        pos_mean_true = scatter(pos_true, batch.batch, dim=0, reduce="mean")[batch.batch]
        loss_pos = F.mse_loss(
            out["pos"] - pos_mean_pred,
            pos_true - pos_mean_true,
            reduction="none",
        ).mean(dim=1)
        return {
            "loss_atom_types": loss_atom_types,
            "loss_lengths": loss_lengths,
            "loss_angles": loss_angles,
            "loss_frac_coords": loss_frac_coords,
            "loss_pos": loss_pos,
        }

    def criterion(
        self,
        batch: Data,
        encoded_batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        reconstruction = self.reconstruction_criterion(batch, out)
        loss_kl = encoded_batch["posterior"].kl()
        weights = {
            "loss_atom_types": self.loss_weights_atom_types[batch.dataset_idx[batch.batch]],
            "loss_lengths": self.loss_weights_lengths[batch.dataset_idx],
            "loss_angles": self.loss_weights_angles[batch.dataset_idx],
            "loss_frac_coords": self.loss_weights_frac_coords[batch.dataset_idx[batch.batch]],
            "loss_pos": self.loss_weights_pos[batch.dataset_idx[batch.batch]],
            "loss_kl": self.loss_weights_kl[batch.dataset_idx[batch.batch]],
        }
        scaled = {
            name: weights[name] * value
            for name, value in {**reconstruction, "loss_kl": loss_kl}.items()
        }
        total = sum(value.mean() for value in scaled.values())
        return {"loss": total, **scaled}

    def _augment(self, batch: Data) -> Dict[str, torch.Tensor]:
        originals: Dict[str, torch.Tensor] = {}
        sample_is_periodic = batch.dataset_idx == DATASET_TO_IDX["mp20"]
        node_is_periodic = sample_is_periodic[batch.batch]

        if self.hparams.augmentations.frac_coords and node_is_periodic.any():
            translation = torch.normal(
                torch.abs(batch.lengths.mean(dim=0)),
                torch.abs(batch.lengths.std(dim=0).nan_to_num(1e-8)),
            ) / 2
            batch.pos = batch.pos + translation
            inverse_cells = torch.linalg.inv(batch.cell[batch.batch][node_is_periodic])
            batch.frac_coords[node_is_periodic] = (
                torch.einsum("bi,bij->bj", batch.pos[node_is_periodic], inverse_cells) % 1.0
            )

        if self.hparams.augmentations.pos:
            rotation = random_rotation_matrix(validate=True, device=self.device)
            batch.pos = batch.pos @ rotation.T
            batch.cell = batch.cell @ rotation.T

        noise_fraction = float(self.hparams.augmentations.noise)
        if noise_fraction > 0.0:
            originals = {
                "atom_types": batch.atom_types.clone(),
                "pos": batch.pos.clone(),
                "frac_coords": batch.frac_coords.clone(),
            }
            total_atoms = int(batch.num_atoms.sum())
            count = int(total_atoms * noise_fraction)
            atom_indices = torch.randperm(total_atoms, device=self.device)[:count]
            batch.atom_types[atom_indices] = 0
            position_indices = torch.randperm(total_atoms, device=self.device)[:count]
            batch.pos[position_indices] += 0.1 * torch.randn_like(batch.pos[position_indices])
            if node_is_periodic.any():
                inverse_cells = torch.linalg.inv(batch.cell[batch.batch][node_is_periodic])
                batch.frac_coords[node_is_periodic] = (
                    torch.einsum("bi,bij->bj", batch.pos[node_is_periodic], inverse_cells)
                    % 1.0
                )
        return originals

    @staticmethod
    def _restore(batch: Data, originals: Dict[str, torch.Tensor]) -> None:
        for name, value in originals.items():
            setattr(batch, name, value)

    def _shared_step(self, batch: Data, stage: str, augment: bool) -> torch.Tensor:
        originals = self._augment(batch) if augment else {}
        out, encoded_batch = self.forward(batch)
        self._restore(batch, originals)
        losses = self.criterion(batch, encoded_batch, out)
        for name, value in losses.items():
            self.log(
                f"{stage}/{name}",
                value.mean(),
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=name == "loss",
                sync_dist=stage != "train",
            )
        return losses["loss"]

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train", augment=True)

    def validation_step(self, batch: Data, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._shared_step(batch, "val", augment=False)

    def test_step(self, batch: Data, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._shared_step(batch, "test", augment=False)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": self.hparams.scheduler_frequency,
            },
        }
