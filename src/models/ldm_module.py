"""Latent flow-matching model for periodic crystal generation."""

import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from hydra.utils import get_class
from lightning import LightningModule
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from src.models.components.kabsch_utils import random_rotation_matrix
from src.models.vae_module import VariationalAutoencoderLitModule


DATASET_TO_IDX = {"mp20": 0}


def _resolve_release_path(path: str) -> str:
    """Resolve a relative input from the working directory or package root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    code_root = Path(__file__).resolve().parents[2]
    for root in (Path.cwd(), code_root):
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    return str(candidate)


def _count_atom_sites_from_cif(cif: str) -> int:
    lines = cif.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        cursor = index + 1
        headers = []
        while cursor < len(lines) and lines[cursor].lstrip().startswith("_"):
            headers.append(lines[cursor].strip())
            cursor += 1
        if "_atom_site_type_symbol" in headers:
            count = 0
            while cursor < len(lines):
                line = lines[cursor].strip()
                if not line or line == "loop_" or line.startswith("_"):
                    break
                count += 1
                cursor += 1
            return count
        index = cursor
    return 0


def _read_crystal_csv_stats(csv_path: str):
    num_nodes = []
    spacegroups = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            num_atoms = _count_atom_sites_from_cif(row["cif"])
            if num_atoms > 0:
                num_nodes.append(num_atoms)
            value = row.get("spacegroup.number")
            if value not in (None, ""):
                try:
                    spacegroups.append(int(float(value)))
                except ValueError:
                    pass
    if not num_nodes:
        raise ValueError(f"No atom sites could be parsed from {csv_path}")
    node_counts = torch.bincount(torch.tensor(num_nodes, dtype=torch.long))
    if spacegroups:
        sg_counts = torch.bincount(
            torch.tensor(spacegroups, dtype=torch.long).clamp(0, 230), minlength=231
        )
    else:
        sg_counts = torch.zeros(231, dtype=torch.long)
        sg_counts[1] = 1
    return node_counts, sg_counts


class LatentDiffusionLitModule(LightningModule):
    """Train a DiT vector field in the frozen VAE latent space."""

    def __init__(
        self,
        autoencoder_ckpt: str,
        denoiser: torch.nn.Module,
        interpolant: DictConfig,
        augmentations: DictConfig,
        sampling: DictConfig,
        conditioning: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: int,
        compile: bool = False,
        autoencoder_strict_loading: bool = True,
        autoencoder_module_target: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        autoencoder_ckpt = _resolve_release_path(autoencoder_ckpt)
        autoencoder_cls = VariationalAutoencoderLitModule
        if autoencoder_module_target is not None:
            autoencoder_cls = get_class(autoencoder_module_target)
            if not issubclass(autoencoder_cls, VariationalAutoencoderLitModule):
                raise TypeError(
                    "autoencoder_module_target must name a VAE LightningModule subclass"
                )
        self.autoencoder = autoencoder_cls.load_from_checkpoint(
            autoencoder_ckpt,
            map_location="cpu",
            weights_only=False,
            strict=autoencoder_strict_loading,
        )
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()
        self.autoencoder_ckpt = autoencoder_ckpt
        self.denoiser = denoiser
        self.interpolant = interpolant

        reference_csv = getattr(sampling, "reference_cif_csv", None)
        if reference_csv is None:
            reference_csv = os.path.join(sampling.data_dir, "mp_20", "train.csv")
        node_counts, spacegroup_counts = _read_crystal_csv_stats(str(reference_csv))
        self.num_nodes_bincount = {
            "mp20": torch.nn.Parameter(node_counts, requires_grad=False)
        }
        self.spacegroups_bincount = {
            "mp20": torch.nn.Parameter(spacegroup_counts, requires_grad=False)
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.autoencoder.eval()
        return self

    def forward(self, batch: Data, sample_posterior: bool = True):
        with torch.no_grad():
            encoded_batch = self.autoencoder.encode(batch)
            posterior = encoded_batch["posterior"]
            encoded_batch["x"] = (
                posterior.sample() if sample_posterior else posterior.mode()
            )
            x_1, mask = to_dense_batch(encoded_batch["x"], encoded_batch["batch"])
            dense_batch = {
                "x_1": x_1,
                "token_mask": mask,
                "diffuse_mask": mask,
            }

        self.interpolant.device = x_1.device
        noisy_batch = self.interpolant.corrupt_batch(dense_batch)
        dataset_idx = batch.dataset_idx + 1
        spacegroup = batch.spacegroup
        if not self.hparams.conditioning.spacegroup:
            spacegroup = torch.zeros_like(spacegroup)

        x_sc = None
        if self.interpolant.self_condition and random.random() < self.interpolant.self_condition_prob:
            with torch.no_grad():
                x_sc = self.denoiser(
                    noisy_batch["x_t"],
                    noisy_batch["t"],
                    dataset_idx,
                    spacegroup,
                    mask,
                    None,
                )
        pred_x = self.denoiser(
            noisy_batch["x_t"],
            noisy_batch["t"],
            dataset_idx,
            spacegroup,
            mask,
            x_sc,
        )
        return pred_x, noisy_batch

    def criterion(
        self,
        batch: Data,
        noisy_batch: Dict[str, torch.Tensor],
        pred_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        del batch
        norm_scale = 1.0 - noisy_batch["t"].unsqueeze(-1).clamp(max=0.9)
        error = (noisy_batch["x_1"] - pred_x) / norm_scale
        mask = noisy_batch["token_mask"] & noisy_batch["diffuse_mask"]
        denominator = mask.sum(dim=-1).clamp_min(1) * pred_x.size(-1)
        x_loss = (error.square() * mask[..., None]).sum(dim=(-1, -2)) / denominator
        return {"loss": x_loss.mean(), "x_loss": x_loss}

    def _augment(self, batch: Data) -> None:
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

    def _shared_step(self, batch: Data, stage: str, augment: bool) -> torch.Tensor:
        if augment:
            with torch.no_grad():
                self._augment(batch)
        pred_x, noisy_batch = self.forward(batch)
        losses = self.criterion(batch, noisy_batch, pred_x)
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

    def sample_and_decode(
        self,
        num_nodes_bincount: torch.Tensor,
        spacegroups_bincount: Optional[torch.Tensor],
        batch_size: int,
        cfg_scale: float = 4.0,
        dataset_idx: int = 0,
    ):
        sample_lengths = torch.multinomial(
            num_nodes_bincount.float(), batch_size, replacement=True
        ).to(self.device)
        dataset_idx_tensor = torch.full(
            (batch_size,), dataset_idx + 1, dtype=torch.long, device=self.device
        )
        if not self.hparams.conditioning.spacegroup or spacegroups_bincount is None:
            spacegroup = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        else:
            spacegroup = torch.multinomial(
                spacegroups_bincount.float(), batch_size, replacement=True
            ).to(self.device)

        max_atoms = int(sample_lengths.max())
        token_mask = torch.arange(max_atoms, device=self.device)[None, :] < sample_lengths[:, None]
        samples = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=batch_size,
            num_tokens=max_atoms,
            emb_dim=self.denoiser.d_x,
            model=self.denoiser,
            dataset_idx=dataset_idx_tensor,
            spacegroup=spacegroup,
            cfg_scale=cfg_scale,
            token_mask=token_mask,
        )
        x = samples["clean_traj"][-1][token_mask]
        decoded_batch = {
            "x": x,
            "num_atoms": sample_lengths,
            "batch": torch.repeat_interleave(
                torch.arange(batch_size, device=self.device), sample_lengths
            ),
            "token_idx": (
                torch.cumsum(token_mask, dim=-1, dtype=torch.long) - 1
            )[token_mask],
        }
        return self.autoencoder.decode(decoded_batch), decoded_batch, samples

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.denoiser = torch.compile(self.denoiser)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.denoiser.parameters())
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
