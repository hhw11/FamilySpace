"""Family-medoid source prior for MP20 latent flow matching."""

import random
from typing import Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from src.models.ldm_module import (
    LatentDiffusionLitModule,
    _resolve_release_path,
)


FAMILY_NAMES = (
    "Alkali",
    "Alkaline earth",
    "Boron group",
    "Carbon group",
    "Pnictogens",
    "Chalcogens",
    "Halogen",
    "Noble gas",
    "Transition metal",
    "Rare earth",
)


def _family_lookup() -> torch.Tensor:
    lookup = torch.full((119,), -1, dtype=torch.long)
    groups = (
        {1, 3, 11, 19, 37, 55, 87},
        {4, 12, 20, 38, 56, 88},
        {5, 13, 31, 49, 81, 113},
        {6, 14, 32, 50, 82, 114},
        {7, 15, 33, 51, 83, 115},
        {8, 16, 34, 52, 84, 116},
        {9, 17, 35, 53, 85, 117},
        {2, 10, 18, 36, 54, 86, 118},
        set(range(21, 31))
        | set(range(39, 49))
        | set(range(72, 81))
        | set(range(104, 113)),
        set(range(57, 72)) | set(range(89, 104)),
    )
    for family_id, atomic_numbers in enumerate(groups):
        lookup[list(atomic_numbers)] = family_id
    return lookup


class FamilyMedoidLatentDiffusionLitModule(LatentDiffusionLitModule):
    """Use atom-family medoids as paired, non-overlapping flow sources."""

    def __init__(
        self,
        *args,
        family_medoid_path: str,
        source_noise_scale: float = 0.1,
        source_radius_fraction: float = 0.2,
        family_probabilities: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            {
                "family_medoid_path": family_medoid_path,
                "source_noise_scale": source_noise_scale,
                "source_radius_fraction": source_radius_fraction,
                "family_probabilities": family_probabilities,
            },
            logger=False,
        )

        family_medoid_path = _resolve_release_path(family_medoid_path)
        centers = np.load(family_medoid_path)
        medoids = torch.from_numpy(
            np.stack([centers[f"{name}_medoid"] for name in FAMILY_NAMES])
        ).float()
        local_stds = torch.from_numpy(
            np.stack([centers[f"{name}_local_std"] for name in FAMILY_NAMES])
        ).float()
        distances = torch.cdist(medoids, medoids)
        distances.fill_diagonal_(torch.inf)
        nearest_distances = distances.min(dim=1).values
        source_radii = float(source_radius_fraction) * nearest_distances
        pairwise_support_gaps = (
            distances - source_radii[:, None] - source_radii[None, :]
        )
        pairwise_support_gaps.fill_diagonal_(torch.inf)

        if family_probabilities is None:
            family_probabilities = (
                19308,
                13216,
                17100,
                17708,
                14222,
                74230,
                25150,
                38,
                73041,
                28834,
            )
        family_probs = torch.as_tensor(family_probabilities, dtype=torch.float32)
        if family_probs.numel() != len(FAMILY_NAMES) or torch.any(family_probs < 0):
            raise ValueError("family_probabilities must contain 10 non-negative values")
        family_probs = family_probs / family_probs.sum()

        self.register_buffer("family_medoids", medoids, persistent=True)
        self.register_buffer("family_local_stds", local_stds, persistent=True)
        self.register_buffer("family_source_radii", source_radii, persistent=True)
        self.register_buffer("family_probabilities", family_probs, persistent=True)
        self.register_buffer("atom_family_lookup", _family_lookup(), persistent=True)
        self.register_buffer(
            "source_support_min_gap",
            pairwise_support_gaps.min(),
            persistent=True,
        )

    def _family_ids_from_atom_types(self, atom_types: torch.Tensor) -> torch.Tensor:
        if torch.any(atom_types < 1) or torch.any(atom_types >= len(self.atom_family_lookup)):
            raise ValueError("Encountered an atomic number outside the family lookup")
        family_ids = self.atom_family_lookup[atom_types.long()]
        if torch.any(family_ids < 0):
            unknown = torch.unique(atom_types[family_ids < 0]).tolist()
            raise ValueError(f"Atomic numbers without a configured family: {unknown}")
        return family_ids

    def _sample_family_source(self, family_ids: torch.Tensor) -> torch.Tensor:
        safe_ids = family_ids.clamp(min=0)
        means = self.family_medoids[safe_ids]
        stds = self.family_local_stds[safe_ids] * float(
            self.hparams.source_noise_scale
        )
        offsets = torch.randn_like(means) * stds
        offset_norms = torch.linalg.vector_norm(offsets, dim=-1, keepdim=True)
        radii = self.family_source_radii[safe_ids].unsqueeze(-1)
        offsets = offsets * torch.clamp(radii / offset_norms.clamp_min(1e-12), max=1.0)
        samples = means + offsets
        return torch.where((family_ids >= 0).unsqueeze(-1), samples, torch.zeros_like(samples))

    def configure_generation_family_templates(
        self,
        template_path: str,
        target_family_idx: int = -1,
        target_min_fraction: float = 0.0,
    ) -> None:
        """Use complete training-set family compositions during generation."""

        templates = np.load(template_path)
        family_ids = torch.from_numpy(templates["family_ids"]).long()
        lengths = torch.from_numpy(templates["lengths"]).long()
        counts = torch.from_numpy(templates["family_counts"]).long()
        if family_ids.ndim != 2 or counts.shape != (len(lengths), len(FAMILY_NAMES)):
            raise ValueError("Invalid family-template array shapes")
        if not 0 <= target_min_fraction <= 1:
            raise ValueError("target_min_fraction must be in [0, 1]")
        if target_family_idx >= len(FAMILY_NAMES):
            raise ValueError(f"target_family_idx must be in [-1, {len(FAMILY_NAMES) - 1}]")

        candidates = torch.arange(len(lengths))
        if target_family_idx >= 0:
            fractions = counts[:, target_family_idx].float() / lengths.clamp_min(1)
            candidates = candidates[
                (counts[:, target_family_idx] > 0)
                & (fractions >= float(target_min_fraction))
            ]
        if candidates.numel() == 0:
            raise ValueError(
                f"No templates satisfy target_family_idx={target_family_idx}, "
                f"target_min_fraction={target_min_fraction}"
            )

        self._generation_family_templates = family_ids
        self._generation_template_lengths = lengths
        self._generation_template_candidates = candidates
        self._generation_target_family_idx = int(target_family_idx)
        self._generation_target_min_fraction = float(target_min_fraction)

    def configure_generation_template_schedule(
        self,
        template_path: str,
        template_ids: Sequence[int],
    ) -> None:
        """Use a deterministic template-id schedule during generation."""

        self.configure_generation_family_templates(template_path)
        schedule = torch.as_tensor(template_ids, dtype=torch.long).flatten()
        if schedule.numel() == 0:
            raise ValueError("template_ids must contain at least one template")
        if torch.any(schedule < 0) or torch.any(
            schedule >= len(self._generation_template_lengths)
        ):
            raise ValueError("template_ids contains an out-of-range template index")
        self._generation_template_schedule = schedule
        self._generation_template_schedule_cursor = 0

    def forward(self, batch: Data, sample_posterior: bool = True):
        with torch.no_grad():
            encoded_batch = self.autoencoder.encode(batch)
            if sample_posterior:
                encoded_batch["x"] = encoded_batch["posterior"].sample()
            else:
                encoded_batch["x"] = encoded_batch["posterior"].mode()

            x_1, mask = to_dense_batch(encoded_batch["x"], encoded_batch["batch"])
            family_ids_flat = self._family_ids_from_atom_types(batch.atom_types)
            x_0_flat = self._sample_family_source(family_ids_flat)
            x_0, _ = to_dense_batch(x_0_flat, encoded_batch["batch"])
            family_ids, _ = to_dense_batch(
                family_ids_flat, encoded_batch["batch"], fill_value=-1
            )
            dense_encoded_batch = {
                "x_0": x_0,
                "x_1": x_1,
                "family_ids": family_ids,
                "token_mask": mask,
                "diffuse_mask": mask,
            }

        self.interpolant.device = dense_encoded_batch["x_1"].device
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        dataset_idx = batch.dataset_idx + 1
        spacegroup = batch.spacegroup
        if not self.hparams.conditioning.spacegroup:
            spacegroup = torch.zeros_like(batch.spacegroup)

        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=noisy_dense_encoded_batch["x_t"],
                    t=noisy_dense_encoded_batch["t"],
                    dataset_idx=dataset_idx,
                    spacegroup=spacegroup,
                    mask=mask,
                    x_sc=None,
                )
        else:
            x_sc = None

        pred_x = self.denoiser(
            x=noisy_dense_encoded_batch["x_t"],
            t=noisy_dense_encoded_batch["t"],
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=mask,
            x_sc=x_sc,
        )
        return pred_x, noisy_dense_encoded_batch

    def sample_and_decode(
        self,
        num_nodes_bincount,
        spacegroups_bincount,
        batch_size,
        cfg_scale=4.0,
        dataset_idx=0,
    ):
        if hasattr(self, "_generation_family_templates"):
            if hasattr(self, "_generation_template_schedule"):
                start = int(self._generation_template_schedule_cursor)
                stop = start + int(batch_size)
                if stop > len(self._generation_template_schedule):
                    raise RuntimeError(
                        "Generation template schedule exhausted: "
                        f"requested [{start}:{stop}] from "
                        f"{len(self._generation_template_schedule)} entries"
                    )
                selected = self._generation_template_schedule[start:stop]
                self._generation_template_schedule_cursor = stop
            else:
                candidates = self._generation_template_candidates
                selected = candidates[
                    torch.randint(len(candidates), (batch_size,), device=candidates.device)
                ]
            sample_lengths = self._generation_template_lengths[selected].to(self.device)
            selected_family_templates = self._generation_family_templates[selected].to(
                self.device
            )
        else:
            sample_lengths = torch.multinomial(
                num_nodes_bincount.float(), batch_size, replacement=True
            ).to(self.device)
            selected_family_templates = None
        dataset_idx_tensor = torch.full(
            (batch_size,), dataset_idx + 1, dtype=torch.int64, device=self.device
        )
        if not self.hparams.conditioning.spacegroup or spacegroups_bincount is None:
            spacegroup = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        else:
            spacegroup = torch.multinomial(
                spacegroups_bincount.float(), batch_size, replacement=True
            ).to(self.device)

        token_mask = torch.zeros(
            batch_size,
            max(sample_lengths),
            dtype=torch.bool,
            device=self.device,
        )
        for idx, length in enumerate(sample_lengths):
            token_mask[idx, :length] = True

        if selected_family_templates is None:
            family_ids = torch.full_like(token_mask, -1, dtype=torch.long)
            family_ids[token_mask] = torch.multinomial(
                self.family_probabilities,
                int(token_mask.sum()),
                replacement=True,
            )
        else:
            family_ids = selected_family_templates[:, : token_mask.size(1)].clone()
            family_ids[~token_mask] = -1
        x_0 = self._sample_family_source(family_ids)

        samples = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=batch_size,
            num_tokens=max(sample_lengths),
            emb_dim=self.denoiser.d_x,
            model=self.denoiser,
            dataset_idx=dataset_idx_tensor,
            spacegroup=spacegroup,
            cfg_scale=cfg_scale,
            x_0=x_0,
            token_mask=token_mask,
        )
        x = samples["clean_traj"][-1][token_mask]
        batch = {
            "x": x,
            "num_atoms": sample_lengths,
            "batch": torch.repeat_interleave(
                torch.arange(len(sample_lengths), device=self.device), sample_lengths
            ),
            "token_idx": (
                torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1
            )[token_mask],
            "source_family_ids": family_ids[token_mask],
        }
        if selected_family_templates is not None:
            batch["source_template_ids"] = selected.to(self.device)
        out = self.autoencoder.decode(batch)
        return out, batch, samples
