"""Extract periodic-family medoids from crystal VAE posterior means."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.components.crystal_dataset import CrystalDataset  # noqa: E402
from src.models.ldm_family_medoid_module import FAMILY_NAMES, _family_lookup  # noqa: E402
from src.models.vae_module import VariationalAutoencoderLitModule  # noqa: E402
from src.models.vae_pbc_module import PBCVariationalAutoencoderLitModule  # noqa: E402


def main(args: argparse.Namespace) -> None:
    os.chdir(args.project)
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_class = {
        "vae": VariationalAutoencoderLitModule,
        "pbc": PBCVariationalAutoencoderLitModule,
    }[args.model_class]
    model = model_class.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu", weights_only=False
    ).eval()
    device = torch.device(args.device)
    model.to(device)

    for module in model.modules():
        if hasattr(module, "mask_indices_cache"):
            module.lmax_cache = None
            module.mmax_cache = None
            module.mask_indices_cache = None
            module.rotate_inv_rescale_cache = None
    dataset = CrystalDataset(
        name=f"{args.dataset}_train",
        path=str(args.data_root / "train.csv"),
        save_path=str(args.processed_dir / "train.pt"),
        prop="formation_energy_per_atom",
        niggli=True,
        primitive=False,
        graph_method="crystalnn",
        tolerance=0.1,
        use_space_group=False,
        preprocess_workers=1,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    means, atom_types = [], []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            encoded = model.encode(batch)
            means.append(encoded["posterior"].mean.cpu().numpy().astype(np.float32))
            atom_types.append(batch.atom_types.cpu().numpy().astype(np.int16))
            if (batch_idx + 1) % 20 == 0:
                print(f"encoded batches: {batch_idx + 1}/{len(loader)}", flush=True)

    means = np.concatenate(means)
    atom_types = np.concatenate(atom_types)
    family_lookup = _family_lookup().numpy()
    families = family_lookup[atom_types]
    if np.any(families < 0):
        raise ValueError("Encountered an atom outside the configured ten families")

    family_latent_dim = int(
        checkpoint.get("hyper_parameters", {}).get("family_latent_dim", means.shape[1])
    )
    medoid_dim = args.medoid_dim or family_latent_dim
    if not 1 <= medoid_dim <= means.shape[1]:
        raise ValueError(f"medoid_dim must be in [1, {means.shape[1]}]")
    medoid_features = means[:, :medoid_dim]
    scaler = StandardScaler().fit(medoid_features)
    scaled = scaler.transform(medoid_features)
    centers = {}
    report_families = {}
    medoids = []

    for family_idx, family_name in enumerate(FAMILY_NAMES):
        family_indices = np.flatnonzero(families == family_idx)
        family_raw = means[family_indices]
        family_scaled = scaled[family_indices]
        centroid_raw = family_raw.mean(axis=0)
        centroid_feature = medoid_features[family_indices].mean(axis=0)
        centroid_scaled = scaler.transform(centroid_feature[None, :])
        medoid_local, centroid_distance = pairwise_distances_argmin_min(
            centroid_scaled, family_scaled, metric="euclidean"
        )
        medoid_local = int(medoid_local[0])
        medoid_global = int(family_indices[medoid_local])
        medoid = means[medoid_global]
        medoid_scaled = scaled[medoid_global : medoid_global + 1]

        neighbors = NearestNeighbors(
            n_neighbors=min(args.local_neighbors, len(family_indices)),
            metric="euclidean",
            n_jobs=-1,
        ).fit(family_scaled)
        neighbor_local = neighbors.kneighbors(medoid_scaled, return_distance=False)[0]
        local_points = family_raw[neighbor_local]
        local_std = np.maximum(local_points.std(axis=0), args.min_local_std)

        global_neighbors = NearestNeighbors(
            n_neighbors=min(args.purity_neighbors, len(means)),
            metric="euclidean",
            n_jobs=-1,
        ).fit(scaled)
        global_idx = global_neighbors.kneighbors(medoid_scaled, return_distance=False)[0]
        purity = float(np.mean(families[global_idx] == family_idx))

        centers[f"{family_name}_centroid"] = centroid_raw.astype(np.float32)
        centers[f"{family_name}_medoid"] = medoid.astype(np.float32)
        centers[f"{family_name}_local_std"] = local_std.astype(np.float32)
        medoids.append(medoid)
        report_families[family_name] = {
            "family_idx": family_idx,
            "atom_count": int(len(family_indices)),
            "medoid_global_atom_index": medoid_global,
            "medoid_atomic_number": int(atom_types[medoid_global]),
            "centroid_to_medoid_standardized_distance": float(centroid_distance[0]),
            "local_neighbors": int(len(neighbor_local)),
            "medoid_global_neighbor_family_purity": purity,
            "local_std_mean": float(local_std.mean()),
            "local_std_max": float(local_std.max()),
        }
        print(f"{family_name}: atoms={len(family_indices)}, purity={purity:.4f}", flush=True)

    medoids = np.stack(medoids)
    pair_distances = np.linalg.norm(medoids[:, None] - medoids[None, :], axis=-1)
    np.fill_diagonal(pair_distances, np.inf)
    nearest = pair_distances.min(axis=1)
    radii = args.source_radius_fraction * nearest
    gaps = pair_distances - radii[:, None] - radii[None, :]
    np.fill_diagonal(gaps, np.inf)
    min_pair = np.unravel_index(np.argmin(gaps), gaps.shape)
    report = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "model_class": model_class.__name__,
        "representation": "posterior_mean",
        "latent_dimension": int(means.shape[1]),
        "medoid_selection_dimension": int(medoid_dim),
        "structures": int(len(dataset)),
        "atoms": int(len(means)),
        "source_radius_fraction": args.source_radius_fraction,
        "minimum_medoid_distance": float(pair_distances.min()),
        "minimum_source_support_gap": float(gaps.min()),
        "minimum_gap_pair": [FAMILY_NAMES[min_pair[0]], FAMILY_NAMES[min_pair[1]]],
        "supports_non_overlapping": bool(gaps.min() > 0),
        "families": report_families,
    }
    if not report["supports_non_overlapping"]:
        raise RuntimeError("Family source supports overlap")

    np.savez_compressed(args.output_dir / "family_centers.npz", **centers)
    (args.output_dir / "family_centers_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mp20", "mpts52"), required=True)
    parser.add_argument("--model-class", choices=("vae", "pbc"), required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--medoid-dim", type=int, default=0)
    parser.add_argument("--local-neighbors", type=int, default=1000)
    parser.add_argument("--purity-neighbors", type=int, default=1000)
    parser.add_argument("--min-local-std", type=float, default=1e-5)
    parser.add_argument("--source-radius-fraction", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
