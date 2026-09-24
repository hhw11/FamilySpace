"""Build complete per-structure element-family templates from crystal data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.components.crystal_dataset import CrystalDataset  # noqa: E402
from src.models.ldm_family_medoid_module import FAMILY_NAMES, _family_lookup  # noqa: E402


def main(args: argparse.Namespace) -> None:
    os.chdir(args.project)
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

    lookup = _family_lookup().numpy()
    sequences = []
    counts = []
    lengths = []
    for graph in dataset:
        atom_types = graph.atom_types.detach().cpu().numpy().astype(np.int64)
        family_ids = lookup[atom_types]
        if np.any(family_ids < 0):
            unknown = np.unique(atom_types[family_ids < 0]).tolist()
            raise ValueError(f"Atomic numbers outside the family map: {unknown}")
        sequences.append(family_ids.astype(np.int16))
        lengths.append(len(family_ids))
        counts.append(
            np.bincount(family_ids, minlength=len(FAMILY_NAMES)).astype(np.int16)
        )

    max_atoms = max(lengths)
    padded = np.full((len(sequences), max_atoms), -1, dtype=np.int16)
    for index, family_ids in enumerate(sequences):
        padded[index, : len(family_ids)] = family_ids

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        family_ids=padded,
        lengths=np.asarray(lengths, dtype=np.int16),
        family_counts=np.stack(counts),
    )
    print(
        {
            "output": str(args.output),
            "structures": len(sequences),
            "max_atoms": max_atoms,
            "family_names": FAMILY_NAMES,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mp20", "mpts52"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
