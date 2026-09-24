"""Generate crystal CIF strings from a trained latent diffusion checkpoint.

This standalone entrypoint supports Gaussian and family-medoid source priors.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.ldm_module import DATASET_TO_IDX, LatentDiffusionLitModule  # noqa: E402
from src.models.ldm_family_medoid_module import (  # noqa: E402
    FAMILY_NAMES,
    FamilyMedoidLatentDiffusionLitModule,
)


def _make_sampling_override(args: argparse.Namespace) -> DictConfig | None:
    """Read sampling config from checkpoint hparams and apply generation overrides."""

    checkpoint = torch.load(str(args.ckpt_path), map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    sampling = hparams.get("sampling")
    if sampling is None:
        return None

    sampling = deepcopy(sampling)
    if not isinstance(sampling, DictConfig):
        sampling = OmegaConf.create(sampling)

    with open_dict(sampling):
        sampling.datasets = [args.dataset]
        sampling.batch_size = args.batch_size
        sampling.num_samples = args.num_samples
        sampling.visualize = False
        if args.cfg_scale is not None:
            sampling.cfg_scale = args.cfg_scale
        if args.data_dir is not None:
            sampling.data_dir = str(args.data_dir)
        if args.reference_cif_csv is not None:
            sampling.reference_cif_csv = str(args.reference_cif_csv)

    return sampling


def load_model(args: argparse.Namespace) -> LatentDiffusionLitModule:
    overrides = {}
    sampling = _make_sampling_override(args)
    if sampling is not None:
        overrides["sampling"] = sampling
    if args.autoencoder_ckpt is not None:
        overrides["autoencoder_ckpt"] = str(args.autoencoder_ckpt)
    if args.autoencoder_module_target is not None:
        overrides["autoencoder_module_target"] = args.autoencoder_module_target
    if args.family_medoid_path is not None:
        overrides["family_medoid_path"] = str(args.family_medoid_path)
    checkpoint = torch.load(str(args.ckpt_path), map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    if "family_medoid_path" in hparams:
        model_class = FamilyMedoidLatentDiffusionLitModule
    else:
        model_class = LatentDiffusionLitModule
    print(f"Loading diffusion module: {model_class.__name__}")
    model = model_class.load_from_checkpoint(
        str(args.ckpt_path),
        map_location="cpu",
        weights_only=False,
        **overrides,
    )
    if args.family_idx is not None and args.family_template_path is not None:
        raise ValueError("--family_idx and --family_template_path are mutually exclusive")
    if args.family_idx is not None:
        if not isinstance(model, FamilyMedoidLatentDiffusionLitModule):
            raise ValueError("--family_idx requires a family-medoid LFM checkpoint")
        if not 0 <= args.family_idx < len(FAMILY_NAMES):
            raise ValueError(f"--family_idx must be in [0, {len(FAMILY_NAMES) - 1}]")
        probabilities = torch.zeros_like(model.family_probabilities)
        probabilities[args.family_idx] = 1.0
        model.family_probabilities.copy_(probabilities)
        print(
            f"Using fixed source family {args.family_idx}: "
            f"{FAMILY_NAMES[args.family_idx]}"
        )
    if args.family_template_path is not None:
        if not isinstance(model, FamilyMedoidLatentDiffusionLitModule):
            raise ValueError("--family_template_path requires a family-medoid LFM checkpoint")
        model.configure_generation_family_templates(
            str(args.family_template_path),
            target_family_idx=args.target_family_idx,
            target_min_fraction=args.target_family_min_fraction,
        )
        target_text = "unconditional"
        if args.target_family_idx >= 0:
            target_text = (
                f"{args.target_family_idx}:{FAMILY_NAMES[args.target_family_idx]}, "
                f"minimum fraction={args.target_family_min_fraction}"
            )
        print(f"Using complete family templates from {args.family_template_path} ({target_text})")
    if args.use_ema:
        _apply_ema_weights(model, args.ckpt_path)
    model.eval()
    return model


def _apply_ema_weights(model: LatentDiffusionLitModule, ckpt_path: Path) -> None:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    optimizer_states = checkpoint.get("optimizer_states", [])
    ema_params = None
    for opt_state in optimizer_states:
        if isinstance(opt_state, dict) and "ema" in opt_state:
            ema_params = opt_state["ema"]
            break

    if ema_params is None:
        print("[Warning] --use_ema was set, but no EMA weights were found in checkpoint.")
        return

    model_params = list(model.parameters())
    if len(ema_params) != len(model_params):
        print(
            "[Warning] EMA parameter count does not match model parameter count: "
            f"{len(ema_params)} vs {len(model_params)}. Falling back to raw weights."
        )
        return

    for param, ema_param in zip(model_params, ema_params):
        param.data.copy_(ema_param.to(device=param.device, dtype=param.dtype))
    print(f"Applied EMA weights from checkpoint optimizer state ({len(ema_params)} tensors).")


def _tensor_to_numpy_rows(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], start_id: int):
    rows = []
    start_idx = 0
    for idx_in_batch, num_atom in enumerate(batch["num_atoms"].tolist()):
        atom_types = out["atom_types"].narrow(0, start_idx, num_atom).argmax(dim=1)
        atom_types[atom_types == 0] = 1
        frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
        lengths = out["lengths"][idx_in_batch] * float(num_atom) ** (1.0 / 3.0)
        angles = torch.rad2deg(out["angles"][idx_in_batch])

        rows.append(
            {
                "id": start_id + len(rows) + 1,
                "atom_types": atom_types.detach().cpu().numpy(),
                "frac_coords": frac_coords.detach().cpu().numpy(),
                "lengths": lengths.detach().cpu().numpy(),
                "angles": angles.detach().cpu().numpy(),
            }
        )
        start_idx += num_atom
    return rows


def _row_to_cif(
    row: dict,
    angle_min: float,
    angle_max: float,
    conversion_mode: str,
) -> tuple[str | None, str | None]:
    lengths = row["lengths"]
    angles = row["angles"]
    if not all(math.isfinite(float(x)) for x in list(lengths) + list(angles)):
        return None, "non_finite_lattice"
    if conversion_mode == "strict" and not all(float(x) > 0.0 for x in lengths):
        return None, "non_positive_lengths"
    if conversion_mode == "strict" and not all(float(angle_min) < float(x) < float(angle_max) for x in angles):
        return None, "angle_out_of_range"

    try:
        structure = Structure(
            lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
            species=row["atom_types"].astype(int).tolist(),
            coords=row["frac_coords"],
            coords_are_cartesian=False,
        )
        if conversion_mode == "official":
            cutoff = 0.5
            dist_mat = structure.distance_matrix
            dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
            if dist_mat.min() < cutoff or structure.volume < 0.1:
                return str(CifWriter(structure)), "struct_invalid"
        return str(CifWriter(structure)), None
    except Exception as exc:
        print(f"[Warning] skip generated structure {row['id']}: {exc}")
        return None, "pymatgen_write_failed"


def _prepare_output(args: argparse.Namespace) -> tuple[bool, int]:
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists() and args.overwrite:
        args.output_csv.unlink()

    if not args.output_csv.exists():
        return True, 0
    if args.append:
        with args.output_csv.open("r", newline="", encoding="utf-8") as f:
            existing_rows = max(sum(1 for _ in f) - 1, 0)
        return False, existing_rows

    raise FileExistsError(
        f"{args.output_csv} already exists. Use --overwrite or --append to continue."
    )


def _append_csv_rows(csv_path: Path, rows: list[dict[str, str | int]], write_header: bool) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "cif"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _save_cif_file(save_dir: Path, row_id: int, cif: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / f"gen_{row_id}.cif").write_text(cif, encoding="utf-8")


def _print_round_stats(rows: list[dict]) -> None:
    if len(rows) == 0:
        return
    lengths = torch.as_tensor([row["lengths"] for row in rows], dtype=torch.float32)
    angles = torch.as_tensor([row["angles"] for row in rows], dtype=torch.float32)
    num_atoms = torch.as_tensor([len(row["atom_types"]) for row in rows], dtype=torch.float32)
    non_positive = (lengths <= 0).any(dim=1).float().mean().item()
    angle_bad = ((angles <= 0) | (angles >= 180)).any(dim=1).float().mean().item()
    print(
        "Decoded stats: "
        f"num_atoms=[{num_atoms.min().item():.0f}, {num_atoms.max().item():.0f}], "
        f"lengths=[{lengths.min().item():.4f}, {lengths.max().item():.4f}], "
        f"angles=[{angles.min().item():.2f}, {angles.max().item():.2f}], "
        f"non_positive_length_rate={non_positive:.3f}, "
        f"invalid_angle_rate={angle_bad:.3f}"
    )


@torch.no_grad()
def generate(args: argparse.Namespace) -> None:
    if args.dataset not in DATASET_TO_IDX:
        raise ValueError(f"Unsupported dataset {args.dataset!r}. Choices: {sorted(DATASET_TO_IDX)}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    write_header, total_saved = _prepare_output(args)
    device = torch.device(args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args).to(device)
    try:
        model.setup("predict")
    except Exception as exc:
        print(f"[Warning] model.setup('predict') failed, continuing: {exc}")

    cfg_scale = args.cfg_scale
    if cfg_scale is None:
        cfg_scale = float(getattr(model.hparams.sampling, "cfg_scale", 4.0))

    print(f"Loaded checkpoint: {args.ckpt_path}")
    print(f"Dataset: {args.dataset}, device: {device}, cfg_scale: {cfg_scale}")
    print(f"Conversion mode: {args.conversion_mode}")
    print(f"Generating until {args.num_samples} CIFs are saved")

    round_id = 0
    while total_saved < args.num_samples:
        round_id += 1
        if args.max_rounds is not None and round_id > args.max_rounds:
            print(f"[Warning] reached --max_rounds={args.max_rounds}, stopping early.")
            break

        current_batch_size = min(args.batch_size, args.num_samples - total_saved)
        print(f"\n=== Round {round_id} | saved {total_saved}/{args.num_samples} ===")
        out, batch, _ = model.sample_and_decode(
            num_nodes_bincount=model.num_nodes_bincount[args.dataset],
            spacegroups_bincount=model.spacegroups_bincount[args.dataset],
            batch_size=current_batch_size,
            cfg_scale=cfg_scale,
            dataset_idx=DATASET_TO_IDX[args.dataset],
        )

        tensor_rows = _tensor_to_numpy_rows(out, batch, start_id=total_saved)
        if args.print_stats:
            _print_round_stats(tensor_rows)
        csv_rows = []
        skip_reasons = Counter()
        saved_invalid = Counter()
        for row in tqdm(tensor_rows, desc="    CIF conversion"):
            cif, skip_reason = _row_to_cif(
                row,
                angle_min=args.angle_min,
                angle_max=args.angle_max,
                conversion_mode=args.conversion_mode,
            )
            if cif is None:
                skip_reasons[skip_reason or "unknown"] += 1
                continue
            if skip_reason is not None:
                saved_invalid[skip_reason] += 1
            row_id = total_saved + len(csv_rows) + 1
            csv_rows.append({"id": row_id, "cif": cif})
            if args.save_cif_dir is not None:
                _save_cif_file(args.save_cif_dir, row_id, cif)

        if len(csv_rows) == 0:
            print("[Warning] no valid CIFs in this round.")
            if args.stop_on_empty_round:
                break
        else:
            _append_csv_rows(args.output_csv, csv_rows, write_header=write_header)
            write_header = False
            total_saved += len(csv_rows)

        print(f"Saved this round: {len(csv_rows)}")
        if skip_reasons:
            reason_text = ", ".join(f"{k}={v}" for k, v in sorted(skip_reasons.items()))
            print(f"Skipped this round: {reason_text}")
        if saved_invalid:
            reason_text = ", ".join(f"{k}={v}" for k, v in sorted(saved_invalid.items()))
            print(f"Saved but marked invalid this round: {reason_text}")
        print(f"Total saved: {total_saved}/{args.num_samples}")

    print(f"\nFinished. Total CIFs saved: {total_saved}")
    print(f"CSV path: {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate crystal CIF CSV from a latent flow checkpoint."
    )
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Diffusion checkpoint path.")
    parser.add_argument("--output_csv", type=Path, required=True, help="Output CSV with id,cif columns.")
    parser.add_argument("--dataset", default="mp20", choices=sorted(DATASET_TO_IDX), help="Sampling dataset.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Target number of saved CIFs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Sampling batch size.")
    parser.add_argument("--cfg_scale", type=float, default=None, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--data_dir", type=Path, default=None, help="Override sampling.data_dir.")
    parser.add_argument(
        "--reference_cif_csv",
        type=Path,
        default=None,
        help="Override MP20 reference CSV used to sample atom-count/spacegroup priors.",
    )
    parser.add_argument(
        "--family_idx",
        type=int,
        default=None,
        help="Fix every source token to one family-medoid region (0-9).",
    )
    parser.add_argument(
        "--family_template_path",
        type=Path,
        default=None,
        help="NPZ containing complete training-set family-ID templates.",
    )
    parser.add_argument(
        "--family_medoid_path",
        type=Path,
        default=None,
        help="Override the family-medoid NPZ path stored in the checkpoint.",
    )
    parser.add_argument(
        "--target_family_idx",
        type=int,
        default=-1,
        help="Restrict templates to those containing this family (0-9).",
    )
    parser.add_argument(
        "--target_family_min_fraction",
        type=float,
        default=0.0,
        help="Minimum target-family fraction for conditional template sampling.",
    )
    parser.add_argument(
        "--autoencoder_ckpt",
        type=Path,
        default=None,
        help="Replace the embedded VAE after diffusion and EMA weights are loaded.",
    )
    parser.add_argument(
        "--autoencoder_module_target",
        default=None,
        help="Import path of the VAE LightningModule subclass used by the override checkpoint.",
    )
    parser.add_argument("--save_cif_dir", type=Path, default=None, help="Optional directory for individual CIFs.")
    parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use EMA weights from optimizer state when available.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_csv if it exists.")
    parser.add_argument("--append", action="store_true", help="Append to output_csv if it exists.")
    parser.add_argument("--max_rounds", type=int, default=None, help="Optional safety cap on sampling rounds.")
    parser.add_argument("--angle_min", type=float, default=50.0, help="Minimum lattice angle to save.")
    parser.add_argument("--angle_max", type=float, default=130.0, help="Maximum lattice angle to save.")
    parser.add_argument(
        "--conversion_mode",
        choices=["strict", "official"],
        default="strict",
        help="strict filters invalid lattices before saving; official applies the reference conversion checks.",
    )
    parser.add_argument("--print_stats", action="store_true", help="Print decoded lattice statistics.")
    parser.add_argument(
        "--stop_on_empty_round",
        action="store_true",
        help="Stop if a full sampling round produces no CIFs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
