# Anonymous Review Code

This package contains the training and inference implementation for the
Transformer PBC-VAE and family-medoid latent flow-matching model used in the
submission. It intentionally contains no datasets, checkpoints, generated
samples, evaluation programs, author names, affiliations, or repository links.

## Included code

- `src/models/vae_module.py`: per-atom Transformer VAE and Gaussian posterior.
- `src/models/vae_pbc_module.py`: minimum-image fractional-coordinate loss.
- `src/models/ldm_module.py`: core latent flow-matching training and sampling.
- `src/models/ldm_family_medoid_module.py`: element-family medoid source prior and
  complete-template conditioning.
- `src/models/encoders/transformer.py` and
  `src/models/decoders/transformer.py`: first-stage Transformer backbone.
- `src/models/denoisers/dit.py`: DiT vector-field network.
- `src/models/interpolants/flow_matching.py`: linear flow interpolant and Euler
  sampler with classifier-free guidance.
- `src/data/`: CSV crystal preprocessing and Lightning datamodule.
- `scripts/`: training, medoid extraction, template construction, and sampling.

The released objective is deliberately minimal. VAE training uses atom-type,
lattice-length, lattice-angle, coordinate, and KL terms. Flow training uses only
the masked latent flow-matching MSE. No path-separation loss, structural
constraint loss, auxiliary validity loss, or evaluation-time loss is included.
Non-overlapping source regions are obtained geometrically by clipping each
family's sampled offset to a fixed fraction of its nearest-medoid distance.

## Environment

Python 3.10 or newer is recommended. Install a PyTorch build appropriate for the
available accelerator first, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

No CUDA-specific wheel is pinned in this package.

## Data layout

Datasets are not distributed with the review code. For either `mp_20` or
`mpts_52`, provide three CSV files with at least `material_id` and `cif` columns:

```text
data/
  mp_20/
    train.csv
    val.csv
    test.csv
  mpts_52/
    train.csv
    val.csv
    test.csv
```

Preprocessed tensors are created on first use under a configurable cache
directory. Dataset paths, run names, and accelerator settings can be overridden
through environment variables or Hydra arguments; no machine-specific path is
embedded in the code.

## Stage 1: train the VAE

MP-20 uses the periodic-boundary-aware fractional-coordinate loss:

```bash
DATA_ROOT=/path/to/data MAX_EPOCHS=2000 bash scripts/train_mp20_vae.sh
```

MPTS-52 can be trained with:

```bash
DATA_ROOT=/path/to/data MAX_EPOCHS=2000 bash scripts/train_mpts52_vae.sh
```

Both scripts configure a `d_model=512`, eight-layer Transformer and an
eight-dimensional per-atom latent. `last.ckpt` is saved by Lightning at the end
of training.

## Stage 2: construct the family source prior

Extract one medoid and a local diagonal scale for each of the ten periodic-table
families from all training-set posterior means:

```bash
python scripts/extract_family_medoids.py \
  --project . \
  --checkpoint /path/to/vae_last.ckpt \
  --data-root /path/to/data/mp_20 \
  --processed-dir /path/to/cache/mp_20 \
  --output-dir /path/to/medoids/mp20 \
  --dataset mp20 \
  --model-class pbc
```

Build complete per-structure family templates for inference:

```bash
python scripts/build_family_templates.py \
  --project . \
  --dataset mp20 \
  --data-root /path/to/data/mp_20 \
  --processed-dir /path/to/cache/mp_20 \
  --output /path/to/medoids/mp20/train_family_templates.npz
```

## Stage 3: train family-medoid LFM

```bash
VAE_CKPT=/path/to/vae_last.ckpt \
MEDOID_PATH=/path/to/medoids/mp20/family_centers.npz \
DATA_ROOT=/path/to/data \
MAX_EPOCHS=2000 \
bash scripts/train_mp20_family_medoid_lfm.sh
```

The released MP-20 configuration uses `d_x=8`, `d_model=768`, 12 attention
heads, 12 DiT layers, self-conditioning probability 0.5, source noise scale
0.1, and source-radius fraction 0.2. The VAE is frozen during this stage.

## Inference

Generate CIF strings using complete family-composition templates sampled from
the training distribution:

```bash
LFM_CKPT=/path/to/lfm_last.ckpt \
VAE_CKPT=/path/to/vae_last.ckpt \
MEDOID_PATH=/path/to/medoids/mp20/family_centers.npz \
TEMPLATE_PATH=/path/to/medoids/mp20/train_family_templates.npz \
DATA_ROOT=/path/to/data \
OUTPUT_CSV=/path/to/output/generated.csv \
NUM_SAMPLES=10000 \
bash scripts/sample_family_medoid_lfm.sh
```

The inference entrypoint also supports `--target_family_idx` and
`--target_family_min_fraction` to restrict sampling to templates containing a
chosen element family. The output is an `id,cif` CSV; metric computation is not
part of this review package.

## Reproducibility notes

- Training scripts default to seed 9 and save `last.ckpt`.
- The family medoids are computed from all atoms in the training split.
- Family-source noise is clipped to non-overlapping support balls; this is a
  source construction rule, not an auxiliary loss.
- The package has no network calls and does not require an API key.
