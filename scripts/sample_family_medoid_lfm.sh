#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-mp20}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CFG_SCALE="${CFG_SCALE:-2.0}"
SEED="${SEED:-9}"
GPU_ID="${GPU_ID:-0}"

case "$DATASET" in
  mp20)
    LFM_CKPT="${LFM_CKPT:?Set LFM_CKPT to a trained flow checkpoint}"
    VAE_CKPT="${VAE_CKPT:?Set VAE_CKPT to the matching VAE checkpoint}"
    MEDOID_PATH="${MEDOID_PATH:?Set MEDOID_PATH to the family medoids}"
    TEMPLATE_PATH="${TEMPLATE_PATH:?Set TEMPLATE_PATH to family templates}"
    REFERENCE_CSV="${REFERENCE_CSV:-$PROJECT_ROOT/data/mp_20/train.csv}"
    VAE_TARGET="src.models.vae_pbc_module.PBCVariationalAutoencoderLitModule"
    ;;
  mpts52)
    LFM_CKPT="${LFM_CKPT:?Set LFM_CKPT to a trained flow checkpoint}"
    VAE_CKPT="${VAE_CKPT:?Set VAE_CKPT to the matching VAE checkpoint}"
    MEDOID_PATH="${MEDOID_PATH:?Set MEDOID_PATH to the family medoids}"
    TEMPLATE_PATH="${TEMPLATE_PATH:?Set TEMPLATE_PATH to family templates}"
    REFERENCE_CSV="${REFERENCE_CSV:-$PROJECT_ROOT/data/mpts_52/train.csv}"
    VAE_TARGET="src.models.vae_module.VariationalAutoencoderLitModule"
    ;;
  *)
    echo "DATASET must be mp20 or mpts52" >&2
    exit 2
    ;;
esac

OUTPUT_CSV="${OUTPUT_CSV:-$PROJECT_ROOT/outputs/${DATASET}_samples_${NUM_SAMPLES}.csv}"
mkdir -p "$(dirname "$OUTPUT_CSV")"
test -s "$LFM_CKPT"
test -s "$TEMPLATE_PATH"
test -s "$REFERENCE_CSV"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

exec "$PYTHON_BIN" src/generate_crystals.py \
  --ckpt_path "$LFM_CKPT" \
  --autoencoder_ckpt "$VAE_CKPT" \
  --autoencoder_module_target "$VAE_TARGET" \
  --family_medoid_path "$MEDOID_PATH" \
  --output_csv "$OUTPUT_CSV" \
  --dataset mp20 \
  --reference_cif_csv "$REFERENCE_CSV" \
  --family_template_path "$TEMPLATE_PATH" \
  --num_samples "$NUM_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --cfg_scale "$CFG_SCALE" \
  --seed "$SEED" \
  --device cuda \
  --use_ema \
  --conversion_mode official \
  --overwrite \
  --print_stats
