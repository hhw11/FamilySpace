#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAE_CKPT="${VAE_CKPT:?Set VAE_CKPT to a trained VAE checkpoint}"
MEDOID_PATH="${MEDOID_PATH:?Set MEDOID_PATH to the extracted family medoids}"
RUN_NAME="${RUN_NAME:-mpts52_z8_family_medoid_lfm}"
FAMILY_COUNTS='[39046,16075,31620,38250,34038,149537,47740,109,107865,37054]'

test -s "$VAE_CKPT"
test -s "$MEDOID_PATH"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HYDRA_FULL_ERROR=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

exec "$PYTHON_BIN" src/train_diffusion.py \
  data=mpts_52 diffusion_module=ldm_family_medoid \
  callbacks=diffusion_mp20_only trainer=gpu logger=csv name="$RUN_NAME" \
  paths.root_dir="$PROJECT_ROOT" paths.data_dir="$DATA_ROOT" \
  paths.log_dir="$PROJECT_ROOT/logs" \
  data.datamodule.root="$DATA_ROOT/mpts_52" \
  data.datamodule.processed_dir="$DATA_ROOT/mpts_52_csv_processed" \
  data.datamodule.batch_size.train="$BATCH_SIZE" \
  data.datamodule.batch_size.val="$BATCH_SIZE" \
  data.datamodule.batch_size.test="$BATCH_SIZE" \
  data.datamodule.num_workers.train="$NUM_WORKERS" \
  data.datamodule.num_workers.val="$NUM_WORKERS" \
  data.datamodule.num_workers.test="$NUM_WORKERS" \
  data.datamodule.preprocess_workers="$NUM_WORKERS" \
  diffusion_module.autoencoder_ckpt="$VAE_CKPT" \
  +diffusion_module.autoencoder_module_target=src.models.vae_module.VariationalAutoencoderLitModule \
  diffusion_module.family_medoid_path="$MEDOID_PATH" \
  "diffusion_module.family_probabilities=$FAMILY_COUNTS" \
  diffusion_module.source_noise_scale=0.1 \
  diffusion_module.source_radius_fraction=0.2 \
  diffusion_module.denoiser.d_x=8 diffusion_module.denoiser.d_model=768 \
  diffusion_module.denoiser.nhead=12 diffusion_module.denoiser.num_layers=12 \
  diffusion_module.interpolant.self_condition=true \
  diffusion_module.interpolant.self_condition_prob=0.5 \
  diffusion_module.sampling.reference_cif_csv="$DATA_ROOT/mpts_52/train.csv" \
  trainer.max_epochs="$MAX_EPOCHS" trainer.check_val_every_n_epoch=250 \
  trainer.log_every_n_steps=20 trainer.num_sanity_val_steps=0 \
  +trainer.limit_val_batches=0 \
  callbacks.model_checkpoint.monitor=null \
  callbacks.model_checkpoint.save_top_k=1 \
  callbacks.model_checkpoint.save_last=true \
  callbacks.model_checkpoint.every_n_epochs=10 \
  callbacks.model_checkpoint.save_on_train_epoch_end=true
