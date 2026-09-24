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
RUN_NAME="${RUN_NAME:-mp20_pbc_balanced_vae_d512_z8}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

exec "$PYTHON_BIN" src/train_autoencoder.py \
  data=mp20_only encoder=transformer decoder=transformer \
  autoencoder_module=vae callbacks=autoencoder_mp20_only logger=csv \
  trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs="$MAX_EPOCHS" \
  trainer.check_val_every_n_epoch=250 trainer.log_every_n_steps=20 \
  trainer.num_sanity_val_steps=0 +trainer.limit_val_batches=0 \
  paths.root_dir="$PROJECT_ROOT" paths.data_dir="$DATA_ROOT" \
  data.datamodule.processed_dir="$DATA_ROOT/mp_20_csv_processed" \
  data.datamodule.batch_size.train="$BATCH_SIZE" \
  data.datamodule.batch_size.val="$BATCH_SIZE" \
  data.datamodule.batch_size.test="$BATCH_SIZE" \
  data.datamodule.num_workers.train="$NUM_WORKERS" \
  data.datamodule.num_workers.val="$NUM_WORKERS" \
  data.datamodule.num_workers.test="$NUM_WORKERS" \
  encoder.d_model=512 encoder.dim_feedforward=2048 encoder.num_layers=8 \
  decoder.d_model=512 decoder.dim_feedforward=2048 decoder.num_layers=8 \
  autoencoder_module._target_=src.models.vae_pbc_module.PBCVariationalAutoencoderLitModule \
  autoencoder_module.latent_dim=8 \
  autoencoder_module.loss_weights.loss_atom_types.mp20=2.0 \
  autoencoder_module.loss_weights.loss_lengths.mp20=1.0 \
  autoencoder_module.loss_weights.loss_angles.mp20=10.0 \
  autoencoder_module.loss_weights.loss_frac_coords.mp20=5.0 \
  autoencoder_module.loss_weights.loss_kl.mp20=0.00001 \
  callbacks.model_checkpoint.monitor=train/loss_epoch \
  callbacks.model_checkpoint.mode=min callbacks.model_checkpoint.save_top_k=1 \
  callbacks.model_checkpoint.save_last=true \
  callbacks.model_checkpoint.every_n_epochs=10 \
  callbacks.model_checkpoint.save_on_train_epoch_end=true \
  name="$RUN_NAME"
