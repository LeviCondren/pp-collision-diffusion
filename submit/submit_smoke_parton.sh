#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=smoke_parton
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs/%j_smoke_parton.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs/%j_smoke_parton.err

TRAIN_SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/per_parton_cond_train.py
INFER_NB=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/per_parton_cond_infer_smoke.ipynb
INFER_NB_OUT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/per_parton_cond_infer_smoke_out.ipynb
CONDA_PYTHON=/pscratch/sd/l/lcondren/.conda/envs/mg5_new/bin/python3
CKPT_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp

mkdir -p ${CKPT_DIR}/logs
mkdir -p ${CKPT_DIR}/parton_smoke

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Stage 1: Training with Horovod ────────────────────────────────────────────
echo "=== Stage 1: per-parton training (1 epoch, tiny model) ==="
(
  module load tensorflow/2.15.0
  export PYTHONPATH=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts
  horovodrun --gloo -np 1 python3 -u $TRAIN_SCRIPT \
      --run_name         parton_smoke \
      --batch            16 \
      --epoch            1 \
      --num_layers       2 \
      --proj_dim         32 \
      --num_part         100 \
      --n_train          50 \
      --n_val            20 \
      --time_limit_hours 0.4
)
TRAIN_EXIT=$?
echo "Training exit code: $TRAIN_EXIT"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "TRAINING FAILED — aborting smoke test"
    exit $TRAIN_EXIT
fi

CKPT=${CKPT_DIR}/parton_smoke/pet_pp.weights.h5
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi
echo "Checkpoint saved: $(ls -lh $CKPT)"

# ── Stage 2: Inference notebook via nbconvert (conda env) ─────────────────────
echo ""
echo "=== Stage 2: per-parton inference notebook ==="
$CONDA_PYTHON -m jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=3600 \
    --output "$INFER_NB_OUT" \
    "$INFER_NB"

INFER_EXIT=$?
echo "Inference notebook exit code: $INFER_EXIT"

if [ $INFER_EXIT -eq 0 ]; then
    echo ""
    echo "Plots saved to: ${CKPT_DIR}/parton_smoke/"
    ls -lh ${CKPT_DIR}/parton_smoke/*.png 2>/dev/null || echo "(no .png files found)"
    echo ""
    echo "SMOKE TEST PASSED"
else
    echo ""
    echo "SMOKE TEST FAILED (inference exit $INFER_EXIT)"
fi

echo "Job ${SLURM_JOB_ID} finished: $(date)"
exit $INFER_EXIT
