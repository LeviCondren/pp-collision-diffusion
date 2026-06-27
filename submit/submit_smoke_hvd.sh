#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=smoke_hvd
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs/%j_smoke_hvd.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs/%j_smoke_hvd.err

TRAIN_SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/train_pp_horovod.py
INFER_SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/smoke_inference_run_hvd.py
CONDA_PYTHON=/pscratch/sd/l/lcondren/.conda/envs/mg5_new/bin/python3

mkdir -p /pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs
mkdir -p /pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/smoke_hvd

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Stage 1: Training with Horovod (tensorflow/2.15.0 module) ─────────────────
# Run in a subshell so module load changes don't affect Stage 2's conda env.
echo "=== Stage 1: Horovod training (1 epoch, tiny model) ==="
(
  module load tensorflow/2.15.0
  export PYTHONPATH=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts
  # horovodrun --gloo uses the gloo CPU communication backend (no MPI needed).
  # With -np 1 all AllReduce calls are no-ops, but all code paths execute normally.
  horovodrun --gloo -np 1 python3 -u $TRAIN_SCRIPT \
      --run_name         smoke_hvd \
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

# Verify checkpoint was saved
CKPT=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/smoke_hvd/pet_pp.weights.h5
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi
echo "Checkpoint saved: $(ls -lh $CKPT)"

# ── Stage 2: Inference with conda env (module env scoped to subshell above) ───
echo ""
echo "=== Stage 2: Inference smoke test ==="
$CONDA_PYTHON -u $INFER_SCRIPT

INFER_EXIT=$?
echo "Inference exit code: $INFER_EXIT"

if [ $INFER_EXIT -eq 0 ]; then
    echo ""
    echo "SMOKE TEST PASSED"
else
    echo ""
    echo "SMOKE TEST FAILED (inference exit $INFER_EXIT)"
fi

echo "Job ${SLURM_JOB_ID} finished: $(date)"
exit $INFER_EXIT
