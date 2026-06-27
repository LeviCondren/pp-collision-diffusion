#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=pet_pp_train
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs/%j.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs/%j.err

PYTHON=/pscratch/sd/l/lcondren/.conda/envs/mg5_new/bin/python3
SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/train_pp.py
RUN_NAME=pet_pp_v1
STATE_FILE=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/${RUN_NAME}/training_state.json

mkdir -p /pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs
mkdir -p /pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/${RUN_NAME}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Check if training is already complete ─────────────────────────────────────
if $PYTHON -c "
import json, sys, os
p = '$STATE_FILE'
if not os.path.exists(p): sys.exit(1)
s = json.load(open(p))
print(f'  epochs_done={s.get(\"epochs_done\",0)}/{s.get(\"total_epochs\",200)}  done={s.get(\"done\",False)}')
sys.exit(0 if s.get('done', False) else 1)
"; then
    echo "Training complete — not resubmitting."
    exit 0
fi

# ── Schedule next job before running (runs after this job, regardless of exit code) ──
NEXT_JOB=$(sbatch --dependency=afterany:${SLURM_JOB_ID} --parsable "$0")
echo "Next job scheduled: ${NEXT_JOB} (dependency on ${SLURM_JOB_ID})"

# ── Run training ──────────────────────────────────────────────────────────────
# --time_limit_hours 3.5 = stop 30 min before the 4h wall time
$PYTHON -u $SCRIPT \
    --run_name          ${RUN_NAME} \
    --batch             128 \
    --epoch             200 \
    --lr                3e-4 \
    --lr_body           1e-4 \
    --num_layers        8 \
    --proj_dim          128 \
    --num_part          500 \
    --patience          30 \
    --n_val             10000 \
    --time_limit_hours  3.5

echo "Job ${SLURM_JOB_ID} finished: $(date)"
