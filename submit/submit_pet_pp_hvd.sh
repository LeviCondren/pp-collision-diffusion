#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=pet_pp_hvd
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs/%j_hvd.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs/%j_hvd.err

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/train_pp_horovod.py
RUN_NAME=pet_pp_v1_hvd
STATE_FILE=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/${RUN_NAME}/training_state.json

mkdir -p /pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs
mkdir -p /pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/${RUN_NAME}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Nodes: ${SLURM_JOB_NODELIST}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Use NERSC tensorflow module — includes Horovod built against OpenMPI
module load tensorflow/2.15.0

# ── Check if training is already complete ─────────────────────────────────────
if python3 -c "
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

# ── Schedule next job before running ──────────────────────────────────────────
NEXT_JOB=$(sbatch --dependency=afterany:${SLURM_JOB_ID} --parsable "$0")
echo "Next job scheduled: ${NEXT_JOB} (dependency on ${SLURM_JOB_ID})"

# ── Run training with srun (MPI launcher for Horovod) ─────────────────────────
# 16 ranks total (4 nodes × 4 GPUs each); srun assigns one rank per GPU
srun python3 -u $SCRIPT \
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
