#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=parton_nlo_npart
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs/%j_parton_nlo_npart.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs/%j_parton_nlo_npart.err

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/per_parton_cond_train_nlo_npart.py
RUN_NAME=parton_v1_nlo_npart
STATE_FILE=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/${RUN_NAME}/training_state.json

mkdir -p /pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs
mkdir -p /pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/${RUN_NAME}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Nodes: ${SLURM_JOB_NODELIST}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

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

# ── Run training: 4 ranks on 1 node ───────────────────────────────────────────
export PYTHONPATH=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts
srun python3 -u $SCRIPT \
    --run_name          ${RUN_NAME} \
    --batch             128 \
    --epoch             200 \
    --lr                3e-4 \
    --lr_body           1e-4 \
    --num_layers        8 \
    --num_gen_layers    2 \
    --proj_dim          128 \
    --num_part          500 \
    --patience          30 \
    --n_val             10000 \
    --time_limit_hours  3.5

echo "Job ${SLURM_JOB_ID} finished: $(date)"
