#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --job-name=e033_locality
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_e033_locality.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_e033_locality.err

# E033 — per-parton mass conditioning with locality bias (bsm_grid_event_c_locality).
# Architecture: stage-2 replaces global event token with two per-parton mass tokens
# (cone_mass_X/Y), locality-gated by particle proximity to each parton direction.
# Stage-1 ResNet unchanged from E032. Stats file reused from E032.
# 4 GPUs via srun. Self-resubmitting until training_state.json marks done=true.

SCRIPT=/global/u2/l/lcondren/pp-collision-diffusion/scripts/bsm_grid_train_event_c_locality.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
RUN_NAME=bsm_grid_event_c_locality
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
STATE_FILE=${CKPT_DIR}/${RUN_NAME}/training_state.json

mkdir -p ${GRID_DIR}/logs
mkdir -p ${CKPT_DIR}/${RUN_NAME}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Nodes: ${SLURM_NODELIST}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint dir: ${CKPT_DIR}/${RUN_NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0

# ── Stop if already done ──────────────────────────────────────────────────────
if python3 -c "
import json, sys, os
p = '${STATE_FILE}'
if not os.path.exists(p): sys.exit(1)
s = json.load(open(p))
print(f'  epochs_done={s.get(\"epochs_done\",0)}/{s.get(\"total_epochs\",200)}  done={s.get(\"done\",False)}')
sys.exit(0 if s.get('done', False) else 1)
"; then
    echo "Training complete — not resubmitting."
    exit 0
fi

# ── Schedule continuation before running ─────────────────────────────────────
NEXT_JOB=$(sbatch --dependency=afterany:${SLURM_JOB_ID} --parsable "$0")
echo "Next job scheduled: ${NEXT_JOB} (depends on ${SLURM_JOB_ID})"

# ── Run training ──────────────────────────────────────────────────────────────
export PYTHONPATH=/global/u2/l/lcondren/pp-collision-diffusion/scripts

srun python3 -u $SCRIPT \
    --grid_dir           ${GRID_DIR} \
    --ckpt_dir           ${CKPT_DIR} \
    --run_name           ${RUN_NAME} \
    --val_start          80000 \
    --n_train            20000 \
    --n_val              10000 \
    --batch              128 \
    --epoch              200 \
    --lr                 3e-4 \
    --lr_body            1e-4 \
    --num_layers         8 \
    --num_gen_layers     2 \
    --proj_dim           128 \
    --num_jet_mlp        512 \
    --num_part           500 \
    --patience           30 \
    --time_limit_hours   3.5

echo "Job ${SLURM_JOB_ID} finished: $(date)"
