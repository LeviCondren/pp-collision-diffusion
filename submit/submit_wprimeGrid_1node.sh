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
#SBATCH --job-name=wprimeGrid
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints/logs/%j_wprimeGrid.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints/logs/%j_wprimeGrid.err

# Train PET_pp_parton on 140 W' grid mass points (144 total minus 4 holdout).
# Holdout region: mX in [300, 350] AND mY in [300, 350] (reserved for inference).
# Events: 10k per file × 140 files = 1.4M total; 9k train / 1k val per file.
# Self-resubmitting every 4h until training_state.json marks done=true.

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/per_parton_cond_train_wprime.py
SIGNAL_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal
RUN_NAME=wprimeGrid
CKPT_DIR=${SIGNAL_DIR}/checkpoints/${RUN_NAME}
STATE_FILE=${CKPT_DIR}/training_state.json

mkdir -p ${SIGNAL_DIR}/checkpoints/logs
mkdir -p ${CKPT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Nodes: ${SLURM_NODELIST}"
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

# ── Schedule next job before running ──────────────────────────────────────────
NEXT_JOB=$(sbatch --dependency=afterany:${SLURM_JOB_ID} --parsable "$0")
echo "Next job scheduled: ${NEXT_JOB} (depends on ${SLURM_JOB_ID})"

# ── Run training ──────────────────────────────────────────────────────────────
export PYTHONPATH=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts

srun python3 -u $SCRIPT \
    --signal_dir         ${SIGNAL_DIR} \
    --run_name           ${RUN_NAME} \
    --holdout_mX_min     300 \
    --holdout_mX_max     350 \
    --holdout_mY_min     300 \
    --holdout_mY_max     350 \
    --n_events_per_file  10000 \
    --val_frac           0.1 \
    --batch              128 \
    --epoch              200 \
    --lr                 3e-4 \
    --lr_body            1e-4 \
    --num_layers         8 \
    --num_gen_layers     2 \
    --proj_dim           128 \
    --num_part           500 \
    --patience           30 \
    --time_limit_hours   3.5

echo "Job ${SLURM_JOB_ID} finished: $(date)"
