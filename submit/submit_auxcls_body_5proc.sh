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
#SBATCH --job-name=auxcls_body_5proc
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/logs/%j_auxcls_body_5proc.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/logs/%j_auxcls_body_5proc.err

# E007 — auxcls_body_5proc
# Auxiliary cross-entropy classification loss on the body output (mean-pooled).
# Fixes E006's design flaw: classifier now sees a learned representation, not
# a direct Dense projection of the one-hot label.
# aux_weight=0.1; no CFG dropout; hyperparameters matched to E000/E001/E006.
# Self-resubmitting until training_state.json marks done=true.

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/proc_label_train_auxcls_body.py
DATA_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed
RUN_NAME=auxcls_body_5proc
CKPT_DIR=${DATA_DIR}/checkpoints/${RUN_NAME}
STATE_FILE=${CKPT_DIR}/training_state.json

mkdir -p ${DATA_DIR}/checkpoints/logs
mkdir -p ${CKPT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Nodes: ${SLURM_NODELIST}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint dir: ${CKPT_DIR}"
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
export PYTHONPATH=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts

srun python3 -u $SCRIPT \
    --data_dir          ${DATA_DIR} \
    --run_name          ${RUN_NAME} \
    --processes         dijet zjets ttbar wjets wprime \
    --val_start         400000 \
    --n_val             10000 \
    --batch             128 \
    --epoch             200 \
    --lr                3e-4 \
    --lr_body           1e-4 \
    --num_layers        8 \
    --num_gen_layers    2 \
    --proj_dim          128 \
    --num_part          500 \
    --patience          30 \
    --aux_weight        0.1 \
    --time_limit_hours  3.5

echo "Job ${SLURM_JOB_ID} finished: $(date)"
