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
#SBATCH --job-name=parton_mixed_5proc
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/logs/%j_mixed.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/logs/%j_mixed.err

# Train per_parton_cond_train.py on 4 SM processes + W'→WZ(500,100) signal:
#   dijet, zjets, ttbar, wjets  (400k train / 10k val each, from 500k files)
#   wprime                       (all ~90k train / 10k val, from 100k file)
#
# val_start=400000 with file-size clamp: SM uses events [0,400k] for train,
# wprime uses all available [0,~100k] for train since it clamps to n_total.
# n_val=10000 gives 10k val events per SM process.
#
# Self-resubmitting: each 4h job resumes from the last checkpoint until
# training_state.json marks done=true.

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/per_parton_cond_train_nlo_vpar.py
DATA_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed
RUN_NAME=parton_mixed_5proc
CKPT_DIR=${DATA_DIR}/checkpoints/${RUN_NAME}
STATE_FILE=${CKPT_DIR}/training_state.json

mkdir -p ${DATA_DIR}/checkpoints/logs
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
    --time_limit_hours  3.5

echo "Job ${SLURM_JOB_ID} finished: $(date)"
