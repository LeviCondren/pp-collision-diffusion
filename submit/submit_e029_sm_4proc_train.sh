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
#SBATCH --job-name=e029_sm_4proc
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/logs/%j_e029_sm_4proc.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/logs/%j_e029_sm_4proc.err

# E029 — SM 4-process full-data training with E022 architecture (event_c, num_gen_layers=4).
# Same as E027 but uses the full training split (all 480k events per process per rank)
# instead of --n_train 20000 (which only used 4% of available data).
# 480k events/rank → 3750 steps/epoch vs E027's 156 steps/epoch.
# Data: dijet, ttbar, wjets, zjets from full_event_mixed/ (500k events each).
# Train [0:480k], validate [480k:490k], holdout [490k:500k] (5k used for inference).
# Self-resubmitting until done=true.

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/sm_4proc_train_event_c_layers4.py
SM_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed
RUN_NAME=sm_4proc_event_c_layers4_full
CKPT_DIR=${SM_DIR}/checkpoints_sm_4proc
STATE_FILE=${CKPT_DIR}/${RUN_NAME}/training_state.json

mkdir -p ${SM_DIR}/logs
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
export PYTHONPATH=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts

srun python3 -u $SCRIPT \
    --sm_dir             ${SM_DIR} \
    --ckpt_dir           ${CKPT_DIR} \
    --run_name           ${RUN_NAME} \
    --val_start          480000 \
    --n_val              10000 \
    --batch              128 \
    --epoch              200 \
    --lr                 3e-4 \
    --lr_body            1e-4 \
    --num_layers         8 \
    --num_gen_layers     4 \
    --proj_dim           128 \
    --num_part           500 \
    --patience           30 \
    --time_limit_hours   3.5

echo "Job ${SLURM_JOB_ID} finished: $(date)"
