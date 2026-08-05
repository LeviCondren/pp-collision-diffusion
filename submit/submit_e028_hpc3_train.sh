#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --job-name=e028_sm_stage1
#SBATCH --output=/pub/lcondren/MCsim/full_event_mixed/logs/%j_e028_sm_stage1.out
#SBATCH --error=/pub/lcondren/MCsim/full_event_mixed/logs/%j_e028_sm_stage1.err

# E028 — SM 4-process training with E023 architecture (stage-1 8-dim diffusion).
# Same architecture as E023 (PET_pp_parton_vpar_bsm_event_c_stage1, num_jet_mlp=512),
# trained on dijet, ttbar, wjets, zjets from full_event_mixed/.
# val_start=20000 (train=[0:20k], val=[20k:30k], holdout=[30k:40k]).
# Self-resubmitting + --requeue for preemption recovery.

# ── HPC3 paths ────────────────────────────────────────────────────────────────
SCRIPT=/data/homezvol0/lcondren/pp-collision-diffusion/scripts/sm_4proc_train_event_c_stage1.py
SM_DIR=/pub/lcondren/MCsim/full_event_mixed
RUN_NAME=sm_4proc_event_c_stage1
CKPT_DIR=${SM_DIR}/checkpoints_sm_4proc
STATE_FILE=${CKPT_DIR}/${RUN_NAME}/training_state.json
CONT_FILE=${CKPT_DIR}/${RUN_NAME}/.next_job

mkdir -p ${SM_DIR}/logs
mkdir -p ${CKPT_DIR}/${RUN_NAME}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Nodes: ${SLURM_NODELIST}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint dir: ${CKPT_DIR}/${RUN_NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pp-diffusion
NVIDIA_LIBS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/lib')))))")
NVIDIA_BINS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/bin')))))")
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}
export PATH=$NVIDIA_BINS:$PATH

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
# Cancel any stale continuation from a previous requeue of this job ID
if [ -f "$CONT_FILE" ]; then
    STALE_JOB=$(cat "$CONT_FILE")
    if [ "$STALE_JOB" != "$SLURM_JOB_ID" ]; then
        scancel "$STALE_JOB" 2>/dev/null || true
    fi
fi
NEXT_JOB=$(sbatch --dependency=afterany:${SLURM_JOB_ID} --parsable "$0")
echo "$NEXT_JOB" > "$CONT_FILE"
echo "Next job scheduled: ${NEXT_JOB} (depends on ${SLURM_JOB_ID})"

# ── Run training ──────────────────────────────────────────────────────────────
export PYTHONPATH=/data/homezvol0/lcondren/pp-collision-diffusion/scripts

$CONDA_PREFIX/bin/python3 -u $SCRIPT \
    --sm_dir             ${SM_DIR} \
    --ckpt_dir           ${CKPT_DIR} \
    --run_name           ${RUN_NAME} \
    --val_start          20000 \
    --n_train            20000 \
    --n_val              10000 \
    --batch              128 \
    --epoch              200 \
    --lr                 3e-4 \
    --lr_body            1e-4 \
    --num_layers         8 \
    --num_gen_layers     2 \
    --num_jet_mlp        512 \
    --proj_dim           128 \
    --num_part           500 \
    --patience           30 \
    --time_limit_hours   3.5

echo "Job ${SLURM_JOB_ID} finished: $(date)"
