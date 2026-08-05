#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --job-name=e031_bsm_mpi_l4
#SBATCH --output=/pub/lcondren/wprime_signal_mpi/logs/%j_e031_bsm_mpi_l4.out
#SBATCH --error=/pub/lcondren/wprime_signal_mpi/logs/%j_e031_bsm_mpi_l4.err

# E031 — W' grid (MPI=on) training, layers4 architecture (bsm_grid_event_c_layers4_mpi).
# Resumed from HuggingFace checkpoint (epoch 34/200, val_loss=5.28).
# Data: 144 signal_mX*.hdf5 + background.hdf5 at wprime_signal_mpi/.
# val_start=80000 (train=[0:80k], val=[80k:90k]); n_train=20000 per file.
# 4 GPUs via Horovod. Self-resubmitting + --requeue for preemption recovery.

GRID_DIR=/pub/lcondren/wprime_signal_mpi
RUN_NAME=bsm_grid_event_c_layers4_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
STATE_FILE=${CKPT_DIR}/${RUN_NAME}/training_state.json
CONT_FILE=${CKPT_DIR}/${RUN_NAME}/.next_job
SCRIPT=/data/homezvol0/lcondren/pp-collision-diffusion/scripts/bsm_grid_train_event_c_layers4.py

mkdir -p ${GRID_DIR}/logs
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

horovodrun --gloo -np ${SLURM_NTASKS} $CONDA_PREFIX/bin/python3 -u $SCRIPT \
    --grid_dir           ${GRID_DIR} \
    --ckpt_dir           ${CKPT_DIR} \
    --run_name           ${RUN_NAME} \
    --val_start          80000 \
    --n_train            20000 \
    --n_val              10000 \
    --batch              32 \
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
