#!/bin/bash
# submit_e023_portable.sh — E023 training submit script for non-NERSC clusters.
#
# BEFORE SUBMITTING: fill in all FILL_IN placeholders below.
# Run `sinfo` and `sacctmgr show qos` to find valid values for your cluster.
#
# To submit:  sbatch submit/submit_e023_portable.sh

# ── Slurm settings — fill in for your cluster ────────────────────────────────
#SBATCH --account=FILL_IN_ACCOUNT
#SBATCH --partition=FILL_IN_GPU_PARTITION   # e.g. gpu, gpu-shared, a100
#SBATCH --nodes=1
#SBATCH --ntasks=1                          # increase to Ngpu if Horovod available
#SBATCH --gpus=1                            # increase to Ngpu if Horovod available
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=e023_bsm_stage1
# Adjust log paths to a writable location on your cluster:
#SBATCH --output=logs/%j_e023_bsm_stage1.out
#SBATCH --error=logs/%j_e023_bsm_stage1.err

# ── Paths — fill in for your cluster ─────────────────────────────────────────
# Root of this cloned repository:
REPO_DIR="FILL_IN_REPO_DIR"   # e.g. /home/user/pp-collision-diffusion

# Where W' HDF5 files live (output of data_generation/generate_wprime_signal.py):
GRID_DIR="FILL_IN_DATA_DIR"   # e.g. /scratch/user/wprime_signal

# Where checkpoints and stats will be written:
CKPT_DIR="${GRID_DIR}/checkpoints_bsm_grid"

# ── Derived ───────────────────────────────────────────────────────────────────
RUN_NAME=bsm_grid_event_c_stage1
SCRIPT="${REPO_DIR}/scripts/bsm_grid_train_event_c_stage1.py"
STATE_FILE="${CKPT_DIR}/${RUN_NAME}/training_state.json"

mkdir -p "${GRID_DIR}/logs"
mkdir -p "${CKPT_DIR}/${RUN_NAME}"

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Nodes: ${SLURM_NODELIST}"
echo "Run name: ${RUN_NAME}"

# ── Activate environment ──────────────────────────────────────────────────────
# If using conda:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pp-diffusion
# If using a venv instead:
# source /path/to/venv/bin/activate

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
export PYTHONPATH="${REPO_DIR}/scripts:${PYTHONPATH:-}"

# If Horovod was installed and you have multiple GPUs, replace the python3 line
# below with:
#   horovodrun -np NGPU python3 -u $SCRIPT ...   (and set --ntasks/--gpus above)
# Otherwise single-GPU:
python3 -u "$SCRIPT" \
    --grid_dir           "${GRID_DIR}" \
    --ckpt_dir           "${CKPT_DIR}" \
    --run_name           "${RUN_NAME}" \
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
