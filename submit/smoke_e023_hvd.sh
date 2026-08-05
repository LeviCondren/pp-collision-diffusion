#!/bin/bash
# smoke_e023_hvd.sh — Smoke test for E023: 1 epoch, tiny data, 4-GPU Horovod, no self-resubmit.

# ── Slurm settings — fill in for your cluster ────────────────────────────────
#SBATCH --account=daniel_lab_gpu
#SBATCH --partition=gpu   # e.g. gpu, gpu-shared, a100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=smoke_e023_hvd
#SBATCH --output=/data/homezvol0/lcondren/pp-collision-diffusion/logs/%j_smoke_e023_hvd.out
#SBATCH --error=/data/homezvol0/lcondren/pp-collision-diffusion/logs/%j_smoke_e023_hvd.err

# ── Paths — fill in for your cluster ─────────────────────────────────────────
# Root of this cloned repository:
REPO_DIR="/data/homezvol0/lcondren/pp-collision-diffusion"

# Where W' HDF5 files live (output of data_generation/generate_wprime_signal.py):
GRID_DIR="/pub/lcondren/wprime_signal_smoke"

# Where checkpoints and stats will be written:
CKPT_DIR="${GRID_DIR}/checkpoints_smoke"

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
NVIDIA_LIBS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/lib')))))")
NVIDIA_BINS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/bin')))))")
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}
export PATH=$NVIDIA_BINS:$PATH
# If using a venv instead:
# source /path/to/venv/bin/activate

# ── Run training ──────────────────────────────────────────────────────────────
export PYTHONPATH="${REPO_DIR}/scripts:${PYTHONPATH:-}"

$CONDA_PREFIX/bin/mpirun -np 4 \
    -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX \
    $CONDA_PREFIX/bin/python3 -u "$SCRIPT" \
    --grid_dir           "${GRID_DIR}" \
    --ckpt_dir           "${CKPT_DIR}" \
    --run_name           "${RUN_NAME}" \
    --val_start          250 \
    --n_train            64 \
    --n_val              32 \
    --batch              16 \
    --epoch              1 \
    --lr                 3e-4 \
    --lr_body            1e-4 \
    --num_layers         8 \
    --num_gen_layers     2 \
    --proj_dim           128 \
    --num_jet_mlp        512 \
    --num_part           500 \
    --patience           30 \
    --time_limit_hours   0.4

echo "Job ${SLURM_JOB_ID} finished: $(date)"
