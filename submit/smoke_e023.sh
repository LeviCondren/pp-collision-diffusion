#!/bin/bash
# Smoke test for E023 — 1 epoch, tiny data, no self-resubmit.
#SBATCH --account=daniel_lab_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=smoke_e023
#SBATCH --output=/data/homezvol0/lcondren/pp-collision-diffusion/logs/%j_smoke_e023.out
#SBATCH --error=/data/homezvol0/lcondren/pp-collision-diffusion/logs/%j_smoke_e023.err

REPO_DIR="/data/homezvol0/lcondren/pp-collision-diffusion"
GRID_DIR="/pub/lcondren/wprime_signal_smoke"
CKPT_DIR="/pub/lcondren/wprime_signal_smoke/checkpoints_smoke"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pp-diffusion
NVIDIA_LIBS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/lib')))))")
NVIDIA_BINS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/bin')))))")
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}
export PATH=$NVIDIA_BINS:$PATH

export PYTHONPATH="${REPO_DIR}/scripts:${PYTHONPATH:-}"
mkdir -p "${CKPT_DIR}/bsm_grid_event_c_stage1"

echo "Smoke E023 started: $(date)"
python3 -u "${REPO_DIR}/scripts/bsm_grid_train_event_c_stage1.py" \
    --grid_dir           "${GRID_DIR}" \
    --ckpt_dir           "${CKPT_DIR}" \
    --run_name           bsm_grid_event_c_stage1 \
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
    --time_limit_hours   0.4 \
    --no_background
echo "Smoke E023 finished: $(date)"
