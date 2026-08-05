#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=elbo_corr
#SBATCH --output=/pub/lcondren/wprime_signal_mpi/logs/%j_elbo_corr.out
#SBATCH --error=/pub/lcondren/wprime_signal_mpi/logs/%j_elbo_corr.err

# ELBO log-likelihood vs predicted cone mass correlation analysis.
# Scores inference NPZ events under the truth mass hypothesis, then builds
# a Pearson correlation matrix heatmap.  Runs against all 4 E032 holdout NPZs.
# Usage (default: all 4 holdout points):
#   sbatch submit_elbo_conemass_corr.sh
# To target a single NPZ, set NPZ_PATH:
#   sbatch --export=ALL,NPZ_PATH=.../bsm_mX0300_mY0300_rank00_of01.npz submit_elbo_conemass_corr.sh

GRID_DIR=/pub/lcondren/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
RUN_NAME=bsm_grid_event_c_stage1_mpi_snap_e127
INFER_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_e2e_hpc3
OUT_DIR=${GRID_DIR}/elbo_correlation
SCRIPT=/data/homezvol0/lcondren/pp-collision-diffusion/scripts/plot_elbo_conemass_correlation.py

mkdir -p ${GRID_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: ${SLURM_NODELIST}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pp-diffusion
NVIDIA_LIBS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/lib')))))")
NVIDIA_BINS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/bin')))))")
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}
export PATH=$NVIDIA_BINS:$PATH
export PYTHONPATH=/data/homezvol0/lcondren/pp-collision-diffusion/scripts

if [ -n "${NPZ_PATH}" ]; then
    # Single NPZ mode
    echo "Single NPZ: ${NPZ_PATH}"
    python3 -u ${SCRIPT} \
        --npz_path   ${NPZ_PATH} \
        --ckpt_dir   ${CKPT_DIR} \
        --run_name   ${RUN_NAME} \
        --out_dir    ${OUT_DIR} \
        --n_events   2000 \
        --n_t        50 \
        --chunk      200 \
        --gpu_id     0
else
    # All 4 holdout NPZs
    for NPZ in ${INFER_DIR}/bsm_mX*.npz; do
        echo ""
        echo "=== Scoring: ${NPZ} ==="
        python3 -u ${SCRIPT} \
            --npz_path   ${NPZ} \
            --ckpt_dir   ${CKPT_DIR} \
            --run_name   ${RUN_NAME} \
            --out_dir    ${OUT_DIR} \
            --n_events   2000 \
            --n_t        50 \
            --chunk      200 \
            --gpu_id     0
    done
fi

echo ""
echo "Job ${SLURM_JOB_ID} finished: $(date)"
