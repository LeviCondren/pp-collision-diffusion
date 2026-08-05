#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=e031_mass_infer
#SBATCH --output=/pub/lcondren/wprime_signal_mpi/logs/%j_e031_mass_infer_mX%x.out
#SBATCH --error=/pub/lcondren/wprime_signal_mpi/logs/%j_e031_mass_infer_mX%x.err

# Mass posterior inference — E031 snapshot at epoch 54.
# One job per heldout mass point; submit 4x with different --obs_m_X/Y.
# Usage:
#   sbatch --export=ALL,OBS_MX=250,OBS_MY=250 submit_e031_mass_infer.sh

OBS_MX=${OBS_MX:-250}
OBS_MY=${OBS_MY:-250}

GRID_DIR=/pub/lcondren/wprime_signal_mpi
SNAP_CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid/bsm_grid_event_c_layers4_mpi_snap_e054
OUT_DIR=${GRID_DIR}/mass_inference_e031_e054
SCRIPT=/data/homezvol0/lcondren/pp-collision-diffusion/scripts/infer_bsm_mass_posterior.py

mkdir -p ${GRID_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: ${SLURM_NODELIST}"
echo "Observed mass: m_X=${OBS_MX}  m_Y=${OBS_MY}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pp-diffusion
NVIDIA_LIBS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/lib')))))")
NVIDIA_BINS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/bin')))))")
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}
export PATH=$NVIDIA_BINS:$PATH

export PYTHONPATH=/data/homezvol0/lcondren/pp-collision-diffusion/scripts

$CONDA_PREFIX/bin/python3 -u $SCRIPT \
    --grid_dir           ${GRID_DIR} \
    --ckpt_dir           ${SNAP_CKPT_DIR} \
    --run_name           "" \
    --stats_path         ${GRID_DIR}/normalisation_stats.json \
    --stats_event_path   ${GRID_DIR}/normalisation_stats_event_c.json \
    --obs_m_X            ${OBS_MX} \
    --obs_m_Y            ${OBS_MY} \
    --val_start          80000 \
    --n_obs              500 \
    --n_t                25 \
    --chunk              200 \
    --npart              500 \
    --proj_dim           128 \
    --num_layers         8 \
    --num_gen_layers     4 \
    --gpu_id             0 \
    --out_dir            ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} finished: $(date)"
