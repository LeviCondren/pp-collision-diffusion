#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=e032_mass_infer
#SBATCH --output=/pub/lcondren/wprime_signal_mpi/logs/%j_e032_mass_infer_mX%x.out
#SBATCH --error=/pub/lcondren/wprime_signal_mpi/logs/%j_e032_mass_infer_mX%x.err

# Mass posterior inference — E032 (stage-1, 8-dim jet) snapshot at epoch 127.
# Stage-1 model predicts [log_npart, 7 event features] jointly (num_jet=8).
# Particle head takes only log_npart (jet[:, :1]) internally — handled in script.
# Stats from unified normalisation_stats_event_c_stage1.json (8-dim jet_mean/std).
# Usage:
#   sbatch --export=ALL,OBS_MX=250,OBS_MY=250 submit_e032_mass_infer.sh

OBS_MX=${OBS_MX:-250}
OBS_MY=${OBS_MY:-250}

GRID_DIR=/pub/lcondren/wprime_signal_mpi
SNAP_CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid/bsm_grid_event_c_stage1_mpi_snap_e127
OUT_DIR=${GRID_DIR}/mass_inference_e032_e127
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
    --stats_path         ${GRID_DIR}/checkpoints_bsm_grid/normalisation_stats_event_c_stage1.json \
    --obs_m_X            ${OBS_MX} \
    --obs_m_Y            ${OBS_MY} \
    --val_start          80000 \
    --n_obs              2000 \
    --n_t                200 \
    --chunk              200 \
    --npart              500 \
    --proj_dim           128 \
    --num_layers         8 \
    --num_gen_layers     2 \
    --num_jet            8 \
    --num_jet_mlp        512 \
    --gpu_id             0 \
    --out_dir            ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} finished: $(date)"
