#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=a026t_e032_truth
#SBATCH --output=/pub/lcondren/wprime_signal_mpi/logs/%j_a026t_e032_truth.out
#SBATCH --error=/pub/lcondren/wprime_signal_mpi/logs/%j_a026t_e032_truth.err

# A026t — E032 ep127 truth-conditioned inference (stage-2 only).
# --use_truth_jet + --use_true_event bypasses stage-1 entirely: truth log_npart
# and truth event features (MET, cone_pT/mass_X/Y) condition stage-2 directly.
# Diagnostic: if cone masses separate well here, poor separation in A024/A025
# is a stage-1 failure. If they still collapse, it is an intrinsic stage-2 issue.

SCRIPT=/data/homezvol0/lcondren/pp-collision-diffusion/scripts/infer_bsm_grid_event_c_stage1.py
GRID_DIR=/pub/lcondren/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
RUN_NAME=bsm_grid_event_c_stage1_mpi_snap_e127
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_truth_hpc3

mkdir -p ${GRID_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: ${SLURM_NODELIST}"
echo "Checkpoint: ${CKPT_DIR}/${RUN_NAME}/pet_pp.weights.h5"
echo "Output dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pp-diffusion
NVIDIA_LIBS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/lib')))))")
NVIDIA_BINS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/bin')))))")
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}
export PATH=$NVIDIA_BINS:$PATH

export PYTHONPATH=/data/homezvol0/lcondren/pp-collision-diffusion/scripts

COMMON="
    --grid_dir          ${GRID_DIR}
    --ckpt_dir          ${CKPT_DIR}
    --run_name          ${RUN_NAME}
    --out_dir           ${OUT_DIR}
    --n_total           5000
    --num_steps         500
    --chunk_size        50
    --num_gen_layers    2
    --num_jet_mlp       512
    --use_truth_jet
    --use_true_event
"

$CONDA_PREFIX/bin/python3 -u $SCRIPT $COMMON --m_X 250 --m_Y 250 --gpu_id 0 &
$CONDA_PREFIX/bin/python3 -u $SCRIPT $COMMON --m_X 250 --m_Y 300 --gpu_id 1 &
$CONDA_PREFIX/bin/python3 -u $SCRIPT $COMMON --m_X 300 --m_Y 250 --gpu_id 2 &
$CONDA_PREFIX/bin/python3 -u $SCRIPT $COMMON --m_X 300 --m_Y 300 --gpu_id 3 &
wait

echo "Job ${SLURM_JOB_ID} finished: $(date)"
ls -lh ${OUT_DIR}/
