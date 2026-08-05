#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=e034_truth_infer
#SBATCH --output=/pub/lcondren/wprime_signal_mpi/logs/%j_e034_truth_infer.out
#SBATCH --error=/pub/lcondren/wprime_signal_mpi/logs/%j_e034_truth_infer.err

# E034 truth-conditioned inference on 4 BSM holdout points.
# --use_truth_jet + --use_true_event bypasses stage-1 entirely: truth log_npart
# and truth event features (including cone_mass_X/Y) condition stage-2 directly.
# Single-pass (no CFG), so 2h wall time is sufficient for 2000 events.
# Diagnostic: isolates stage-2 generation quality from stage-1 predictions.

SCRIPT=/data/homezvol0/lcondren/pp-collision-diffusion/scripts/infer_bsm_grid_event_c_stage1_cfg.py
GRID_DIR=/pub/lcondren/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
RUN_NAME=bsm_grid_event_c_stage1_cfg
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_truth

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
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$NVIDIA_LIBS
export PATH=$NVIDIA_BINS:$PATH

export PYTHONPATH=/data/homezvol0/lcondren/pp-collision-diffusion/scripts

COMMON="
    --grid_dir          ${GRID_DIR}
    --ckpt_dir          ${CKPT_DIR}
    --run_name          ${RUN_NAME}
    --out_dir           ${OUT_DIR}
    --n_total           2000
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
