#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=e008_holdout_infer
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal/logs/%j_e008_holdout_infer.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal/logs/%j_e008_holdout_infer.err

# E008 holdout inference — 4 held-out mass points for interpolation validation.
# Each point is run as a separate infer_bsm_grid.py invocation so it produces
# its own .npz file, making the 4 points separable as "processes" in plotting.
#
# Holdout points:  (250,250)  (250,300)  (300,250)  (300,300)
# Output dir:  {CKPT_DIR}/bsm_grid/infer_holdout_steps500/
#   bsm_mX0250_mY0250_rank00_of01.npz
#   bsm_mX0250_mY0300_rank00_of01.npz
#   bsm_mX0300_mY0250_rank00_of01.npz
#   bsm_mX0300_mY0300_rank00_of01.npz
#
# Submit AFTER E008 training completes (checkpoint must exist).

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/infer_bsm_grid.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal
RUN_NAME=bsm_grid
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_steps500

CKPT_PATH=${CKPT_DIR}/${RUN_NAME}/pet_pp.weights.h5

if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: checkpoint not found at ${CKPT_PATH}" >&2
    exit 1
fi

mkdir -p ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Checkpoint: ${CKPT_PATH}"
echo "Output dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0

export PYTHONPATH=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts
export CUDA_VISIBLE_DEVICES=0

COMMON_ARGS="
    --grid_dir    ${GRID_DIR}
    --ckpt_dir    ${CKPT_DIR}
    --run_name    ${RUN_NAME}
    --out_dir     ${OUT_DIR}
    --rank        0
    --world_size  1
    --num_steps   500
    --n_total     10000
    --npart       500
    --num_layers  8
    --proj_dim    128
"

# Run each holdout point separately so each produces its own labelled .npz file.
# A plotting script loads these 4 files and treats each (m_X, m_Y) as a process.

echo "=== Holdout point (250, 250) ==="
python3 -u $SCRIPT $COMMON_ARGS --m_X 250 --m_Y 250

echo "=== Holdout point (250, 300) ==="
python3 -u $SCRIPT $COMMON_ARGS --m_X 250 --m_Y 300

echo "=== Holdout point (300, 250) ==="
python3 -u $SCRIPT $COMMON_ARGS --m_X 300 --m_Y 250

echo "=== Holdout point (300, 300) ==="
python3 -u $SCRIPT $COMMON_ARGS --m_X 300 --m_Y 300

echo "Job ${SLURM_JOB_ID} finished: $(date)"
echo "Output files:"
ls -lh ${OUT_DIR}/
