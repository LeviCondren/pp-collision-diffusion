#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=e008_trained_ep019
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal/logs/%j_e008_trained_ep019.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal/logs/%j_e008_trained_ep019.err

# Diagnostic inference on 4 TRAINED grid points near the held-out 2×2 block.
# Purpose: compare trained-point vs held-out W1 on MET and jet mass to distinguish
# architectural limitations from interpolation failure.
#
# Trained points: (200,200) (200,350) (350,200) (350,350)
# Same settings as submit_e008_holdout_infer_ep019.sh: 5k events, 500 steps.
# Output: checkpoints_bsm_grid/bsm_grid/infer_trained_ep019_5k/

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/infer_bsm_grid.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal
RUN_NAME=bsm_grid
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_trained_ep019_5k

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
    --grid_dir       ${GRID_DIR}
    --ckpt_dir       ${CKPT_DIR}
    --run_name       ${RUN_NAME}
    --out_dir        ${OUT_DIR}
    --rank           0
    --world_size     1
    --num_steps      500
    --n_total        5000
    --npart          500
    --num_layers     8
    --num_gen_layers 2
    --proj_dim       128
"

echo "=== Trained point (200, 200) ==="
python3 -u $SCRIPT $COMMON_ARGS --m_X 200 --m_Y 200

echo "=== Trained point (200, 350) ==="
python3 -u $SCRIPT $COMMON_ARGS --m_X 200 --m_Y 350

echo "=== Trained point (350, 200) ==="
python3 -u $SCRIPT $COMMON_ARGS --m_X 350 --m_Y 200

echo "=== Trained point (350, 350) ==="
python3 -u $SCRIPT $COMMON_ARGS --m_X 350 --m_Y 350

echo "Job ${SLURM_JOB_ID} finished: $(date)"
echo "Output files:"
ls -lh ${OUT_DIR}/
