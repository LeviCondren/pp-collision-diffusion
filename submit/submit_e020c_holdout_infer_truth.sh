#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=e020c_infer_truth
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal/logs/%j_e020c_infer_truth.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal/logs/%j_e020c_infer_truth.err

# E020c holdout inference — TRUTH-CONDITIONED variant.
# Passes truth event features (MET computed from HDF5 truth particles) to E020a.
# 4 holdout points in parallel on 4 GPUs: (250,250) (250,300) (300,250) (300,300)
# 5000 events each, 500 steps.

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/infer_bsm_grid_event_c.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal
RUN_NAME=bsm_grid_event_c
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_truth

mkdir -p ${GRID_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Checkpoint: ${CKPT_DIR}/${RUN_NAME}/pet_pp.weights.h5"
echo "Output dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts

COMMON_ARGS="
    --grid_dir       ${GRID_DIR}
    --ckpt_dir       ${CKPT_DIR}
    --run_name       ${RUN_NAME}
    --out_dir        ${OUT_DIR}
    --world_size     1
    --rank           0
    --num_steps      500
    --n_total        5000
    --npart          500
    --num_layers     8
    --num_gen_layers 2
    --proj_dim       128
"

echo "=== Launching 4 holdout points in parallel ==="
python3 -u $SCRIPT $COMMON_ARGS --gpu_id 0 --m_X 250 --m_Y 250 &
python3 -u $SCRIPT $COMMON_ARGS --gpu_id 1 --m_X 250 --m_Y 300 &
python3 -u $SCRIPT $COMMON_ARGS --gpu_id 2 --m_X 300 --m_Y 250 &
python3 -u $SCRIPT $COMMON_ARGS --gpu_id 3 --m_X 300 --m_Y 300 &
wait

echo "Job ${SLURM_JOB_ID} finished: $(date)"
echo "Output files:"
ls -lh ${OUT_DIR}/
