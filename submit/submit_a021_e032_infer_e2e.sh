#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=a021_e032_e2e
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_a021_e032_e2e.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_a021_e032_e2e.err

# A021 — E032 mid-training end-to-end inference on 4 BSM holdout points.
# Architecture: stage-1 (num_gen_layers=2, num_jet_mlp=512, num_jet=8).
# Stage-1 generates all 8 event scalars; stage-2 conditions on those.

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/infer_bsm_grid_event_c_stage1.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
RUN_NAME=bsm_grid_event_c_stage1_mpi
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_e2e

mkdir -p ${GRID_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts

COMMON="
    --grid_dir      ${GRID_DIR}
    --ckpt_dir      ${CKPT_DIR}
    --run_name      ${RUN_NAME}
    --out_dir       ${OUT_DIR}
    --n_total       5000
    --num_steps     500
    --num_gen_layers 2
    --num_jet_mlp   512
"

python3 -u $SCRIPT $COMMON --m_X 250 --m_Y 250 --gpu_id 0 &
python3 -u $SCRIPT $COMMON --m_X 250 --m_Y 300 --gpu_id 1 &
python3 -u $SCRIPT $COMMON --m_X 300 --m_Y 250 --gpu_id 2 &
python3 -u $SCRIPT $COMMON --m_X 300 --m_Y 300 --gpu_id 3 &
wait

echo "Job ${SLURM_JOB_ID} finished: $(date)"
ls -lh ${OUT_DIR}/
