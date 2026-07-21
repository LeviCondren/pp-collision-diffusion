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
#SBATCH --job-name=a020_e031_e2e
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_a020_e031_e2e.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_a020_e031_e2e.err

# A020 — E031 mid-training end-to-end inference on 4 BSM holdout points.
# Architecture: layers4 (num_gen_layers=4). Stage-1 generates log_npart;
# truth event features always condition stage-2 (layers4 design).
# Holdout: (250,250) (250,300) (300,250) (300,300) — each on one GPU.

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/infer_bsm_grid_event_c_layers4.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
RUN_NAME=bsm_grid_event_c_layers4_mpi
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_e2e

mkdir -p ${GRID_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts

COMMON="
    --grid_dir   ${GRID_DIR}
    --ckpt_dir   ${CKPT_DIR}
    --run_name   ${RUN_NAME}
    --out_dir    ${OUT_DIR}
    --n_total    5000
    --num_steps  500
"

python3 -u $SCRIPT $COMMON --m_X 250 --m_Y 250 --gpu_id 0 &
python3 -u $SCRIPT $COMMON --m_X 250 --m_Y 300 --gpu_id 1 &
python3 -u $SCRIPT $COMMON --m_X 300 --m_Y 250 --gpu_id 2 &
python3 -u $SCRIPT $COMMON --m_X 300 --m_Y 300 --gpu_id 3 &
wait

echo "Job ${SLURM_JOB_ID} finished: $(date)"
ls -lh ${OUT_DIR}/
