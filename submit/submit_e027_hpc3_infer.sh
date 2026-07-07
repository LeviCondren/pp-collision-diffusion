#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=e027_sm_infer
#SBATCH --output=/pub/lcondren/MCsim/full_event_mixed/logs/%j_e027_sm_infer.out
#SBATCH --error=/pub/lcondren/MCsim/full_event_mixed/logs/%j_e027_sm_infer.err

# E027 holdout inference — HPC3 version.
# Uses compact data (40k events/file); holdout is at [30000:40000].

SCRIPT=/data/homezvol0/lcondren/pp-collision-diffusion/scripts/sm_4proc_infer_event_c_layers4.py
SM_DIR=/pub/lcondren/MCsim/full_event_mixed
RUN_NAME=sm_4proc_event_c_layers4
CKPT_DIR=${SM_DIR}/checkpoints_sm_4proc
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_truth

mkdir -p ${SM_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0     # <-- verify with: module avail tensorflow

export PYTHONPATH=/data/homezvol0/lcondren/pp-collision-diffusion/scripts

COMMON_ARGS="
    --sm_dir         ${SM_DIR}
    --ckpt_dir       ${CKPT_DIR}
    --run_name       ${RUN_NAME}
    --out_dir        ${OUT_DIR}
    --holdout_start  30000
    --n_total        5000
    --npart          500
    --num_layers     8
    --num_gen_layers 4
    --proj_dim       128
    --num_steps      500
"

echo "=== Launching 4 SM processes in parallel ==="
python3 -u $SCRIPT $COMMON_ARGS --process dijet  --gpu_id 0 &
python3 -u $SCRIPT $COMMON_ARGS --process ttbar  --gpu_id 1 &
python3 -u $SCRIPT $COMMON_ARGS --process wjets  --gpu_id 2 &
python3 -u $SCRIPT $COMMON_ARGS --process zjets  --gpu_id 3 &
wait

echo "Job ${SLURM_JOB_ID} finished: $(date)"
ls -lh ${OUT_DIR}/
