#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=infer_holdout
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints/wprimeGrid/logs/%j_infer_holdout.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints/wprimeGrid/logs/%j_infer_holdout.err

SCRIPTS=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts
SIGNAL_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal
RUN_NAME=wprimeGrid

mkdir -p ${SIGNAL_DIR}/checkpoints/wprimeGrid/logs

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=${SCRIPTS}

srun python3 -u ${SCRIPTS}/infer_wprime_holdout.py \
    --signal_dir   ${SIGNAL_DIR} \
    --run_name     ${RUN_NAME} \
    --n_events     20000 \
    --num_steps    50 \
    --chunk_size   200 \
    --npart        500 \
    --proj_dim     128 \
    --num_layers   8 \
    --num_gen_layers 2 \
    --gpu_id       0

echo "Job ${SLURM_JOB_ID} finished: $(date)"
