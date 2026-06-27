#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=a002_steps100
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/infer_truejet_steps100/logs/%j_a002_steps100.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/infer_truejet_steps100/logs/%j_a002_steps100.err

# A002 — diffusion-step sweep, 100 steps, true log_npart
# Diagnostic: test whether eta over-dispersion improves with more diffusion steps.
# Uses E000 checkpoint (proc_label_5proc_p3), --use_true_jet to isolate stage-2.
# 5000 events/process × 5 processes; estimated ~93 min.

SCRIPTS=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts
DATA_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed
WPRIME_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_inference
RUN_NAME=proc_label_5proc_p3
OUT_DIR=${DATA_DIR}/checkpoints/${RUN_NAME}/infer_truejet_steps100

mkdir -p ${OUT_DIR}/logs

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: $(hostname)"
echo "num_steps=100  n_total=5000  use_true_jet=True"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=${SCRIPTS}

srun python3 -u ${SCRIPTS}/infer_pp_5proc_truelogn_comparison.py \
    --rank 0 --world_size 1 \
    --data_dir        ${DATA_DIR} \
    --wprime_dir      ${WPRIME_DIR} \
    --run_name        ${RUN_NAME} \
    --val_start       400000 \
    --n_total         5000 \
    --num_steps       100 \
    --chunk_size      200 \
    --npart           500 \
    --proj_dim        128 \
    --num_layers      8 \
    --num_gen_layers  2 \
    --processes       dijet zjets ttbar wjets wprime \
    --out_dir         ${OUT_DIR} \
    --use_true_jet

echo "Job ${SLURM_JOB_ID} finished: $(date)"
