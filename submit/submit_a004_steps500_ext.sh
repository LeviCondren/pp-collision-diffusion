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
#SBATCH --job-name=a004_steps500_ext
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/infer_truejet_steps500_ext/logs/%j_a004.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/infer_truejet_steps500_ext/logs/%j_a004.err

# A004 — baseline 500-step inference extension (10k additional events)
# Extends A002-s500 (events 400000-401999) with next 2k/proc (events 402000-403999).
# 2k events/proc × 5 procs = 10k total; uses E000 checkpoint; --use_true_jet.
# val_start=402000 avoids overlap with A002-s500 (val_start=400000, n_total=2000).

SCRIPTS=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts
DATA_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed
WPRIME_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_inference
RUN_NAME=proc_label_5proc_p3
OUT_DIR=${DATA_DIR}/checkpoints/${RUN_NAME}/infer_truejet_steps500_ext

mkdir -p ${OUT_DIR}/logs

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: $(hostname)"
echo "num_steps=500  n_total=2000  val_start=402000  use_true_jet=True"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=${SCRIPTS}

srun python3 -u ${SCRIPTS}/infer_pp_5proc_truelogn_comparison.py \
    --rank 0 --world_size 1 \
    --data_dir        ${DATA_DIR} \
    --wprime_dir      ${WPRIME_DIR} \
    --run_name        ${RUN_NAME} \
    --val_start       402000 \
    --n_total         2000 \
    --num_steps       500 \
    --chunk_size      200 \
    --npart           500 \
    --proj_dim        128 \
    --num_layers      8 \
    --num_gen_layers  2 \
    --processes       dijet zjets ttbar wjets wprime \
    --out_dir         ${OUT_DIR} \
    --use_true_jet

echo "Job ${SLURM_JOB_ID} finished: $(date)"
