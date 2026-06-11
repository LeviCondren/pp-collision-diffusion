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
#SBATCH --job-name=e002_sampled
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/infer_20k_sampled/logs/%j_e002_sampled.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/infer_20k_sampled/logs/%j_e002_sampled.err

# E002 — infer_truejet_5proc, mode A (stage-1 sampled log_npart)
# Uses E000 checkpoint (proc_label_5proc_p3). Stage-1 ema_jet samples log_npart.
# Writes to infer_20k_sampled/ (separate from the existing infer_20k/ to avoid
# skip-if-exists silently producing zero output).
# Pair with submit_infer_e002_truejet.sh for the true-vs-sampled comparison (E002).

SCRIPTS=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts
DATA_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed
WPRIME_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_inference
RUN_NAME=proc_label_5proc_p3
OUT_DIR=${DATA_DIR}/checkpoints/${RUN_NAME}/infer_20k_sampled

mkdir -p ${OUT_DIR}/logs

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: $(hostname)"
echo "Mode: stage-1 sampled log_npart (no --use_true_jet)"
echo "Checkpoint: ${DATA_DIR}/checkpoints/${RUN_NAME}/pet_pp.weights.h5"
echo "Output: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=${SCRIPTS}

srun python3 -u ${SCRIPTS}/infer_pp_5proc_truelogn_comparison.py \
    --rank 0 --world_size 1 \
    --data_dir    ${DATA_DIR} \
    --wprime_dir  ${WPRIME_DIR} \
    --run_name    ${RUN_NAME} \
    --val_start   400000 \
    --n_total     20000 \
    --num_steps   50 \
    --chunk_size  200 \
    --npart           500 \
    --proj_dim        128 \
    --num_layers      8 \
    --num_gen_layers  2 \
    --processes       dijet zjets ttbar wjets wprime \
    --out_dir         ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} finished: $(date)"
