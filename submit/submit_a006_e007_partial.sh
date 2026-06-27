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
#SBATCH --job-name=a006_e007_partial
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/auxcls_body_5proc/infer_e007_partial_steps500/logs/%j_a006.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/auxcls_body_5proc/infer_e007_partial_steps500/logs/%j_a006.err

# A006 — E007 partial checkpoint inference (19/200 epochs, auxcls body training)
# Loads auxcls_body checkpoint into base ProcLabelPET; aux classifier layers (functional_3)
# are silently skipped by TF2's H5 loader (confirmed: no base layers missing from checkpoint).
# 1k events/proc × 5 procs = 5k total; 500 steps; --use_true_jet to match A002/A004/A005 protocol.

SCRIPTS=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts
DATA_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed
WPRIME_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_inference
RUN_NAME=auxcls_body_5proc
OUT_DIR=${DATA_DIR}/checkpoints/${RUN_NAME}/infer_e007_partial_steps500

mkdir -p ${OUT_DIR}/logs

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: $(hostname)"
echo "E007 partial (19 epochs) | num_steps=500  n_total=1000  val_start=400000  use_true_jet=True"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=${SCRIPTS}

srun python3 -u ${SCRIPTS}/infer_pp_5proc_truelogn_comparison.py \
    --rank 0 --world_size 1 \
    --data_dir        ${DATA_DIR} \
    --wprime_dir      ${WPRIME_DIR} \
    --run_name        ${RUN_NAME} \
    --val_start       400000 \
    --n_total         1000 \
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
