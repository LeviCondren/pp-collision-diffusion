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
#SBATCH --job-name=a015_e029_val5k
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_val5k/logs/%j_a015_e029_val5k.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_val5k/logs/%j_a015_e029_val5k.err

# A015 — E029 inference on the new 5k validation MC events (full_event_val5k/).
# Checkpoint: E029 epoch-178 weights (same as A013).
# Input data: /pscratch/sd/l/lcondren/MCsim/full_event_val5k/ (5k events per process).
# holdout_start=0 because val5k files start at index 0.
# Stats files taken from original full_event_mixed/ — normalisation must match training.

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/sm_4proc_infer_event_c_layers4.py
VAL_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_val5k
SM_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed
RUN_NAME=sm_4proc_event_c_layers4_full
CKPT_DIR=${SM_DIR}/checkpoints_sm_4proc
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_val5k_ep178

mkdir -p ${VAL_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Input data: ${VAL_DIR}"
echo "Checkpoint: ${CKPT_DIR}/${RUN_NAME}/pet_pp.weights.h5"
echo "Output dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts

COMMON_ARGS="
    --sm_dir           ${VAL_DIR}
    --ckpt_dir         ${CKPT_DIR}
    --run_name         ${RUN_NAME}
    --out_dir          ${OUT_DIR}
    --stats_path       ${SM_DIR}/normalisation_stats_sm4proc.json
    --stats_event_path ${SM_DIR}/normalisation_stats_event_c_sm4proc.json
    --holdout_start    0
    --n_total          5000
    --npart            500
    --num_layers       8
    --num_gen_layers   4
    --proj_dim         128
    --num_steps        500
    --use_truth_jet
"

echo "=== Launching 4 SM processes in parallel ==="
python3 -u $SCRIPT $COMMON_ARGS --process dijet  --gpu_id 0 &
python3 -u $SCRIPT $COMMON_ARGS --process ttbar  --gpu_id 1 &
python3 -u $SCRIPT $COMMON_ARGS --process wjets  --gpu_id 2 &
python3 -u $SCRIPT $COMMON_ARGS --process zjets  --gpu_id 3 &
wait

echo "Job ${SLURM_JOB_ID} finished: $(date)"
echo "Output files:"
ls -lh ${OUT_DIR}/
