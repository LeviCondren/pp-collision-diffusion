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
#SBATCH --job-name=a017t_e030_e2e
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/logs/%j_a017t_e030_e2e.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/logs/%j_a017t_e030_e2e.err

# A017-t — E030 holdout inference at epoch 153, end-to-end (no use_truth_jet).
# Stage-1 (jet model) runs first and its output conditions stage-2 (particle model).
# Identical to A013-t except --use_truth_jet is removed.
# Architecture: PET_pp_parton_vpar_bsm_event_c_stage1 (num_jet_mlp=512).
# 4 SM processes in parallel (one per GPU), 5k events each, 500 DDPM steps.

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/sm_5proc_infer_event_c_stage1.py
SM_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed
RUN_NAME=sm_5proc_event_c_stage1
CKPT_DIR=${SM_DIR}/checkpoints_sm_5proc
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_e2e_10k_ep153

mkdir -p ${SM_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Checkpoint: ${CKPT_DIR}/${RUN_NAME}/pet_pp.weights.h5"
echo "Output dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts

COMMON_ARGS="
    --sm_dir         ${SM_DIR}
    --ckpt_dir       ${CKPT_DIR}
    --run_name       ${RUN_NAME}
    --out_dir        ${OUT_DIR}
    --stats_path     ${CKPT_DIR}/normalisation_stats_sm5proc_stage1.json
    --holdout_start  490000
    --n_total        10000
    --npart          500
    --num_layers     8
    --num_gen_layers 4
    --num_jet_mlp    512
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
echo "Output files:"
ls -lh ${OUT_DIR}/
