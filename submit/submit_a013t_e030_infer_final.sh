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
#SBATCH --job-name=a013t_e030_infer
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/logs/%j_a013t_e030_infer.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/logs/%j_a013t_e030_infer.err

# A013-t — E030 holdout inference at final checkpoint (epoch 153).
# Same config as A012-t but new output dir; checkpoint has more training.
# Architecture: PET_pp_parton_vpar_bsm_event_c_stage1 (num_jet=8).
# 4 SM processes in parallel on 4 GPUs, 5k events each, 500 DDPM steps.
# Truth event features injected (use_truth_jet); wprime skipped (no holdout).

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/sm_5proc_infer_event_c_stage1.py
SM_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed
RUN_NAME=sm_5proc_event_c_stage1
CKPT_DIR=${SM_DIR}/checkpoints_sm_5proc
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_truth_ep153

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
    --holdout_start  490000
    --n_total        5000
    --npart          500
    --num_layers     8
    --num_gen_layers 4
    --num_jet_mlp    512
    --proj_dim       128
    --num_steps      500
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
