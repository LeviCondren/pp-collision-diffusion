#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --job-name=e035_cfg_sweep
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_e035_cfg_sweep.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_e035_cfg_sweep.err

# E035 — CFG guidance scale sweep on E034 checkpoint (inference only, no retraining).
# Tests s = 1.0, 1.5, 3.0, 5.0, 7.5, 10.0 on 4 holdout mass points:
#   (250,250), (250,300), (300,250), (300,300)
# 24 total runs (6 scales × 4 mass points), 5000 events × 500 steps each.
# Array task ID mapping: task 0-3 = scale 1.0, 4-7 = scale 1.5, ..., 20-23 = scale 10.0
# Output: {CKPT_DIR}/bsm_grid_event_c_stage1_cfg/infer_cfg_sweep/s{SCALE_STR}/

SCALES=("1.0" "1.5" "3.0" "5.0" "7.5" "10.0")
SCALE_STRS=("1p0" "1p5" "3p0" "5p0" "7p5" "10p0")
MASS_X=(250 250 300 300)
MASS_Y=(250 300 250 300)

SCALE_IDX=$((SLURM_ARRAY_TASK_ID / 4))
MASS_IDX=$((SLURM_ARRAY_TASK_ID % 4))

SCALE=${SCALES[$SCALE_IDX]}
SCALE_STR=${SCALE_STRS[$SCALE_IDX]}
MX=${MASS_X[$MASS_IDX]}
MY=${MASS_Y[$MASS_IDX]}

SCRIPT=/global/u2/l/lcondren/pp-collision-diffusion/scripts/infer_bsm_grid_event_c_stage1_cfg.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
OUT_DIR=${CKPT_DIR}/bsm_grid_event_c_stage1_cfg/infer_cfg_sweep/s${SCALE_STR}

mkdir -p ${GRID_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Array job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started: $(date)"
echo "Task: guidance_scale=${SCALE}  m_X=${MX}  m_Y=${MY}"
echo "Output dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0

export PYTHONPATH=/global/u2/l/lcondren/pp-collision-diffusion/scripts

python3 -u $SCRIPT \
    --m_X            ${MX} \
    --m_Y            ${MY} \
    --grid_dir       ${GRID_DIR} \
    --ckpt_dir       ${CKPT_DIR} \
    --guidance_scale ${SCALE} \
    --n_total        5000 \
    --num_steps      500 \
    --chunk_size     100 \
    --out_dir        ${OUT_DIR} \
    --rank           0 \
    --world_size     1 \
    --gpu_id         0

echo "Array task ${SLURM_ARRAY_TASK_ID} finished: $(date)"
