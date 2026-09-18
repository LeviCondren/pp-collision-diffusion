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
#SBATCH --array=0-39
#SBATCH --job-name=e037_asym_sweep
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_e037_asym_sweep.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_e037_asym_sweep.err

# E037 — Asymmetric CFG guidance scale sweep on E036 checkpoint (inference only).
# Tests 10 (gs_X, gs_Y) pairs on 4 holdout mass points: (250,250), (250,300), (300,250), (300,300).
# 40 total tasks (10 pairs × 4 mass points), 5000 events × 500 steps each.
#
# Scale pairs:
#   0: (1.0, 1.0)  — symmetric baseline
#   1: (1.5, 1.5)
#   2: (3.0, 3.0)
#   3: (5.0, 5.0)
#   4: (7.5, 7.5)
#   5: (10.0,10.0)
#   6: (3.0, 1.0)  — X-heavy
#   7: (1.0, 3.0)  — Y-heavy
#   8: (5.0, 3.0)  — X-heavier
#   9: (3.0, 5.0)  — Y-heavier
#
# Task ID mapping: task_id = scale_pair_idx * 4 + mass_idx
# Output: {CKPT_DIR}/bsm_grid_event_c_stage1_cfg_asym/infer_cfg_sweep/sX{GS_X}_Y{GS_Y}/
#
# Stats: reuses E034/E036 shared stats (normalisation_stats_event_c_stage1_cfg.json)
# Checkpoint: E036 (bsm_grid_event_c_stage1_cfg_asym/pet_pp.weights.h5)

SCALES_X=(1.0  1.5  3.0  5.0  7.5  10.0  3.0  1.0  5.0  3.0)
SCALES_Y=(1.0  1.5  3.0  5.0  7.5  10.0  1.0  3.0  3.0  5.0)
SCALE_STRS=("1p0_Y1p0" "1p5_Y1p5" "3p0_Y3p0" "5p0_Y5p0" "7p5_Y7p5" "10p0_Y10p0" "3p0_Y1p0" "1p0_Y3p0" "5p0_Y3p0" "3p0_Y5p0")

MASS_X=(250 250 300 300)
MASS_Y=(250 300 250 300)

SCALE_IDX=$((SLURM_ARRAY_TASK_ID / 4))
MASS_IDX=$((SLURM_ARRAY_TASK_ID % 4))

GS_X=${SCALES_X[$SCALE_IDX]}
GS_Y=${SCALES_Y[$SCALE_IDX]}
SCALE_STR=${SCALE_STRS[$SCALE_IDX]}
MX=${MASS_X[$MASS_IDX]}
MY=${MASS_Y[$MASS_IDX]}

SCRIPT=/global/u2/l/lcondren/pp-collision-diffusion/scripts/infer_asym.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
STATS_PATH=${CKPT_DIR}/normalisation_stats_event_c_stage1_cfg.json
OUT_DIR=${CKPT_DIR}/bsm_grid_event_c_stage1_cfg_asym/infer_cfg_sweep/sX${SCALE_STR}

mkdir -p ${GRID_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Array job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started: $(date)"
echo "Task: guidance_scale_X=${GS_X}  guidance_scale_Y=${GS_Y}  m_X=${MX}  m_Y=${MY}"
echo "Output dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0

export PYTHONPATH=/global/u2/l/lcondren/pp-collision-diffusion/scripts

python3 -u $SCRIPT \
    --m_X              ${MX} \
    --m_Y              ${MY} \
    --grid_dir         ${GRID_DIR} \
    --ckpt_dir         ${CKPT_DIR} \
    --stats_path       ${STATS_PATH} \
    --guidance_scale_X ${GS_X} \
    --guidance_scale_Y ${GS_Y} \
    --n_total          5000 \
    --num_steps        500 \
    --chunk_size       50 \
    --out_dir          ${OUT_DIR} \
    --rank             0 \
    --world_size       1 \
    --gpu_id           0

echo "Array task ${SLURM_ARRAY_TASK_ID} finished: $(date)"
