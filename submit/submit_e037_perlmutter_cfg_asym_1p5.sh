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
#SBATCH --array=0-7
#SBATCH --job-name=e037_asym_1p5
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_e037_asym_1p5.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_e037_asym_1p5.err

# E037 sanity check — Asymmetric CFG at gs_X=1.5, gs_Y=1.5 on E036 checkpoint.
# 4 holdout mass points: (250,250), (250,300), (300,250), (300,300).
# 5000 events per mass point, split into 2 parts of 2500 to fit within the 4h wall time.
#
# Task mapping: task_id = rank * 4 + mass_idx
#   tasks 0-3: rank 0 (events [0,    2500))
#   tasks 4-7: rank 1 (events [2500, 5000))
#
# Each task takes ~3.1h (50 chunks × ~224 sec). Output files:
#   bsm_mX{MX}_mY{MY}_rank00_of02.npz  (first  half)
#   bsm_mX{MX}_mY{MY}_rank01_of02.npz  (second half)
# Skip-if-exists guard means re-running is safe.

GS_X=1.5
GS_Y=1.5

MASS_X=(250 250 300 300)
MASS_Y=(250 300 250 300)

MASS_IDX=$((SLURM_ARRAY_TASK_ID % 4))
RANK=$((SLURM_ARRAY_TASK_ID / 4))
WORLD_SIZE=2

MX=${MASS_X[$MASS_IDX]}
MY=${MASS_Y[$MASS_IDX]}

SCRIPT=/global/u2/l/lcondren/pp-collision-diffusion/scripts/infer_asym.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
STATS_PATH=${CKPT_DIR}/normalisation_stats_event_c_stage1_cfg.json
OUT_DIR=${CKPT_DIR}/bsm_grid_event_c_stage1_cfg_asym/infer_cfg_sweep/sX1p5_Y1p5

mkdir -p ${GRID_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Array job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started: $(date)"
echo "Task: guidance_scale_X=${GS_X}  guidance_scale_Y=${GS_Y}  m_X=${MX}  m_Y=${MY}  rank=${RANK}/${WORLD_SIZE}"
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
    --rank             ${RANK} \
    --world_size       ${WORLD_SIZE} \
    --gpu_id           0

echo "Array task ${SLURM_ARRAY_TASK_ID} finished: $(date)"
