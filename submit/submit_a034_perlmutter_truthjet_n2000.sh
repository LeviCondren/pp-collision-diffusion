#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --job-name=a034_truthjet_n2000
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_a034_truthjet_n2000.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_a034_truthjet_n2000.err

# A034 regeneration at n_total=2000 (up from the original A034's 100), to give A036's
# ELBO rescoring enough events to match A028-A032 precedent (2000 events x 100 MC
# timesteps x 144 hypotheses). Same checkpoint/formula/gs=1.5/--use_truth_jet as the
# original A034 (scripts/infer_composable.py) — only n_total and chunk_size change.
# Writes to a NEW output dir (infer_composable_truthjet_gs1p5_n2000) so the original
# 100-event A034 output is untouched.
#
# No built-in checkpoint/resume in infer_composable.py — if a task doesn't finish
# within the time limit it needs a clean resubmit, not a partial recovery. 10h budget
# is generous headroom over the naive ~20x-linear estimate from the original A034 run
# (100 events in ~11-13 min at chunk_size=50 => ~2000 events in ~3.7-4.3h at the same
# per-event cost; chunk_size raised to 100 here for better GPU utilization, expected
# to help not hurt).

MASS_X=(250 250 300 300)
MASS_Y=(250 300 250 300)

MX=${MASS_X[$SLURM_ARRAY_TASK_ID]}
MY=${MASS_Y[$SLURM_ARRAY_TASK_ID]}

SCRIPT=/global/u2/l/lcondren/pp-collision-diffusion/scripts/infer_composable.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
RUN_NAME=bsm_grid_event_c_stage1_cfg_asym_e038_snap_e043
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_composable_truthjet_gs1p5_n2000
STATS_PATH=${CKPT_DIR}/normalisation_stats_event_c_stage1_cfg.json

mkdir -p ${GRID_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Array job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started: $(date)"
echo "Task: m_X=${MX}  m_Y=${MY}"
echo "Output dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=/global/u2/l/lcondren/pp-collision-diffusion/scripts

python3 -u $SCRIPT \
    --m_X                 ${MX} \
    --m_Y                 ${MY} \
    --grid_dir            ${GRID_DIR} \
    --ckpt_dir             ${CKPT_DIR} \
    --run_name             ${RUN_NAME} \
    --stats_path           ${STATS_PATH} \
    --out_dir              ${OUT_DIR} \
    --rank                 0 \
    --world_size            1 \
    --gpu_id                0 \
    --num_steps           500 \
    --chunk_size           100 \
    --n_total             2000 \
    --use_truth_jet \
    --guidance_scale_X     1.5 \
    --guidance_scale_Y     1.5

echo "Array task ${SLURM_ARRAY_TASK_ID} finished: $(date)"

OUT_FILE=${OUT_DIR}/bsm_mX$(printf '%04.0f' ${MX})_mY$(printf '%04.0f' ${MY})_rank00_of01.npz
if [ -f "${OUT_FILE}" ]; then
    echo "Output complete: ${OUT_FILE}"
else
    echo "WARNING: expected output not found: ${OUT_FILE}"
fi
