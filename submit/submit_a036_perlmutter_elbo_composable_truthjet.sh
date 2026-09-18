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
#SBATCH --array=0-3
#SBATCH --job-name=a036_elbo_truthjet
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_a036_elbo_truthjet.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_a036_elbo_truthjet.err

# A036 — ELBO posterior inference on A034's truth-conditioned composable-CFG
# generated events (E038 checkpoint, epoch 43/200 snapshot, gs_X=gs_Y=1.5,
# --use_truth_jet). Same scoring script and hypothesis grid as A035, but
# npz_dir points at infer_composable_truthjet_gs1p5 instead of
# infer_composable_gs1p5 — jets_gen in that NPZ is a verified exact echo of
# event_feat_truth (--use_truth_jet bypasses stage-1 sampling), so the event
# conditioning used for scoring is genuine truth-level information rather
# than stage-1's predicted event features. Everything else (model, hypothesis
# grid, parton conditioning construction, n_obs/n_t/chunk) identical to A035
# for direct comparability. 4 tasks: one per holdout mass point.

MASS_X=(250 250 300 300)
MASS_Y=(250 300 250 300)

MX=${MASS_X[$SLURM_ARRAY_TASK_ID]}
MY=${MASS_Y[$SLURM_ARRAY_TASK_ID]}

SCRIPT=/global/u2/l/lcondren/pp-collision-diffusion/scripts/infer_mass_posterior_composable.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
RUN_NAME=bsm_grid_event_c_stage1_cfg_asym_e038_snap_e043
NPZ_DIR=${CKPT_DIR}/${RUN_NAME}/infer_composable_truthjet_gs1p5
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/mass_inference_a036_composable_truthjet_gs1p5

mkdir -p ${GRID_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Array job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started: $(date)"
echo "Task: obs_m_X=${MX}  obs_m_Y=${MY}"
echo "NPZ dir: ${NPZ_DIR}"
echo "Output dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0

export PYTHONPATH=/global/u2/l/lcondren/pp-collision-diffusion/scripts

python3 -u $SCRIPT \
    --npz_dir   ${NPZ_DIR} \
    --grid_dir  ${GRID_DIR} \
    --ckpt_dir  ${CKPT_DIR} \
    --run_name  ${RUN_NAME} \
    --stats_path ${CKPT_DIR}/normalisation_stats_event_c_stage1_cfg.json \
    --obs_m_X   ${MX} \
    --obs_m_Y   ${MY} \
    --n_obs     100 \
    --n_t       100 \
    --chunk     100 \
    --gpu_id    0 \
    --out_dir   ${OUT_DIR} \
    --max_minutes 225

echo "Array task ${SLURM_ARRAY_TASK_ID} finished: $(date)"

# Self-resubmit if the final output NPZ is not yet written (scoring incomplete)
FINAL_NPZ=${OUT_DIR}/posterior_mX${MX}_mY${MY}.npz
if [ ! -f "${FINAL_NPZ}" ]; then
    echo "Output not complete — resubmitting task ${SLURM_ARRAY_TASK_ID}"
    sbatch --array=${SLURM_ARRAY_TASK_ID} "$(realpath $0)"
else
    echo "Output complete: ${FINAL_NPZ}"
fi
