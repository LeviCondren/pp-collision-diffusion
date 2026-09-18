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
#SBATCH --job-name=a036_elbo_truthjet_n2000
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_a036_elbo_truthjet_n2000.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_a036_elbo_truthjet_n2000.err

# A036 rescoring at n_obs=2000 (up from the original A036's 100), matching A028-A032
# precedent (2000 events x 100 MC timesteps x 144 hypotheses). Scores the
# infer_composable_truthjet_gs1p5_n2000 regeneration (submit_a034_perlmutter_truthjet_n2000.sh)
# instead of the original 100-event infer_composable_truthjet_gs1p5. Same script/model/
# grid/n_t(100) as the original A036, only --npz_dir/--out_dir/--n_obs/--chunk changed.
# chunk raised to 500 (from 100) given the larger n_obs — same self-resubmit-if-incomplete
# pattern as A035/A036 v1, needed here even more: 20x n_obs => roughly 20x scoring compute
# per task, likely several resubmissions through --max_minutes 225 checkpoint/resume before
# a task's posterior_mX*_mY*.npz is written.
#
# Submit with: sbatch --dependency=aftercorr:<gen_job_id> submit_a036_perlmutter_elbo_composable_truthjet_n2000.sh
# (aftercorr so each array task waits on the matching mass-point's regeneration task, not
# the whole regeneration array.)

MASS_X=(250 250 300 300)
MASS_Y=(250 300 250 300)

MX=${MASS_X[$SLURM_ARRAY_TASK_ID]}
MY=${MASS_Y[$SLURM_ARRAY_TASK_ID]}

SCRIPT=/global/u2/l/lcondren/pp-collision-diffusion/scripts/infer_mass_posterior_composable.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
RUN_NAME=bsm_grid_event_c_stage1_cfg_asym_e038_snap_e043
NPZ_DIR=${CKPT_DIR}/${RUN_NAME}/infer_composable_truthjet_gs1p5_n2000
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/mass_inference_a036_composable_truthjet_gs1p5_n2000

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
    --n_obs     2000 \
    --n_t       100 \
    --chunk     500 \
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
