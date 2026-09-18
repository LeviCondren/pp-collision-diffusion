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
#SBATCH --job-name=a030_elbo_fixcone
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_a030_elbo_fixcone.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_a030_elbo_fixcone.err

# A030 — ELBO posterior inference on A029 cone+MET-fixed events (s=1.5, s1p5_fixcone/).
# Same protocol as A028 but uses postprocessed NPZs from apply_fix_cone_met.py.
# Comparison: A028 (raw s=1.5) vs A030 (fixed) — does postprocessing improve mass resolution?
# 4 tasks: one per holdout mass point.

MASS_X=(250 250 300 300)
MASS_Y=(250 300 250 300)

MX=${MASS_X[$SLURM_ARRAY_TASK_ID]}
MY=${MASS_Y[$SLURM_ARRAY_TASK_ID]}

SCRIPT=/global/u2/l/lcondren/pp-collision-diffusion/scripts/infer_bsm_mass_posterior_cfg.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
NPZ_DIR=${CKPT_DIR}/bsm_grid_event_c_stage1_cfg/infer_cfg_sweep/s1p5_fixcone
OUT_DIR=${CKPT_DIR}/bsm_grid_event_c_stage1_cfg/mass_inference_a030_s1p5_fixcone

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
    --run_name  bsm_grid_event_c_stage1_cfg \
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
