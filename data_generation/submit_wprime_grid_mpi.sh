#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=wprime_grid_mpi
#SBATCH --array=0-143
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_wprime.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%A_%a_wprime.err

# E028 — W' signal grid regeneration with MPI ON
# Same 144-point mass grid as E026 but PartonLevel:MPI=on (matching SM data).
# Writes to wprime_signal_mpi/ to preserve old MPI-off data used by E008/E020.
# Time raised to 2h (MPI adds ~30-40% event generation cost).

SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/fpcd_full_event/generate_wprime_signal.py
OUT_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi

mkdir -p ${OUT_DIR}/logs

echo "Job ${SLURM_JOB_ID}  array task ${SLURM_ARRAY_TASK_ID}  started: $(date)"
mX=$((50 + (SLURM_ARRAY_TASK_ID / 12) * 50))
mY=$((50 + (SLURM_ARRAY_TASK_ID % 12) * 50))
echo "Mass point: mX=${mX} mY=${mY}"

/pscratch/sd/l/lcondren/.conda/envs/mg5_new/bin/python3 \
  ${SCRIPT} \
  --task-id ${SLURM_ARRAY_TASK_ID} \
  --nevents 100000 \
  --out     ${OUT_DIR}

echo "Job ${SLURM_JOB_ID}  array task ${SLURM_ARRAY_TASK_ID}  finished: $(date)"
