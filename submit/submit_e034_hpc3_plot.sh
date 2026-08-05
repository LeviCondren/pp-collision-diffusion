#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=e034_plot
#SBATCH --output=/pub/lcondren/wprime_signal_mpi/logs/%j_e034_plot.out
#SBATCH --error=/pub/lcondren/wprime_signal_mpi/logs/%j_e034_plot.err

# E034 inference plots — run after both infer_e2e and infer_truth are complete.
# Produces the full suite (multiplicity, pT, HT, MET, cone mass, EFPs, etc.)
# for both end-to-end and truth-conditioned outputs, side by side.

SCRIPT=/data/homezvol0/lcondren/pp-collision-diffusion/scripts/plot_infer_wprime_holdout.py
GRID_DIR=/pub/lcondren/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
RUN_NAME=bsm_grid_event_c_stage1_cfg

EPOCH=$(python3 -c "import json; print(json.load(open('${CKPT_DIR}/${RUN_NAME}/training_state.json'))['epochs_done'])" 2>/dev/null || echo "?")

mkdir -p ${GRID_DIR}/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pp-diffusion
NVIDIA_LIBS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/lib')))))")
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$NVIDIA_LIBS
export PYTHONPATH=/data/homezvol0/lcondren/pp-collision-diffusion/scripts

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Epoch: ${EPOCH}"

# ── End-to-end plots ──────────────────────────────────────────────────────────
E2E_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_e2e
E2E_PLOT=${CKPT_DIR}/${RUN_NAME}/plots_e2e_ep${EPOCH}
mkdir -p ${E2E_PLOT}

echo "Plotting E2E inference from ${E2E_DIR} → ${E2E_PLOT}"
python3 -u $SCRIPT \
    --infer_dir ${E2E_DIR} \
    --out_dir   ${E2E_PLOT} \
    --n_events  2000

# ── Truth-conditioned plots ───────────────────────────────────────────────────
TRUTH_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_truth
TRUTH_PLOT=${CKPT_DIR}/${RUN_NAME}/plots_truth_ep${EPOCH}
mkdir -p ${TRUTH_PLOT}

echo "Plotting truth-conditioned inference from ${TRUTH_DIR} → ${TRUTH_PLOT}"
python3 -u $SCRIPT \
    --infer_dir ${TRUTH_DIR} \
    --out_dir   ${TRUTH_PLOT} \
    --n_events  2000

echo "Job ${SLURM_JOB_ID} finished: $(date)"
echo "E2E plots:   ${E2E_PLOT}"
echo "Truth plots: ${TRUTH_PLOT}"
ls -lh ${E2E_PLOT}/ ${TRUTH_PLOT}/
