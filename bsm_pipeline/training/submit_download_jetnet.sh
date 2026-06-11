#!/bin/bash
#SBATCH --job-name=download_jetnet
#SBATCH --account=m2616
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/jetnet_data/download_%j.log

JETNET_DIR=/pscratch/sd/l/lcondren/MCsim/jetnet_data
FPCD_DATA_DIR=/pscratch/sd/l/lcondren/MCsim/fpcd_training
SCRIPT_DIR=/global/u2/l/lcondren/ContinuousParamFit/bsm_pipeline/training

mkdir -p $JETNET_DIR $FPCD_DATA_DIR
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

source /global/homes/l/lcondren/.bashrc 2>/dev/null || true
export PATH="/pscratch/sd/l/lcondren/.conda/envs/pipeline_copy-gpu2/bin:$PATH"
export PYTHONPATH=""

# ── Step 1: Install dependencies if needed ────────────────────────────────────
pip install jetnet h5py --quiet --ignore-installed 2>/dev/null || true

# ── Step 2: Download JetNet (exact data used in FPCD paper) ──────────────────
echo "Downloading JetNet datasets..."
python3 -u $SCRIPT_DIR/download_jetnet.py \
    --out-dir $JETNET_DIR \
    --version both

# ── Step 3: Convert to FPCD HDF5 format ──────────────────────────────────────
echo "Preparing FPCD training files..."
python3 -u $SCRIPT_DIR/prepare_fpcd_training.py \
    --jetnet-dir $JETNET_DIR \
    --out-dir    $FPCD_DATA_DIR \
    --version    both

echo "Done at $(date)"
echo "Training data: $FPCD_DATA_DIR"
echo ""
echo "Next steps:"
echo "  git clone https://github.com/ViniciusMikuni/GSGM"
echo "  sbatch submit_train_fpcd.sh"
