#!/bin/bash
# submit_wprime_grid_portable.sh — generate the W' signal mass grid on a generic cluster.
#
# Generates signal_mX{mX:04d}_mY{mY:04d}.hdf5 files for the 12×12 mass grid:
#   mX, mY ∈ {50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600} GeV
# Each point takes ~5-10 minutes on a CPU node with 4 cores.
#
# BEFORE SUBMITTING: fill in all FILL_IN placeholders.
# Run as a Slurm array: sbatch data_generation/submit_wprime_grid_portable.sh

# ── Slurm settings — fill in for your cluster ────────────────────────────────
#SBATCH --account=FILL_IN_ACCOUNT
#SBATCH --partition=FILL_IN_CPU_PARTITION   # CPU partition — signal gen needs no GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=wprime_grid
#SBATCH --output=logs/%j_%a_wprime_grid.out
#SBATCH --error=logs/%j_%a_wprime_grid.err
#SBATCH --array=0-143   # 144 mass-point pairs (12×12)

# ── Paths — fill in for your cluster ─────────────────────────────────────────
REPO_DIR="FILL_IN_REPO_DIR"
OUT_DIR="FILL_IN_DATA_DIR"   # same GRID_DIR used in the training submit scripts

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pp-diffusion

# ── Mass grid ─────────────────────────────────────────────────────────────────
MASSES=(50 100 150 200 250 300 350 400 450 500 550 600)
N_MASSES=${#MASSES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
MX_IDX=$(( IDX / N_MASSES ))
MY_IDX=$(( IDX % N_MASSES ))
MX=${MASSES[$MX_IDX]}
MY=${MASSES[$MY_IDX]}

mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/../logs" 2>/dev/null || mkdir -p logs

echo "Array task ${IDX}: mX=${MX} mY=${MY}"
OUTFILE="${OUT_DIR}/signal_mX$(printf '%04d' ${MX})_mY$(printf '%04d' ${MY}).hdf5"

if [ -f "${OUTFILE}" ]; then
    echo "Already exists: ${OUTFILE} — skipping."
    exit 0
fi

python3 "${REPO_DIR}/data_generation/generate_wprime_signal.py" \
    --process signal \
    --mass-x "${MX}" \
    --mass-y "${MY}" \
    --nevents 100000 \
    --seed $(( MX * 1000 + MY )) \
    --out "${OUT_DIR}"

echo "Done: ${OUTFILE}"
