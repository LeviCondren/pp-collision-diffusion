#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=e033_e2e_infer
#SBATCH --output=/pub/lcondren/wprime_signal_mpi/logs/%j_e033_e2e_infer.out
#SBATCH --error=/pub/lcondren/wprime_signal_mpi/logs/%j_e033_e2e_infer.err

# E033 end-to-end inference on 4 BSM holdout points (current checkpoint).
# Architecture: per-parton mass conditioning with locality bias.
# Stage-1 identical to E032; stage-2 uses gated mass tokens.
# All 4 holdout mass points run in parallel, one per GPU.

SCRIPT=/data/homezvol0/lcondren/pp-collision-diffusion/scripts/infer_bsm_grid_event_c_locality.py
SUBMIT_SCRIPT=/data/homezvol0/lcondren/pp-collision-diffusion/submit/submit_e033_hpc3_infer_e2e.sh
GRID_DIR=/pub/lcondren/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
RUN_NAME=bsm_grid_event_c_locality
OUT_DIR=${CKPT_DIR}/${RUN_NAME}/infer_holdout_e2e_hpc3

mkdir -p ${GRID_DIR}/logs
mkdir -p ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: ${SLURM_NODELIST}"
echo "Checkpoint: ${CKPT_DIR}/${RUN_NAME}/pet_pp.weights.h5"
echo "Output dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pp-diffusion
NVIDIA_LIBS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/lib')))))")
NVIDIA_BINS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/bin')))))")
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}
export PATH=$NVIDIA_BINS:$PATH

export PYTHONPATH=/data/homezvol0/lcondren/pp-collision-diffusion/scripts

COMMON="
    --grid_dir          ${GRID_DIR}
    --ckpt_dir          ${CKPT_DIR}
    --run_name          ${RUN_NAME}
    --out_dir           ${OUT_DIR}
    --n_total           5000
    --num_steps         500
    --chunk_size        50
    --num_gen_layers    2
    --num_jet_mlp       512
    --time_budget_hours 3.5
    --submit_script     ${SUBMIT_SCRIPT}
"

$CONDA_PREFIX/bin/python3 -u $SCRIPT $COMMON --m_X 250 --m_Y 250 --gpu_id 0 &
$CONDA_PREFIX/bin/python3 -u $SCRIPT $COMMON --m_X 250 --m_Y 300 --gpu_id 1 &
$CONDA_PREFIX/bin/python3 -u $SCRIPT $COMMON --m_X 300 --m_Y 250 --gpu_id 2 &
$CONDA_PREFIX/bin/python3 -u $SCRIPT $COMMON --m_X 300 --m_Y 300 --gpu_id 3 &
wait

echo "Job ${SLURM_JOB_ID} finished: $(date)"
ls -lh ${OUT_DIR}/

# If any output file is still missing the Python processes will have already
# resubmitted via sbatch (lock file prevents duplicate submissions).
# Report status for the log.
all_done=1
for tag in bsm_mX0250_mY0250 bsm_mX0250_mY0300 bsm_mX0300_mY0250 bsm_mX0300_mY0300; do
    f="${OUT_DIR}/${tag}_rank00_of01.npz"
    if [ ! -f "$f" ]; then
        echo "MISSING: ${f}"
        all_done=0
    fi
done
if [ "$all_done" -eq 1 ]; then
    echo "All 4 mass points complete."
    rm -f "${OUT_DIR}/.resubmit.lock"
fi
