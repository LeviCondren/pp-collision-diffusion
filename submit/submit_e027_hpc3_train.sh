#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=gpu          # or free-gpu for lower priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=e027_sm_4proc
#SBATCH --output=/pub/lcondren/MCsim/full_event_mixed/logs/%j_e027_sm_4proc.out
#SBATCH --error=/pub/lcondren/MCsim/full_event_mixed/logs/%j_e027_sm_4proc.err

# E027 — SM 4-process training (HPC3 version).
# Uses compact data extracted by extract_sm_slim.py (40k events/file).
# val_start=20000 (train=[0:20k], val=[20k:30k], holdout=[30k:40k]).
# Self-resubmitting until done=true.
#
# TODO before submitting:
#   1. Verify TF module name:  module avail tensorflow
#   2. Verify Horovod is installed:  python3 -c "import horovod.tensorflow.keras as hvd; print(hvd.__version__)"
#      If Horovod is NOT available, use submit_e027_hpc3_train_singlegpu.sh instead.
#   3. Confirm GPU count per node and update --gpus= and --ntasks= accordingly.
#   4. Update SM_DIR to wherever the compact HDF5 files were rsync'd.

# ── HPC3 paths ────────────────────────────────────────────────────────────────
SCRIPT=/data/homezvol0/lcondren/pp-collision-diffusion/scripts/sm_4proc_train_singlegpu.py
SM_DIR=/pub/lcondren/MCsim/full_event_mixed       # <-- update if different
RUN_NAME=sm_4proc_event_c_layers4
CKPT_DIR=${SM_DIR}/checkpoints_sm_4proc
STATE_FILE=${CKPT_DIR}/${RUN_NAME}/training_state.json

mkdir -p ${SM_DIR}/logs
mkdir -p ${CKPT_DIR}/${RUN_NAME}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Nodes: ${SLURM_NODELIST}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint dir: ${CKPT_DIR}/${RUN_NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Load modules ──────────────────────────────────────────────────────────────
# TODO: replace with correct module name for HPC3
# Common options: tensorflow/2.15.0  or  ml-stack/tf2.15  or  anaconda/tf-2.15
module load tensorflow/2.15.0     # <-- verify with: module avail tensorflow

# ── Stop if already done ──────────────────────────────────────────────────────
if python3 -c "
import json, sys, os
p = '${STATE_FILE}'
if not os.path.exists(p): sys.exit(1)
s = json.load(open(p))
print(f'  epochs_done={s.get(\"epochs_done\",0)}/{s.get(\"total_epochs\",200)}  done={s.get(\"done\",False)}')
sys.exit(0 if s.get('done', False) else 1)
"; then
    echo "Training complete — not resubmitting."
    exit 0
fi

# ── Schedule continuation before running ─────────────────────────────────────
NEXT_JOB=$(sbatch --dependency=afterany:${SLURM_JOB_ID} --parsable "$0")
echo "Next job scheduled: ${NEXT_JOB} (depends on ${SLURM_JOB_ID})"

# ── Run training ──────────────────────────────────────────────────────────────
export PYTHONPATH=/data/homezvol0/lcondren/pp-collision-diffusion/scripts

python3 -u $SCRIPT \
    --sm_dir             ${SM_DIR} \
    --ckpt_dir           ${CKPT_DIR} \
    --run_name           ${RUN_NAME} \
    --val_start          20000 \
    --n_train            20000 \
    --n_val              10000 \
    --batch              128 \
    --epoch              200 \
    --lr                 3e-4 \
    --lr_body            1e-4 \
    --num_layers         8 \
    --num_gen_layers     4 \
    --proj_dim           128 \
    --num_part           500 \
    --patience           30 \
    --time_limit_hours   3.5

echo "Job ${SLURM_JOB_ID} finished: $(date)"
