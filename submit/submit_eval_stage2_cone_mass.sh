#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --job-name=eval_s2_cm
#SBATCH --output=/pub/lcondren/wprime_signal_mpi/logs/%j_eval_s2_cm.out
#SBATCH --error=/pub/lcondren/wprime_signal_mpi/logs/%j_eval_s2_cm.err

GRID_DIR=/pub/lcondren/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
FIG_DIR=${GRID_DIR}/figures
mkdir -p ${FIG_DIR}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pp-diffusion
NVIDIA_LIBS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/lib')))))")
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}

export PYTHONPATH=/data/homezvol0/lcondren/pp-collision-diffusion/scripts

EPOCH=$(python3 -c "import json; print(json.load(open('${CKPT_DIR}/bsm_grid_event_c_stage1_cfg/training_state.json'))['epochs_done'])" 2>/dev/null || echo "?")

python3 -u /data/homezvol0/lcondren/pp-collision-diffusion/scripts/eval_stage2_cone_mass.py \
    --run_name  bsm_grid_event_c_stage1_cfg \
    --arch      stage1_cfg \
    --ckpt_dir  ${CKPT_DIR} \
    --grid_dir  ${GRID_DIR} \
    --n_events  500 \
    --linear \
    --plot_out  ${FIG_DIR}/stage2_cone_mass_e034_ep${EPOCH}_linear.png \
    --npz_out   ${FIG_DIR}/stage2_cone_mass_e034_ep${EPOCH}_arrays.npz

echo "Done: $(date)"
