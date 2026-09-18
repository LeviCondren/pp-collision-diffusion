#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --job-name=e038_cfg_asym_hidrop
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_e038_cfg_asym_hidrop.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_e038_cfg_asym_hidrop.err

# E038 — Asymmetric per-cone CFG dropout, raised dropout rate (bsm_grid_event_c_stage1_cfg_asym_e038).
# Same architecture and script as E036 (bsm_grid_train_event_c_stage1_cfg_asym.py); only
# cfg_drop_x_prob / cfg_drop_y_prob change, and the checkpoint dir is separate so E036's
# checkpoint is untouched.
#
# Motivation: E036 used independent Bernoulli(0.10) per cone, giving
#   drop_none=81%  drop_X=10%  drop_Y=10%  drop_both(=v_null)=1%
# E037's asym-CFG inference (generate_asym_cfg) leans on v_null as its base anchor, and at
# only 1% training exposure that branch is badly calibrated — this was diagnosed as a
# contributor to E037's cone-mass quality regression vs E035 (symmetric CFG). Independent
# Bernoulli(0.35) per cone raises the joint-drop rate to ~12.25% (E034's symmetric CFG used
# a flat 15% joint-drop rate for comparison), and as a side effect also raises the marginal
# per-cone branches (v_no_x, v_no_y) from 10%->22.75% each, which is what the new
# generate_composable_cfg (E038 formula, see PET_pp_parton_vpar_bsm_event_c_stage1_cfg_asym.py)
# uses as its guidance anchors instead of v_full.
#   drop_none=42.25%  drop_X=22.75%  drop_Y=22.75%  drop_both(=v_null)=12.25%
#
# Inference: intended for generate_composable_cfg / DDPMSamplerComposableCFG (E038 formula,
# v_guided = v_null + gs_x*(v_no_y-v_null) + gs_y*(v_no_x-v_null); v_full is not referenced,
# so it cannot double-count v_full the way E036/E037's generate_asym_cfg does).
# Stats reused from E034 (identical data, identical normalization format).
# 4 GPUs via srun. Self-resubmitting until training_state.json marks done=true.

SCRIPT=/global/u2/l/lcondren/pp-collision-diffusion/scripts/bsm_grid_train_event_c_stage1_cfg_asym.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
RUN_NAME=bsm_grid_event_c_stage1_cfg_asym_e038
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
STATE_FILE=${CKPT_DIR}/${RUN_NAME}/training_state.json

# Reuse E034 stats (same data, same normalization format — no need to recompute)
STATS_PATH=${CKPT_DIR}/normalisation_stats_event_c_stage1_cfg.json

mkdir -p ${GRID_DIR}/logs
mkdir -p ${CKPT_DIR}/${RUN_NAME}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Nodes: ${SLURM_NODELIST}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint dir: ${CKPT_DIR}/${RUN_NAME}"
echo "Stats: ${STATS_PATH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0

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
export PYTHONPATH=/global/u2/l/lcondren/pp-collision-diffusion/scripts

srun python3 -u $SCRIPT \
    --grid_dir           ${GRID_DIR} \
    --ckpt_dir           ${CKPT_DIR} \
    --run_name           ${RUN_NAME} \
    --stats_path         ${STATS_PATH} \
    --val_start          80000 \
    --n_train            20000 \
    --n_val              10000 \
    --batch              128 \
    --epoch              200 \
    --lr                 3e-4 \
    --lr_body            1e-4 \
    --num_layers         8 \
    --num_gen_layers     2 \
    --proj_dim           128 \
    --num_jet_mlp        512 \
    --num_part           500 \
    --patience           30 \
    --cfg_drop_x_prob    0.35 \
    --cfg_drop_y_prob    0.35 \
    --time_limit_hours   3.5

echo "Job ${SLURM_JOB_ID} finished: $(date)"
