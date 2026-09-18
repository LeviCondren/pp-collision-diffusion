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
#SBATCH --job-name=e036_cfg_asym
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_e036_cfg_asym.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_e036_cfg_asym.err

# E036 — Asymmetric per-cone CFG dropout training (bsm_grid_event_c_stage1_cfg_asym).
# Architecture identical to E034. Training change: independent Bernoulli dropout per cone:
#   cone_mass_X (index 4 of 7-dim event vector) dropped with prob 0.10
#   cone_mass_Y (index 6 of 7-dim event vector) dropped with prob 0.10
# Trains all four conditioning combinations: (X,Y), (noX,Y), (X,noY), (noX,noY).
# Inference: four-pass per-cone CFG via generate_asym_cfg / DDPMSamplerAsymCFG.
# Stats reused from E034 (identical data, identical normalization format).
# 4 GPUs via srun. Self-resubmitting until training_state.json marks done=true.

SCRIPT=/global/u2/l/lcondren/pp-collision-diffusion/scripts/bsm_grid_train_event_c_stage1_cfg_asym.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
RUN_NAME=bsm_grid_event_c_stage1_cfg_asym
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
    --cfg_drop_x_prob    0.10 \
    --cfg_drop_y_prob    0.10 \
    --time_limit_hours   3.5

echo "Job ${SLURM_JOB_ID} finished: $(date)"
