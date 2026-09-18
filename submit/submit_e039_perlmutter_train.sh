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
#SBATCH --job-name=e039_cfg_asym_joint
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_e039_cfg_asym_joint.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_e039_cfg_asym_joint.err

# E039 — Joint mass-hypothesis + cone-mass CFG dropout (bsm_grid_event_c_stage1_cfg_asym_e039).
# Same structure as submit_e038_perlmutter_train.sh (4 GPUs via srun, self-resubmitting to
# 200 epochs, E034 stats reused), but trains the new joint-dropout class
# (PET_pp_parton_vpar_bsm_event_c_stage1_cfg_asym_e039.py / WeightedBSMPET_event_c_cfg_asym):
# whenever cone_mass_X/Y is dropped, mass_x/mass_y (indices 20/27 of the 32-dim parton
# conditioning vector y) is now ALSO zeroed by the SAME drop_X/drop_Y draw — closing the
# leakage path where the model could infer a "dropped" cone mass from the ever-present mass
# hypothesis. cfg_drop_x_prob=cfg_drop_y_prob=0.35 (E038's production value, carried over
# since the joint-drop null branch is if anything harder to learn, not easier — see
# EXPERIMENTS.md E039 entry, obstacle 1).
#
# Smoke-tested 2026-09-17 (job 58472497, qos=debug, 1 epoch, n_train=9024, cfg_drop 0.35/0.35):
# exit 0, training_state.json done=true, val_loss=6.593 (finite, comparable to E038's 6.524
# 1-epoch smoke value), dropout telemetry drop_X=0.349 drop_Y=0.343 drop_both=0.119
# drop_none=0.427 (matches 0.35/0.35/0.1225/0.4225 targets within batch noise), no NaN/Inf.
# Standalone masking-logic check also passed (index 20/27 zeroed iff drop_X/drop_Y, all
# other features untouched, all 4 drop-combinations verified).
#
# Checkpoint dir separate from E038 (bsm_grid_event_c_stage1_cfg_asym_e038 untouched).
# Inference/sampling code is unchanged from E038 (generate_composable_cfg/DDPMSamplerComposableCFG
# live in the architecture file and are not touched by this change) — once trained, existing
# infer_composable.py works against this checkpoint by pointing --run_name at it.

SCRIPT=/global/u2/l/lcondren/pp-collision-diffusion/scripts/bsm_grid_train_event_c_stage1_cfg_asym_e039.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
RUN_NAME=bsm_grid_event_c_stage1_cfg_asym_e039
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
