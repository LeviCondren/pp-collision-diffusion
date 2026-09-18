#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --job-name=e039_smoke
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_e039_smoke.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/logs/%j_e039_smoke.err

# Smoke test for E039 (joint mass_x/mass_y + cone_mass CFG dropout), ported from
# the sandbox to the canonical repo. Mirrors the E038 smoke-test procedure
# (EXPERIMENTS.md E038 detail section): throwaway run_name, tiny n_train/n_val,
# 1 epoch, cfg_drop_x_prob=cfg_drop_y_prob=0.35 (E038's production value).
# Deleted after inspection — this is NOT the production training job.

SCRIPT=/global/u2/l/lcondren/pp-collision-diffusion/scripts/bsm_grid_train_event_c_stage1_cfg_asym_e039.py
GRID_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
RUN_NAME=smoke_e039_cfg_joint
STATS_PATH=${CKPT_DIR}/normalisation_stats_event_c_stage1_cfg.json

mkdir -p ${GRID_DIR}/logs
rm -rf ${CKPT_DIR}/${RUN_NAME}
mkdir -p ${CKPT_DIR}/${RUN_NAME}

echo "Job ${SLURM_JOB_ID} started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=/global/u2/l/lcondren/pp-collision-diffusion/scripts

python3 -u $SCRIPT \
    --grid_dir           ${GRID_DIR} \
    --ckpt_dir            ${CKPT_DIR} \
    --run_name            ${RUN_NAME} \
    --stats_path          ${STATS_PATH} \
    --val_start           80000 \
    --n_train             64 \
    --n_val               32 \
    --batch               32 \
    --epoch               1 \
    --lr                  3e-4 \
    --lr_body             1e-4 \
    --num_layers          8 \
    --num_gen_layers      2 \
    --proj_dim            128 \
    --num_jet_mlp         512 \
    --num_part            500 \
    --patience            30 \
    --cfg_drop_x_prob     0.35 \
    --cfg_drop_y_prob     0.35
TRAIN_EXIT=$?

echo ""
echo "=== Smoke test checks ==="
PASS=0; FAIL=0
check() {
    if [ "$2" -eq 1 ]; then echo "  PASS: $1"; PASS=$((PASS+1));
    else echo "  FAIL: $1"; FAIL=$((FAIL+1)); fi
}
check "train script exit code == 0" "$([ $TRAIN_EXIT -eq 0 ] && echo 1 || echo 0)"

STATE_FILE=${CKPT_DIR}/${RUN_NAME}/training_state.json
check "training_state.json exists" "$([ -f $STATE_FILE ] && echo 1 || echo 0)"

python3 -c "
import json, sys
try:
    s = json.load(open('$STATE_FILE'))
except Exception as e:
    print(f'  FAIL: cannot load training_state.json: {e}'); sys.exit(1)
print(f'  epochs_done={s.get(\"epochs_done\")}  done={s.get(\"done\")}  val_loss={s.get(\"val_loss\")}')
vl = s.get('val_loss')
if vl is None or vl != vl or vl in (float('inf'), float('-inf')):
    print('  FAIL: val_loss is missing/NaN/Inf'); sys.exit(1)
if not s.get('done', False):
    print('  FAIL: done is not True'); sys.exit(1)
print('  PASS: training_state.json sane (done, finite val_loss)')
"
check "training_state.json sane" "$([ $? -eq 0 ] && echo 1 || echo 0)"

echo ""
echo "=== Smoke test summary: ${PASS} passed, ${FAIL} failed ==="
if [ "$FAIL" -gt 0 ]; then
    echo "SMOKE TEST FAILED"
    exit 1
else
    echo "SMOKE TEST PASSED — safe to submit production job"
fi

echo "Job ${SLURM_JOB_ID} finished: $(date)"
