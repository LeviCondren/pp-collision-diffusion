#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=e033_smoke
#SBATCH --output=/pub/lcondren/wprime_signal_mpi/logs/%j_e033_smoke.out
#SBATCH --error=/pub/lcondren/wprime_signal_mpi/logs/%j_e033_smoke.err

# Smoke test for checkpoint/resume logic in infer_bsm_grid_event_c_locality.py.
#
# Phase 1: normal two-chunk completion.
# Phase 2: reconstruct a valid partial from Phase 1 (reverse denormalization),
#          run again, verify the first-50 events are reproduced and shape is correct.
#
# SLURM_JOB_END_TIME is unset for Python invocations so the time-budget check
# stays inactive (a short job limit would otherwise trigger it before chunk 0).

SCRIPT=/data/homezvol0/lcondren/pp-collision-diffusion/scripts/infer_bsm_grid_event_c_locality.py
GRID_DIR=/pub/lcondren/wprime_signal_mpi
CKPT_DIR=${GRID_DIR}/checkpoints_bsm_grid
RUN_NAME=bsm_grid_event_c_locality
SMOKE_DIR=${CKPT_DIR}/${RUN_NAME}/smoke_test
STATS=${CKPT_DIR}/normalisation_stats_event_c_stage1.json

mkdir -p ${GRID_DIR}/logs
mkdir -p ${SMOKE_DIR}
rm -f ${SMOKE_DIR}/*.npz ${SMOKE_DIR}/.resubmit.lock

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: ${SLURM_NODELIST}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pp-diffusion
NVIDIA_LIBS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/lib')))))")
NVIDIA_BINS=$(python3 -c "import glob,os; print(':'.join(sorted(glob.glob(os.path.join(os.environ['CONDA_PREFIX'],'lib/python3.10/site-packages/nvidia/*/bin')))))")
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}
export PATH=$NVIDIA_BINS:$PATH
export PYTHONPATH=/data/homezvol0/lcondren/pp-collision-diffusion/scripts

# Unset SLURM_JOB_END_TIME so the script's time-budget check stays inactive.
PYTHON="env -u SLURM_JOB_END_TIME $CONDA_PREFIX/bin/python3"

COMMON="
    --grid_dir       ${GRID_DIR}
    --ckpt_dir       ${CKPT_DIR}
    --run_name       ${RUN_NAME}
    --out_dir        ${SMOKE_DIR}
    --m_X 250 --m_Y 250
    --gpu_id 0
    --n_total        100
    --chunk_size     50
    --num_steps      10
    --num_gen_layers 2
    --num_jet_mlp    512
"

OUT=${SMOKE_DIR}/bsm_mX0250_mY0250_rank00_of01.npz
PARTIAL=${SMOKE_DIR}/bsm_mX0250_mY0250_rank00_of01_partial.npz
PASS=0; FAIL=0

check() {
    local label="$1"; local cond="$2"
    if [ "$cond" -eq 1 ]; then
        echo "  PASS: $label"; PASS=$((PASS+1))
    else
        echo "  FAIL: $label"; FAIL=$((FAIL+1))
    fi
}

# ── Phase 1: normal two-chunk completion ─────────────────────────────────────
echo ""
echo "=== Phase 1: normal completion (100 events, 2 chunks, 10 steps) ==="
$PYTHON -u $SCRIPT $COMMON
P1_EXIT=$?
check "exit code == 0"     "$([ $P1_EXIT -eq 0 ] && echo 1 || echo 0)"
check "output NPZ exists"  "$([ -f $OUT ] && echo 1 || echo 0)"
check "partial cleaned up" "$([ ! -f $PARTIAL ] && echo 1 || echo 0)"

$PYTHON << PYEOF
import numpy as np, sys
try:
    d = np.load('$OUT')
except Exception as e:
    print(f'  FAIL: cannot load output: {e}'); sys.exit(1)
expected = {'parts_truth','parts_gen','mask','mask_gen','parton_feat',
            'mass_x','mass_y','event_feat_truth','jets_gen'}
missing = expected - set(d.files)
if missing:
    print(f'  FAIL: missing keys {missing}'); sys.exit(1)
print(f'  PASS: all keys present')
jets = d['jets_gen']; parts = d['parts_gen']
assert jets.shape  == (100, 8), f'jets shape {jets.shape} != (100,8)'
assert parts.shape[0] == 100,   f'parts n_events {parts.shape[0]} != 100'
print(f'  PASS: jets_gen {jets.shape}  parts_gen {parts.shape}')
PYEOF
check "NPZ keys and shapes" "$([ $? -eq 0 ] && echo 1 || echo 0)"

if [ ! -f "$OUT" ]; then
    echo "  (Phase 1 output missing — skipping Phase 2)"
    FAIL=$((FAIL+1))
else

# ── Phase 2: resume from a reconstructed partial ─────────────────────────────
# _save_partial() stores raw (pre-denorm) model output as 'parts_done'.
# Reconstruct from Phase 1's final NPZ by inverting denormalization:
#   parts_raw = (parts_phys - mean) / std  for valid particles, 0 for padding.
# On resume, re-denormalizing recovers the same parts_phys to float32 precision.
echo ""
echo "=== Phase 2: resume from reconstructed partial (1/2 chunks done) ==="

$PYTHON << PYEOF
import numpy as np, json, sys

d = np.load('$OUT')
jets_chunk0       = d['jets_gen'][:50]
parts_phys_chunk0 = d['parts_gen'][:50]

with open('$STATS') as f:
    stats = json.load(f)
jet_mean  = float(np.array(stats['jet_mean'])[0])
jet_std   = float(np.array(stats['jet_std'])[0])
part_mean = np.array(stats['part_mean'], dtype=np.float32)
part_std  = np.array(stats['part_std'],  dtype=np.float32)

log_npart = jets_chunk0[:, 0] * jet_std + jet_mean
npart     = np.clip(np.round(np.exp(log_npart)).astype(int), 1, 500)
valid     = (np.arange(500)[None, :] < npart[:, None])[:, :, None]

parts_raw = np.where(
    valid,
    (parts_phys_chunk0 - part_mean) / part_std,
    np.float32(0.0)
).astype(np.float32)

np.savez_compressed('$PARTIAL',
    jets_done     = jets_chunk0,
    parts_done    = parts_raw,
    n_chunks_done = np.int32(1),
)
print(f'  Wrote partial: jets_done {jets_chunk0.shape}  parts_done {parts_raw.shape}  n_chunks_done=1')
PYEOF
check "partial construction" "$([ $? -eq 0 ] && echo 1 || echo 0)"

mv ${OUT} ${OUT}.phase1

$PYTHON -u $SCRIPT $COMMON
P2_EXIT=$?
check "resume exit code == 0"        "$([ $P2_EXIT -eq 0 ] && echo 1 || echo 0)"
check "resume output NPZ exists"     "$([ -f $OUT ] && echo 1 || echo 0)"
check "partial cleaned up on finish" "$([ ! -f $PARTIAL ] && echo 1 || echo 0)"

$PYTHON << PYEOF
import numpy as np, sys
try:
    d1 = np.load('${OUT}.phase1')
    d2 = np.load('$OUT')
except Exception as e:
    print(f'  FAIL: {e}'); sys.exit(1)

jd   = np.abs(d1['jets_gen'][:50] - d2['jets_gen'][:50]).max()
ok_j = jd < 1e-6
print(f'  {"PASS" if ok_j else "FAIL"}: jets_gen[:50] identical to phase-1  (maxdiff={jd:.2e})')

pd   = np.abs(d1['parts_gen'][:50] - d2['parts_gen'][:50]).max()
ok_p = pd < 1e-4
print(f'  {"PASS" if ok_p else "FAIL"}: parts_gen[:50] match phase-1        (maxdiff={pd:.2e})')

ok_s = d2['jets_gen'].shape == (100, 8) and d2['parts_gen'].shape[0] == 100
print(f'  {"PASS" if ok_s else "FAIL"}: output shape (100,...)  jets {d2["jets_gen"].shape}  parts {d2["parts_gen"].shape}')

if not (ok_j and ok_p and ok_s): sys.exit(1)
PYEOF
check "resume data integrity" "$([ $? -eq 0 ] && echo 1 || echo 0)"

fi  # end: phase 1 output exists guard

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test summary: ${PASS} passed, ${FAIL} failed ==="
rm -f ${OUT}.phase1

if [ "$FAIL" -gt 0 ]; then
    echo "SMOKE TEST FAILED"
    exit 1
else
    echo "SMOKE TEST PASSED — safe to submit production job"
fi
