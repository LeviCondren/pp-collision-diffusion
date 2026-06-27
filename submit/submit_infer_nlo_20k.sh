#!/bin/bash
#SBATCH --job-name=infer_nlo_20k
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs/%j_infer_nlo_20k.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/logs/%j_infer_nlo_20k.err

WORLD_SIZE=4
NUM_STEPS=500
CHUNK_SIZE=200
N_TOTAL=20000
VAL_START=400000
RUN_NAME=parton_v1_nlo
SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/infer_pp.py
NLO_DATA=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd_nlo
LO_DATA=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd
OUT_DIR=${LO_DATA}/checkpoints_pet_pp/${RUN_NAME}/infer_nlo_20k

mkdir -p ${LO_DATA}/checkpoints_pet_pp/logs
mkdir -p ${OUT_DIR}

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PIDS=()
LOGS=()
for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
    LOG="${OUT_DIR}/rank${RANK}.log"
    LOGS+=("$LOG")
    echo "  Launching rank $RANK on GPU $RANK  ->  $LOG"
    conda run -n mg5_new python "$SCRIPT" \
        --rank       "$RANK" \
        --world_size "$WORLD_SIZE" \
        --gpu_id     "$RANK" \
        --num_steps  "$NUM_STEPS" \
        --chunk_size "$CHUNK_SIZE" \
        --n_total    "$N_TOTAL" \
        --val_start  "$VAL_START" \
        --run_name   "$RUN_NAME" \
        --data_dir   "$NLO_DATA" \
        --stats_dir  "$LO_DATA" \
        --ckpt_dir   "${LO_DATA}/checkpoints_pet_pp" \
        --out_dir    "$OUT_DIR" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "All $WORLD_SIZE ranks launched. PIDs: ${PIDS[*]}"
tail -f "${LOGS[0]}" &
TAIL_PID=$!

FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    EXIT=$?
    if [ "$EXIT" -ne 0 ]; then
        echo "ERROR: rank $i exited with code $EXIT"
        tail -30 "${LOGS[$i]}"
        FAILED=1
    fi
done

kill "$TAIL_PID" 2>/dev/null

if [ "$FAILED" -eq 0 ]; then
    echo "All ranks finished. Concatenating..."
    conda run -n mg5_new python \
        /global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/concat_infer.py \
        --world_size "$WORLD_SIZE" \
        --run_name   "$RUN_NAME" \
        --out_dir    "$OUT_DIR"
    echo "Job ${SLURM_JOB_ID} finished: $(date)"
else
    echo "One or more ranks failed — check logs in $OUT_DIR"
    exit 1
fi
