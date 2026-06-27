#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=theoryspec_parton
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/logs/%j_theoryspec_parton.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/logs/%j_theoryspec_parton.err

SCRIPT_DIR=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts
SM_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed
BSM_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_signal
CKPT_BASE=${SM_DIR}/checkpoints
RUN_NAME=theoryspec_parton_gen

mkdir -p ${CKPT_BASE}/logs

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=${SCRIPT_DIR}

# ── Check if training is already done ────────────────────────────────────────
STATE=${CKPT_BASE}/${RUN_NAME}/training_state.json
if [ -f "$STATE" ]; then
    DONE=$(python3 -c "import json; s=json.load(open('$STATE')); print(s.get('done',False))")
    if [ "$DONE" = "True" ]; then
        echo "Training already complete (training_state.json says done=True). Exiting."
        exit 0
    fi
    EP=$(python3 -c "import json; s=json.load(open('$STATE')); print(s.get('epochs_done',0))")
    echo "Resuming from epoch ${EP}"
fi

horovodrun --gloo -np 2 python3 -u ${SCRIPT_DIR}/theoryspec_parton_gen.py \
    --sm_dir            ${SM_DIR} \
    --sm_processes      dijet zjets ttbar wjets \
    --bsm_dir           ${BSM_DIR} \
    --bsm_type          wprime \
    --theory_dim        2 \
    --theory_ref        600.0 600.0 \
    --run_name          ${RUN_NAME} \
    --ckpt_base         ${CKPT_BASE} \
    --batch             512 \
    --epoch             300 \
    --lr                3e-4 \
    --context_dim       256 \
    --hidden            256 \
    --num_layers        6 \
    --t_emb_dim         128 \
    --patience          30 \
    --val_start         400000 \
    --n_val_per_proc    5000 \
    --time_limit_hours  3.5

EXIT_CODE=$?
echo "Training exit code: ${EXIT_CODE}"

# ── Self-resubmit if not done ─────────────────────────────────────────────────
if [ -f "$STATE" ]; then
    DONE=$(python3 -c "import json; s=json.load(open('$STATE')); print(s.get('done',False))")
    if [ "$DONE" != "True" ]; then
        echo "Resubmitting..."
        sbatch ${BASH_SOURCE[0]}
    else
        echo "Training complete — not resubmitting."
    fi
fi

echo "Job ${SLURM_JOB_ID} finished: $(date)"
