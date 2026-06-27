#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=infer_p3
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/infer_20k/logs/%j_infer.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/infer_20k/logs/%j_infer.err

SCRIPTS=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts
DATA_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed
WPRIME_DIR=/pscratch/sd/l/lcondren/MCsim/wprime_inference
RUN_NAME=proc_label_5proc_p3
OUT_DIR=${DATA_DIR}/checkpoints/${RUN_NAME}/infer_20k

mkdir -p ${OUT_DIR}/logs

echo "Job ${SLURM_JOB_ID} started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load tensorflow/2.15.0
export PYTHONPATH=${SCRIPTS}

echo "=== Step 1: Generate samples (20k events x 5 processes) ==="
srun python3 -u ${SCRIPTS}/infer_pp_proc_label.py \
    --rank 0 --world_size 1 \
    --data_dir    ${DATA_DIR} \
    --wprime_dir  ${WPRIME_DIR} \
    --run_name    ${RUN_NAME} \
    --val_start   400000 \
    --n_total     20000 \
    --num_steps   50 \
    --chunk_size  200 \
    --npart           500 \
    --proj_dim        128 \
    --num_layers      8 \
    --num_gen_layers  2 \
    --processes       dijet zjets ttbar wjets wprime \
    --out_dir         ${OUT_DIR}

echo "=== Step 2: Concatenate per-rank npz files ==="
python3 -u ${SCRIPTS}/concat_infer_5proc.py \
    --run_name   ${RUN_NAME} \
    --world_size 1 \
    --data_dir   ${DATA_DIR} \
    --out_dir    ${OUT_DIR}

echo "=== Step 3: Generate all plots ==="
python3 -u ${SCRIPTS}/plot_infer_5proc.py \
    --run_name  ${RUN_NAME} \
    --n_events  20000 \
    --data_dir  ${DATA_DIR} \
    --infer_dir ${OUT_DIR} \
    --out_dir   ${DATA_DIR}/checkpoints/${RUN_NAME}/plots_20k

echo "Job ${SLURM_JOB_ID} finished: $(date)"
