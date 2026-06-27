#!/bin/bash
#SBATCH --job-name=pet_infer
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/logs/pet_infer_%A_%a.log

export PYTHONUNBUFFERED=1

conda run -n mg5_new python -u \
    /global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/infer_pp.py \
    --rank       "$SLURM_ARRAY_TASK_ID" \
    --world_size 4 \
    --gpu_id     0 \
    --num_steps  500 \
    --chunk_size 200 \
    --n_total    20000 \
    --val_start  400000 \
    --run_name   parton_v1_1node \
    --out_dir    /pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/parton_v1_1node/infer_20k
