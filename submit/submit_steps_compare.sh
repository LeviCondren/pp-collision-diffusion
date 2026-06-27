#!/bin/bash
#SBATCH --job-name=pet_steps_compare
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/logs/pet_steps_compare_%j.log

export PYTHONUNBUFFERED=1
SCRIPT=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/infer_pp.py
OUT_BASE=/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/parton_v1_1node

echo "=== 50-step run ==="
conda run -n mg5_new python -u "$SCRIPT" \
    --rank 0 --world_size 1 --gpu_id 0 \
    --n_total 1000 --num_steps 50 --chunk_size 50 \
    --out_dir "${OUT_BASE}/infer_steps_compare_50"

echo "=== 500-step run ==="
conda run -n mg5_new python -u "$SCRIPT" \
    --rank 0 --world_size 1 --gpu_id 0 \
    --n_total 1000 --num_steps 500 --chunk_size 50 \
    --out_dir "${OUT_BASE}/infer_steps_compare"
