#!/bin/bash
#SBATCH --job-name=pet_plots_100k
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/logs/pet_plots_100k_%j.log

export PYTHONUNBUFFERED=1

conda run -n mg5_new python -u \
    /global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/plot_infer_100k.py \
    --run_name parton_v1_1node \
    --n_events 100000
