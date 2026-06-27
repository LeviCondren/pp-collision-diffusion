#!/bin/bash
#SBATCH --job-name=pet_infer_100k
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=192G
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --output=/tmp/pet_infer_100k_%j.log

cd /global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp
bash launch_infer_4gpu.sh 50 200
