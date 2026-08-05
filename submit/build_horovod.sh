#!/bin/bash
#SBATCH --account=daniel_lab
#SBATCH --partition=free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=build_horovod
#SBATCH --output=/data/homezvol0/lcondren/pp-collision-diffusion/logs/%j_build_horovod.out
#SBATCH --error=/data/homezvol0/lcondren/pp-collision-diffusion/logs/%j_build_horovod.err

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pp-diffusion

export MPICC=$CONDA_PREFIX/bin/mpicc
export MPICXX=$CONDA_PREFIX/bin/mpicxx

HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 \
HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 \
pip install horovod==0.28.1 --no-binary horovod --no-cache-dir --no-build-isolation

echo "Exit code: $?"
python -c "import horovod.tensorflow as hvd; hvd.init(); print('rank', hvd.rank(), '/ MPI built:', hvd.mpi_built())"
