#!/bin/bash
# install_env.sh — set up the Python environment for pp-collision-diffusion training.
#
# Run once on the new cluster before submitting any jobs:
#   bash setup/install_env.sh
#
# After this script completes, activate the env with:
#   conda activate pp-diffusion
# or source it in your submit script (see submit/*_portable.sh).
#
# Requirements: conda or mamba available in PATH.
# The Horovod install is attempted but not required; training falls back to
# single-GPU if Horovod is unavailable (the training scripts detect this).

set -euo pipefail

ENV_NAME="pp-diffusion"
PYTHON_VERSION="3.10"

echo "=== Creating conda environment: ${ENV_NAME} ==="
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
conda activate "${ENV_NAME}" || source activate "${ENV_NAME}"

echo "=== Installing core dependencies ==="
pip install --upgrade pip
pip install tensorflow==2.15.0
pip install numpy h5py

echo "=== Installing Pythia8 Python bindings ==="
pip install pythia8

echo "=== Attempting Horovod install (requires MPI; failure is non-fatal) ==="
# Load MPI if available via module system — adjust for your cluster.
# Examples:
#   module load openmpi   (many clusters)
#   module load mpi/openmpi-4.1.1
# If MPI headers are in a non-standard location, set:
#   export HOROVOD_MPI_HOME=/path/to/mpi
if python -c "import mpi4py" 2>/dev/null || command -v mpicc >/dev/null 2>&1; then
    echo "  MPI detected — building Horovod with TensorFlow backend."
    HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 pip install horovod[tensorflow] \
        && echo "  Horovod installed successfully." \
        || echo "  Horovod build failed — training will use single-GPU fallback."
else
    echo "  No MPI found — skipping Horovod. Training will use single-GPU fallback."
    echo "  To enable multi-GPU later: load your cluster MPI module, then:"
    echo "    HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 pip install horovod[tensorflow]"
fi

echo ""
echo "=== Environment setup complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo ""
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import pythia8; print('Pythia8 OK')"
python -c "import horovod; print(f'Horovod {horovod.__version__}')" 2>/dev/null \
    || echo "Horovod not installed — single-GPU mode."
