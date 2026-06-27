"""Quick smoke test: build model, load 100 events, run one training step."""
import os, sys
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import ctypes as _c
for _lib in ["/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64/libcudart.so.12",
             "/global/common/software/nersc9/cudnn/8.9.3-cuda12/lib/libcudnn.so.8"]:
    try: _c.CDLL(_lib, mode=_c.RTLD_GLOBAL)
    except OSError: pass

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
print(f"GPUs: {[g.name for g in gpus]}", flush=True)
if not gpus: raise RuntimeError("No GPU found — run on a node with free GPU")

from PET_pp import PET_pp
from dataloader_pp import PPDataLoader
from tensorflow.keras.optimizers import Adam

_FULL_DATA = '/pscratch/sd/l/lcondren/MCsim/full_event_fpcd'
STATS      = f'{_FULL_DATA}/normalisation_stats.json'

print("Loading 100 events ...", flush=True)
loader = PPDataLoader(_FULL_DATA, STATS, processes=['dijet','zjets'],
                      batch_size=16, val_start=400000, n_events=50,
                      split='val', num_part=500)

model = PET_pp(num_feat=loader.num_feat, num_jet=loader.num_jet,
               num_cond=loader.num_cond, num_part=500,
               num_layers=4, projection_dim=64,
               local=True, K=5, layer_scale=True, mode='generator')

model.compile(Adam(1e-4), Adam(3e-4))
print("Running 1 epoch (1 batch) ...", flush=True)
tf_data = loader.make_tfdata()
hist = model.fit(tf_data, epochs=1, verbose=1)
print(f"Loss: {hist.history['loss'][0]:.4f}", flush=True)

# Test generate
print("Testing generate() ...", flush=True)
import numpy as np, json
with open(STATS) as f: s = json.load(f)
dummy_cond = np.random.randn(4, 24).astype(np.float32)
parts, jets = model.generate(dummy_cond,
                              jet_mean=s['jet_mean'][0], jet_std=s['jet_std'][0],
                              nsplit=1)
print(f"parts shape: {parts.shape}  jets shape: {jets.shape}", flush=True)
print("SMOKE TEST PASSED", flush=True)
