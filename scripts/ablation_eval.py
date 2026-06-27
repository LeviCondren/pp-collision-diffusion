#!/usr/bin/env python3
"""
Ablation evaluation for proc_label_5proc_p3 checkpoint.

Strategy: build the model with the ORIGINAL (unmodified) graph, load the
checkpoint normally, then zero out the Dense layer weights for each conditioning
path AFTER loading.  This keeps the graph topology identical to the checkpoint
so weight loading never fails.

Paths ablated:
  A: proc_token cross-attention token in generator head
     -> zero first pair of Dense layers with proc_label input in model.head
  B: proc_emb additive on cond_token in generator head
     -> zero second pair of Dense layers with proc_label input in model.head
  C: proc_emb in ResNet jet head
     (prior result: val_loss=5.4652, part=4.4843, jet=0.9809 -- included in table)
  D: paths A + C combined

All configs (baseline, A, B, D) are run in a single invocation.
model.evaluate() calls PET_pp_parton_vpar.test_step which uses self.model_part
(live head, not EMA) and self.model_jet (live, not EMA).
"""

import os, sys, json
import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['TF_GPU_ALLOCATOR']     = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2')

import ctypes
for _lib in [
    '/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64/libcudart.so.12',
    '/global/common/software/nersc9/cudnn/8.9.3-cuda12/lib/libcudnn.so.8',
]:
    try: ctypes.CDLL(_lib, ctypes.RTLD_GLOBAL)
    except OSError: pass

import tensorflow as tf
import h5py

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

import argparse

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='/pscratch/sd/l/lcondren/MCsim/full_event_mixed')
    p.add_argument('--run_name',   default='proc_label_5proc_p3')
    p.add_argument('--val_start',  type=int, default=400000)
    p.add_argument('--n_val',      type=int, default=2000,
                   help='Val events per process (5 procs -> 10k total)')
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()

args = _parse()

SCRIPTS   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

# proc_label_train.py calls hvd.init() at import time; mock horovod for
# single-process evaluation so no horovodrun/srun is needed.
import types as _types
_hvd = _types.ModuleType('horovod.tensorflow.keras')
_hvd.init         = lambda: None
_hvd.rank         = lambda: 0
_hvd.local_rank   = lambda: 0
_hvd.size         = lambda: 1
_hvd_tf           = _types.ModuleType('horovod.tensorflow')
_hvd_root         = _types.ModuleType('horovod')
sys.modules['horovod']                  = _hvd_root
sys.modules['horovod.tensorflow']       = _hvd_tf
sys.modules['horovod.tensorflow.keras'] = _hvd

from proc_label_train import ProcLabelPET
from tensorflow.keras.optimizers import Adam

PROCESSES   = ['dijet', 'zjets', 'ttbar', 'wjets', 'wprime']
N_PROC      = 5
MAX_PARTONS = 6
PARTON_FEAT = 6
NUM_PART    = 500
CKPT_PATH   = f'{args.data_dir}/checkpoints/{args.run_name}/pet_pp.weights.h5'
STATS_PATH  = f'{args.data_dir}/normalisation_stats.json'

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f'Checkpoint not found: {CKPT_PATH}')

print(f'Checkpoint: {CKPT_PATH}', flush=True)

# ── Load validation data ──────────────────────────────────────────────────────
with open(STATS_PATH) as fh:
    stats = json.load(fh)

part_mean = np.array(stats['part_mean'], dtype=np.float32)
part_std  = np.array(stats['part_std'],  dtype=np.float32)
jet_mean  = float(stats['jet_mean'][0])
jet_std   = float(stats['jet_std'][0])

n_cond_feat   = MAX_PARTONS * PARTON_FEAT
cond_mean_raw = np.array(stats['cond_mean'], dtype=np.float32)
cond_std_raw  = np.array(stats['cond_std'],  dtype=np.float32)
cond_mean     = np.zeros(n_cond_feat, dtype=np.float32)
cond_std      = np.ones(n_cond_feat,  dtype=np.float32)
n_fill        = min(len(cond_mean_raw), n_cond_feat)
cond_mean[:n_fill] = cond_mean_raw[:n_fill]
cond_std[:n_fill]  = np.where(cond_std_raw[:n_fill] > 0, cond_std_raw[:n_fill], 1.0)

all_pf, all_mask, all_cond, all_jet = [], [], [], []

for proc_idx, proc in enumerate(PROCESSES):
    path = f'{args.data_dir}/{proc}.hdf5'
    with h5py.File(path, 'r') as f:
        n_total = f['particle_features'].shape[0]
        r0 = min(args.val_start, n_total) if proc != 'wprime' else 0
        r1 = min(r0 + args.n_val, n_total)
        if r0 >= r1:
            print(f'  {proc}: no events in [{r0},{r1}), skipping', flush=True)
            continue
        pf_raw   = f['particle_features'][r0:r1].astype(np.float32)
        part_raw = f['parton_features'][r0:r1].astype(np.float32)
        if 'n_partons' in f:
            n_par       = f['n_partons'][r0:r1].astype(np.int32)
            parton_mask = (np.arange(MAX_PARTONS)[None, :] <
                           n_par[:, None]).astype(np.float32)
        else:
            raw = (np.linalg.norm(part_raw[:, :MAX_PARTONS], axis=2) > 1e-6
                   ).astype(np.float32)
            if raw.shape[1] < MAX_PARTONS:
                pad = np.zeros((len(raw), MAX_PARTONS - raw.shape[1]), np.float32)
                raw = np.concatenate([raw, pad], axis=1)
            parton_mask = raw[:, :MAX_PARTONS]

    if part_raw.shape[1] < MAX_PARTONS:
        pad      = np.zeros((len(part_raw), MAX_PARTONS - part_raw.shape[1],
                             PARTON_FEAT), np.float32)
        part_raw = np.concatenate([part_raw, pad], axis=1)
    part_raw = part_raw[:, :MAX_PARTONS, :]

    mask = pf_raw[:, :NUM_PART, 6].astype(np.float32)
    pf6  = pf_raw[:, :NUM_PART, :6]
    npart = mask.sum(axis=1, keepdims=True).astype(np.float32)
    jet   = (np.log(np.maximum(npart, 1.0)) - jet_mean) / jet_std

    cond_kin  = part_raw.reshape(len(part_raw), n_cond_feat)
    cond_norm = (cond_kin - cond_mean) / cond_std
    proc_oh   = np.zeros((len(cond_norm), N_PROC), dtype=np.float32)
    proc_oh[:, proc_idx] = 1.0
    cond = np.concatenate([cond_norm, parton_mask, proc_oh], axis=1)

    pf6_norm = (pf6 - part_mean) / part_std * mask[:, :, None]
    all_pf.append(pf6_norm)
    all_mask.append(mask)
    all_cond.append(cond)
    all_jet.append(jet)
    print(f'  {proc}: {len(pf6_norm)} events  rows [{r0},{r1})', flush=True)

pf_all   = np.concatenate(all_pf)
mask_all = np.concatenate(all_mask)
cond_all = np.concatenate(all_cond)
jet_all  = np.concatenate(all_jet)

rng = np.random.default_rng(args.seed)
idx = rng.permutation(len(pf_all))
pf_all   = pf_all[idx];  mask_all = mask_all[idx]
cond_all = cond_all[idx]; jet_all  = jet_all[idx]
print(f'Total val events: {len(pf_all):,}', flush=True)

val_ds = (tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices({
        'input_features': pf_all,
        'input_mask':     mask_all,
        'input_jet':      jet_all,
    }),
    tf.data.Dataset.from_tensor_slices(cond_all)))
    .batch(args.batch_size)
    .prefetch(tf.data.AUTOTUNE))

# ── Build model (once; reused for all ablations via load_weights) ─────────────
tf.random.set_seed(args.seed)

model = ProcLabelPET(
    num_proc_labels=N_PROC,
    num_feat=6, num_jet=1,
    max_partons=MAX_PARTONS, parton_feat=PARTON_FEAT,
    num_part=NUM_PART,
    projection_dim=128,
    local=True, K=5,
    num_layers=8,
    num_gen_layers=2,
    drop_probability=0.0,
    simple=False, layer_scale=True, talking_head=False,
    mode='generator',
)
model.compile(body_optimizer=Adam(1e-4), head_optimizer=Adam(3e-4))

# Load checkpoint once to confirm it works and for diagnostic
print('\nLoading checkpoint...', flush=True)
ref_before = model.weights[10].numpy().copy()
model.load_weights(CKPT_PATH)
ref_after  = model.weights[10].numpy()
if np.allclose(ref_before, ref_after):
    raise RuntimeError('weights[10] unchanged after load_weights -- checkpoint not loaded')
print(f'Checkpoint loaded OK: {len(model.weights)} weights', flush=True)

# ── Identify proc-label Dense layers by tensor connectivity ──────────────────

def _find_proc_label_pairs(submodel, n_proc=5):
    """
    Find Dense layer pairs where:
      d1: 2-D input with last dim == n_proc  (takes proc_label directly)
      d2: takes d1's output tensor as its input  (second Dense of the pair)
    Returns list of (d1, d2) in the order d1 appears in submodel.layers.
    d2 is None if no consumer Dense is found.
    """
    d1_list = []
    for layer in submodel.layers:
        if not isinstance(layer, tf.keras.layers.Dense):
            continue
        try:
            in_shape = layer.input_shape
        except (AttributeError, RuntimeError):
            continue
        if isinstance(in_shape, (list, tuple)) and len(in_shape) == 2 and in_shape[-1] == n_proc:
            d1_list.append(layer)

    pairs = []
    for d1 in d1_list:
        d2 = None
        try:
            d1_out = d1.output
        except (AttributeError, RuntimeError):
            pairs.append((d1, None))
            continue
        for candidate in submodel.layers:
            if candidate is d1 or not isinstance(candidate, tf.keras.layers.Dense):
                continue
            try:
                if candidate.input is d1_out:
                    d2 = candidate
                    break
            except (AttributeError, RuntimeError):
                continue
        pairs.append((d1, d2))
    return pairs

print('\n=== proc-label Dense layers in model.head ===', flush=True)
pairs_head = _find_proc_label_pairs(model.head, n_proc=N_PROC)
for i, (d1, d2) in enumerate(pairs_head):
    d2_str = (f'{d2.name} in={d2.input_shape} out={d2.output_shape}' if d2
              else 'NOT FOUND')
    print(f'  pair {i}: {d1.name} in={d1.input_shape} out={d1.output_shape}'
          f'  |  d2: {d2_str}', flush=True)

print('\n=== proc-label Dense layers in model.model_jet ===', flush=True)
pairs_jet = _find_proc_label_pairs(model.model_jet, n_proc=N_PROC)
for i, (d1, d2) in enumerate(pairs_jet):
    print(f'  pair {i}: {d1.name} in={d1.input_shape} out={d1.output_shape}',
          flush=True)

# Safety checks
if len(pairs_head) != 2:
    print(f'\nERROR: expected 2 proc-label Dense pairs in model.head, '
          f'got {len(pairs_head)}.  Full Dense list:', flush=True)
    for layer in model.head.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            try:
                print(f'  {layer.name}  in={layer.input_shape}  out={layer.output_shape}',
                      flush=True)
            except Exception:
                print(f'  {layer.name}  (shapes unavailable)', flush=True)
    raise SystemExit(1)

if len(pairs_jet) < 1:
    print(f'\nERROR: expected >=1 proc-label Dense pair in model.model_jet, '
          f'got {len(pairs_jet)}.', flush=True)
    raise SystemExit(1)

for i, (d1, d2) in enumerate(pairs_head):
    if d2 is None:
        print(f'\nERROR: Dense-2 not found for head pair {i} ({d1.name}). '
              'Cannot ablate safely.', flush=True)
        raise SystemExit(1)

# Assign paths: pair 0 = A (proc_token xattn), pair 1 = B (proc_emb additive)
PATH_A = [(model.head, pairs_head[0][0]), (model.head, pairs_head[0][1])]
PATH_B = [(model.head, pairs_head[1][0]), (model.head, pairs_head[1][1])]
PATH_C = [(model.model_jet, pairs_jet[0][0])]

CONFIGS = {
    'baseline': [],
    'A':        PATH_A,
    'B':        PATH_B,
    'D':        PATH_A + PATH_C,
}

print('\nAblation layer assignment:')
print(f'  Path A (xattn token): {[l.name for _,l in PATH_A]}')
print(f'  Path B (cond_token additive): {[l.name for _,l in PATH_B]}')
print(f'  Path C (resnet): {[l.name for _,l in PATH_C]}')

# ── Evaluation helper ─────────────────────────────────────────────────────────

def _run(config_name, layers_to_zero):
    print(f'\n--- ablation={config_name} ---', flush=True)
    model.load_weights(CKPT_PATH)

    for submodel, layer in layers_to_zero:
        old_w  = layer.get_weights()
        layer.set_weights([np.zeros_like(w) for w in old_w])
        shapes = [w.shape for w in old_w]
        print(f'  zeroed {layer.name} ({shapes})', flush=True)

    for m in model.metrics:
        m.reset_states()

    metrics  = model.evaluate(val_ds, return_dict=True, verbose=0)
    val_loss = float(metrics['loss'])
    part_l   = float(metrics.get('part', float('nan')))
    jet_l    = float(metrics.get('jet',  float('nan')))
    print(f'RESULT ablation={config_name}  val_loss={val_loss:.4f}  '
          f'part={part_l:.4f}  jet={jet_l:.4f}', flush=True)
    return val_loss, part_l, jet_l

# ── Run all ablations ─────────────────────────────────────────────────────────
results = {}
for name, layers in CONFIGS.items():
    results[name] = _run(name, layers)

# ── Final table ───────────────────────────────────────────────────────────────
print('\n' + '=' * 62, flush=True)
print(f'{"Config":<10}  {"val_loss":>10}  {"part":>10}  {"jet":>10}',
      flush=True)
print('-' * 62, flush=True)
for name in ['baseline', 'A', 'B', 'C', 'D']:
    if name == 'C':
        print(f'{"C":<10}  {"5.4652":>10}  {"4.4843":>10}  {"0.9809":>10}'
              '  (prior run)', flush=True)
    else:
        v, p, j = results[name]
        print(f'{name:<10}  {v:>10.4f}  {p:>10.4f}  {j:>10.4f}', flush=True)
print('=' * 62, flush=True)
