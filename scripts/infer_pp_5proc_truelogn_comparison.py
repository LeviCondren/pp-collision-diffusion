#!/usr/bin/env python3
"""
Diagnostic inference script: compare stage-1-sampled log_npart vs true log_npart.

Use --use_true_jet to supply ground-truth log_npart to model.generate (bypassing
the stage-1 ema_jet DDPM sampler).  Without the flag the jet stage runs normally.

Conditioning vector (47d):
  [0:36]   normalised parton kinematics  (6 partons × 6 features)
  [36:42]  binary parton mask             (6 bits)
  [42:47]  one-hot process label          (5 dims: dijet, zjets, ttbar, wjets, wprime)

Output directories:
  --use_true_jet:   {ckpt_dir}/{run_name}/infer_20k_truejet/
  (default):        {ckpt_dir}/{run_name}/infer_20k/

Each rank saves:
    {out_dir}/{proc}_rank{rank:02d}_of{world_size:02d}.npz
with keys: parts_truth, parts_gen, mask, mask_gen, parton_feat
"""

import os, sys, json, argparse, time
import numpy as np

PROCESSES_ORDERED = ['dijet', 'zjets', 'ttbar', 'wjets', 'wprime']

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--rank',       type=int,
                   default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    p.add_argument('--world_size', type=int,
                   default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)))
    p.add_argument('--gpu_id',     type=int, default=None)
    p.add_argument('--num_steps',  type=int, default=50)
    p.add_argument('--chunk_size', type=int, default=200)
    p.add_argument('--val_start',  type=int, default=400000,
                   help='First reserved event in SM files')
    p.add_argument('--n_total',    type=int, default=20000,
                   help='Reserved events per process across all ranks')
    p.add_argument('--npart',      type=int, default=500)
    p.add_argument('--proj_dim',   type=int, default=128)
    p.add_argument('--num_layers', type=int, default=8)
    p.add_argument('--num_gen_layers', type=int, default=2)
    p.add_argument('--run_name',   type=str, default='proc_label_5proc')
    p.add_argument('--data_dir',   type=str,
                   default='/pscratch/sd/l/lcondren/MCsim/full_event_mixed')
    p.add_argument('--wprime_dir', type=str, default=None,
                   help='Directory containing the wprime inference HDF5')
    p.add_argument('--stats_dir',  type=str, default=None)
    p.add_argument('--ckpt_dir',   type=str, default=None)
    p.add_argument('--out_dir',    type=str, default=None)
    p.add_argument('--processes',      nargs='+', default=PROCESSES_ORDERED,
                   help='Ordered list matching training order (sets process label index)')
    p.add_argument('--use_true_jet', action='store_true', default=False,
                   help='Supply ground-truth log_npart to model.generate, bypassing stage-1 sampling')
    return p.parse_args()

args = _parse()

_gpu_id = args.gpu_id if args.gpu_id is not None else args.rank
os.environ['CUDA_VISIBLE_DEVICES']  = str(_gpu_id)
os.environ['TF_GPU_ALLOCATOR']      = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2')

import ctypes as _ctypes
for _lib in [
    '/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64/libcudart.so.12',
    '/global/common/software/nersc9/cudnn/8.9.3-cuda12/lib/libcudnn.so.8',
]:
    try: _ctypes.CDLL(_lib, mode=_ctypes.RTLD_GLOBAL)
    except OSError: pass

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.random.set_seed(42 + args.rank)

print(f'[rank {args.rank}/{args.world_size}] CUDA_VISIBLE_DEVICES={_gpu_id} '
      f'  visible TF GPUs: {len(gpus)}')

MAX_PARTONS = 6
PARTON_FEAT = 6
N_PROC      = len(args.processes)  # 5 by default
NUM_COND    = MAX_PARTONS * PARTON_FEAT + MAX_PARTONS + N_PROC  # 47

CKPT_BASE  = args.ckpt_dir  or f'{args.data_dir}/checkpoints'
STATS_DIR  = args.stats_dir or args.data_dir
WPRIME_DIR = args.wprime_dir or '/pscratch/sd/l/lcondren/MCsim/wprime_inference'
CKPT_PATH  = f'{CKPT_BASE}/{args.run_name}/pet_pp.weights.h5'

_out_suffix = '_truejet' if args.use_true_jet else ''
OUT_DIR     = args.out_dir or f'{CKPT_BASE}/{args.run_name}/infer_20k{_out_suffix}'
os.makedirs(OUT_DIR, exist_ok=True)

print(f'[rank {args.rank}] processes={args.processes}  n_proc={N_PROC}  num_cond={NUM_COND}')
print(f'[rank {args.rank}] checkpoint: {CKPT_PATH}')
print(f'[rank {args.rank}] use_true_jet={args.use_true_jet}  out_dir={OUT_DIR}')

# ── Per-rank event slice ──────────────────────────────────────────────────────
n_per_rank = args.n_total // args.world_size
remainder  = args.n_total % args.world_size
my_n       = n_per_rank + (1 if args.rank < remainder else 0)
my_start   = args.val_start + args.rank * n_per_rank + min(args.rank, remainder)
my_end     = my_start + my_n
# wprime uses its own file from row 0
wp_start   = args.rank * n_per_rank + min(args.rank, remainder)
wp_end     = wp_start + my_n

print(f'[rank {args.rank}] SM event range [{my_start}, {my_end}) = {my_n} events')
print(f'[rank {args.rank}] wprime event range [{wp_start}, {wp_end}) = {my_n} events')

# ── Normalisation stats ───────────────────────────────────────────────────────
import h5py
with open(f'{STATS_DIR}/normalisation_stats.json') as fh:
    stats = json.load(fh)

jet_mean  = float(stats['jet_mean'][0])
jet_std   = float(stats['jet_std'][0])
part_mean = np.array(stats['part_mean'], dtype=np.float32)
part_std  = np.array(stats['part_std'],  dtype=np.float32)

# Cond stats: now 36 dims (6 partons × 6 features)
n_cond_feat   = MAX_PARTONS * PARTON_FEAT  # 36
cond_mean_raw = np.array(stats['cond_mean'], dtype=np.float32)
cond_std_raw  = np.array(stats['cond_std'],  dtype=np.float32)
cond_mean = np.zeros(n_cond_feat, dtype=np.float32)
cond_std  = np.ones(n_cond_feat,  dtype=np.float32)
n_fill = min(len(cond_mean_raw), n_cond_feat)
cond_mean[:n_fill] = cond_mean_raw[:n_fill]
cond_std[:n_fill]  = np.where(cond_std_raw[:n_fill] > 0, cond_std_raw[:n_fill], 1.0)

print(f'[rank {args.rank}] cond_mean dims={len(cond_mean_raw)} -> using {n_cond_feat}')

# ── Data loading helper ───────────────────────────────────────────────────────
def _load_proc(path, s, e, proc_idx, n_proc):
    """Load events [s, e) and build the 47-dim conditioning vector."""
    with h5py.File(path, 'r') as f:
        total = f['particle_features'].shape[0]
        s = min(s, total)
        e = min(e, total)
        if s >= e:
            return None
        pf   = f['particle_features'][s:e].astype(np.float32)
        part = f['parton_features'][s:e].astype(np.float32)  # (N, 6, 6)

        # Use stored n_partons for exact mask (all files have it after add_boson_parton.py)
        if 'n_partons' in f:
            n_par       = f['n_partons'][s:e].astype(np.int32)
            parton_mask = (np.arange(MAX_PARTONS)[None, :] <
                           n_par[:, None]).astype(np.float32)   # (N, 6)
        else:
            # Fallback: infer from is_valid column.
            # part may have fewer slots than MAX_PARTONS before padding below;
            # zero-pad the mask so cond always has shape (N, MAX_PARTONS).
            raw = (np.linalg.norm(part[:, :MAX_PARTONS], axis=2) > 1e-6
                   ).astype(np.float32)
            if raw.shape[1] < MAX_PARTONS:
                pad_m = np.zeros((len(part), MAX_PARTONS - raw.shape[1]),
                                 dtype=np.float32)
                parton_mask = np.concatenate([raw, pad_m], axis=1)
            else:
                parton_mask = raw

    # Pad or truncate to MAX_PARTONS slots
    if part.shape[1] < MAX_PARTONS:
        pad  = np.zeros((len(part), MAX_PARTONS - part.shape[1], PARTON_FEAT),
                        dtype=np.float32)
        part = np.concatenate([part, pad], axis=1)
    part = part[:, :MAX_PARTONS, :]   # (N, 6, 6)

    # Normalise kinematic part of conditioning
    cond_raw  = part.reshape(len(part), n_cond_feat)   # (N, 36)
    cond_norm = (cond_raw - cond_mean) / cond_std       # (N, 36)

    # One-hot process label
    proc_label = np.zeros((len(cond_norm), n_proc), dtype=np.float32)
    proc_label[:, proc_idx] = 1.0                       # (N, 5)

    # Full conditioning: [kinematics | mask | proc_label]
    cond = np.concatenate([cond_norm, parton_mask, proc_label], axis=1)  # (N, 47)

    mask  = pf[:, :, 6]         # (N, npart_max)
    X_raw = pf[:, :, :6]        # (N, npart_max, 6)

    npart     = mask.sum(axis=1, keepdims=True).astype(np.float32)  # (N, 1)
    log_npart = np.log(np.maximum(npart, 1.0))
    jet_truth = (log_npart - jet_mean) / jet_std                     # (N, 1)

    print(f'  loaded {len(pf)} events from {os.path.basename(path)} '
          f'(mean npart={mask.sum(axis=1).mean():.1f}  '
          f'mean n_par={parton_mask.sum(axis=1).mean():.2f}  '
          f'proc_label={proc_idx}/{n_proc})')
    return {'X_raw': X_raw, 'mask': mask, 'y': cond, 'parton_feat': part,
            'jet_truth': jet_truth}

# ── Load all processes ────────────────────────────────────────────────────────
PROCS_SM = [p for p in args.processes if p != 'wprime']
per_proc = {}

for proc in PROCS_SM:
    proc_idx = args.processes.index(proc)
    path     = f'{args.data_dir}/{proc}.hdf5'
    print(f'[rank {args.rank}] {proc} (proc_idx={proc_idx}): {path}  rows [{my_start},{my_end})')
    per_proc[proc] = _load_proc(path, my_start, my_end, proc_idx, N_PROC)
    if per_proc[proc] is None:
        print(f'[rank {args.rank}] {proc}: no events in range, skipping')

if 'wprime' in args.processes:
    wp_idx   = args.processes.index('wprime')
    wp_path  = f'{WPRIME_DIR}/signal_mX0500_mY0100.hdf5'
    # Also try the main data dir
    if not os.path.exists(wp_path):
        wp_path = f'{args.data_dir}/wprime.hdf5'
    print(f'[rank {args.rank}] wprime (proc_idx={wp_idx}): {wp_path}  rows [{wp_start},{wp_end})')
    per_proc['wprime'] = _load_proc(wp_path, wp_start, wp_end, wp_idx, N_PROC)
    if per_proc['wprime'] is None:
        print(f'[rank {args.rank}] wprime: no events in range, skipping')

# ── Load model ────────────────────────────────────────────────────────────────
_scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts)
from proc_label_train import ProcLabelPET

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f'Checkpoint not found: {CKPT_PATH}')

model = ProcLabelPET(
    num_proc_labels=N_PROC,
    num_feat=6, num_jet=1,
    max_partons=MAX_PARTONS, parton_feat=PARTON_FEAT,
    num_part=args.npart,
    projection_dim=args.proj_dim,
    local=True, K=5,
    num_layers=args.num_layers,
    num_gen_layers=args.num_gen_layers,
    drop_probability=0.0,
    simple=False, layer_scale=True, talking_head=False,
    mode='generator',
)
model.load_weights(CKPT_PATH)
print(f'[rank {args.rank}] Loaded {CKPT_PATH}')

# ── Inference ─────────────────────────────────────────────────────────────────
t0_total = time.perf_counter()

for proc in args.processes:
    if per_proc.get(proc) is None:
        continue
    out_file = f'{OUT_DIR}/{proc}_rank{args.rank:02d}_of{args.world_size:02d}.npz'
    if os.path.exists(out_file):
        print(f'[rank {args.rank}] {proc}: already done, skipping -> {out_file}')
        continue
    d    = per_proc[proc]
    cond = d['y']
    N    = len(cond)

    nsplit       = max(1, N // args.chunk_size)
    actual_chunk = N // nsplit
    print(f'[rank {args.rank}] {proc}: {N} events, '
          f'nsplit={nsplit} ({actual_chunk} events/chunk), '
          f'num_steps={args.num_steps}')

    t1 = time.perf_counter()
    if args.use_true_jet:
        parts_gen, jets_gen = model.generate(
            cond=cond,
            jet_mean=jet_mean,
            jet_std=jet_std,
            nsplit=nsplit,
            num_steps=args.num_steps,
            jets=d['jet_truth'],
            use_tqdm=True,
        )
    else:
        parts_gen, jets_gen = model.generate(
            cond=cond,
            jet_mean=jet_mean,
            jet_std=jet_std,
            nsplit=nsplit,
            num_steps=args.num_steps,
            use_tqdm=True,
        )
    dt = time.perf_counter() - t1
    print(f'[rank {args.rank}] {proc}: {dt/60:.2f} min  ({dt/N*1000:.0f} ms/event)')

    log_npart_gen = jets_gen[:, 0] * jet_std + jet_mean
    npart_gen     = np.clip(np.round(np.exp(log_npart_gen)).astype(int), 1, args.npart)
    mask_gen      = (np.arange(args.npart)[None, :] < npart_gen[:, None]).astype(np.float32)
    parts_phys    = (parts_gen * part_std + part_mean) * mask_gen[:, :, None]
    parts_phys[:, :, 5] = np.round(parts_phys[:, :, 5])

    np.savez_compressed(out_file,
        parts_truth  = d['X_raw'],
        parts_gen    = parts_phys,
        mask         = d['mask'],
        mask_gen     = mask_gen,
        parton_feat  = d['parton_feat'],
    )
    print(f'[rank {args.rank}] {proc}: saved -> {out_file}')

    # Sanity check for --use_true_jet: npart_gen should match truth npart
    if args.use_true_jet:
        npart_truth = d['mask'].sum(axis=1).astype(int)
        match_frac  = (npart_gen == npart_truth).mean()
        print(f'[rank {args.rank}] {proc}: true-jet sanity: '
              f'npart match={match_frac:.3f}  '
              f'sample npart_gen={npart_gen[:5].tolist()}  '
              f'sample npart_truth={npart_truth[:5].tolist()}')

print(f'[rank {args.rank}] All done in {(time.perf_counter()-t0_total)/60:.2f} min')
