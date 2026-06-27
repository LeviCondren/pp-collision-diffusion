#!/usr/bin/env python3
"""
Inference script for the 5-process vpar model (dijet, zjets, ttbar, wjets, wprime).

Extends infer_pp.py to use PET_pp_parton_vpar (MAX_PARTONS=5, num_cond=35).
SM processes use events [val_start, val_start+n_total] from full_event_mixed/.
wprime uses [0, n_total] from a dedicated inference file.

Each rank saves:
    {out_dir}/{proc}_rank{rank:02d}_of{world_size:02d}.npz
with keys: parts_truth, parts_gen, mask, mask_gen, parton_feat
"""

import os, sys, json, argparse, time
import numpy as np

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
    p.add_argument('--num_layers',     type=int, default=8)
    p.add_argument('--num_gen_layers', type=int, default=2)
    p.add_argument('--processes',      nargs='+',
                   default=['dijet', 'zjets', 'ttbar', 'wjets', 'wprime'],
                   help='Process names in the training order (sets one-hot index)')
    p.add_argument('--run_name',       type=str, default='proc_label_5proc')
    p.add_argument('--data_dir',   type=str,
                   default='/pscratch/sd/l/lcondren/MCsim/full_event_mixed')
    p.add_argument('--wprime_dir', type=str, default=None,
                   help='Directory containing the wprime inference HDF5 '
                        '(default: /pscratch/sd/l/lcondren/MCsim/wprime_inference)')
    p.add_argument('--stats_dir',  type=str, default=None)
    p.add_argument('--ckpt_dir',   type=str, default=None)
    p.add_argument('--out_dir',    type=str, default=None)
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

MAX_PARTONS = 5
PARTON_FEAT = 6

PROC_ORDER = args.processes          # names in training order → one-hot index
N_PROC     = len(PROC_ORDER)

CKPT_BASE  = args.ckpt_dir  or f'{args.data_dir}/checkpoints'
STATS_DIR  = args.stats_dir or args.data_dir
WPRIME_DIR = args.wprime_dir or '/pscratch/sd/l/lcondren/MCsim/wprime_inference'
CKPT_PATH  = f'{CKPT_BASE}/{args.run_name}/pet_pp.weights.h5'
OUT_DIR    = args.out_dir or f'{CKPT_BASE}/{args.run_name}/infer_20k'
os.makedirs(OUT_DIR, exist_ok=True)

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

# Extend cond stats to MAX_PARTONS*PARTON_FEAT (stats file has 4-parton, 24-dim)
n_cond_feat  = MAX_PARTONS * PARTON_FEAT  # 30
cond_mean_raw = np.array(stats['cond_mean'], dtype=np.float32)
cond_std_raw  = np.array(stats['cond_std'],  dtype=np.float32)
cond_mean = np.zeros(n_cond_feat, dtype=np.float32)
cond_std  = np.ones(n_cond_feat,  dtype=np.float32)
n_fill = min(len(cond_mean_raw), n_cond_feat)
cond_mean[:n_fill] = cond_mean_raw[:n_fill]
cond_std[:n_fill]  = np.where(cond_std_raw[:n_fill] > 0, cond_std_raw[:n_fill], 1.0)

# ── Data loading helper ───────────────────────────────────────────────────────
def _load_proc(path, s, e, proc_idx=None):
    with h5py.File(path, 'r') as f:
        total = f['particle_features'].shape[0]
        s = min(s, total)
        e = min(e, total)
        if s >= e:
            return None
        pf   = f['particle_features'][s:e].astype(np.float32)
        part = f['parton_features'][s:e].astype(np.float32)

    # Pad parton slots to MAX_PARTONS
    if part.shape[1] < MAX_PARTONS:
        pad = np.zeros((len(part), MAX_PARTONS - part.shape[1], PARTON_FEAT),
                       dtype=np.float32)
        part = np.concatenate([part, pad], axis=1)
    part = part[:, :MAX_PARTONS, :]  # (N, 5, 6)

    # Build parton mask from occupancy (last feature column)
    parton_mask = (np.linalg.norm(part, axis=2) > 1e-6).astype(np.float32)  # (N, 5)

    mask   = pf[:, :, 6]                            # (N, 500)
    X_raw  = pf[:, :, :6]                           # (N, 500, 6)
    cond_raw  = part.reshape(len(part), n_cond_feat)  # (N, 30)
    cond_norm = (cond_raw - cond_mean) / cond_std
    cond = np.concatenate([cond_norm, parton_mask], axis=1)  # (N, 35)

    # Append one-hot process label to match ProcLabelPET conditioning
    if proc_idx is not None:
        proc_label = np.zeros((len(cond), N_PROC), dtype=np.float32)
        proc_label[:, proc_idx] = 1.0
        cond = np.concatenate([cond, proc_label], axis=1)  # (N, 40)

    print(f'  loaded {len(pf)} events (mean npart={mask.sum(axis=1).mean():.1f})')
    return {'X_raw': X_raw, 'mask': mask, 'y': cond, 'parton_feat': part}

# ── Load all processes ────────────────────────────────────────────────────────
PROCS_SM = ['dijet', 'zjets', 'ttbar', 'wjets']
per_proc = {}

for proc in PROCS_SM:
    path = f'{args.data_dir}/{proc}.hdf5'
    print(f'[rank {args.rank}] {proc}: {path}  rows [{my_start},{my_end})')
    pidx = PROC_ORDER.index(proc) if proc in PROC_ORDER else None
    per_proc[proc] = _load_proc(path, my_start, my_end, proc_idx=pidx)
    if per_proc[proc] is None:
        print(f'[rank {args.rank}] {proc}: no events in range, skipping')

wp_path = f'{WPRIME_DIR}/signal_mX0500_mY0100.hdf5'
print(f'[rank {args.rank}] wprime: {wp_path}  rows [{wp_start},{wp_end})')
wp_idx = PROC_ORDER.index('wprime') if 'wprime' in PROC_ORDER else None
per_proc['wprime'] = _load_proc(wp_path, wp_start, wp_end, proc_idx=wp_idx)
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
ALL_PROCS = PROCS_SM + ['wprime']

for proc in ALL_PROCS:
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

print(f'[rank {args.rank}] All done in {(time.perf_counter()-t0_total)/60:.2f} min')
