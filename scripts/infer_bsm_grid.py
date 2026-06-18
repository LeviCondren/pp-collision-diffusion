#!/usr/bin/env python3
"""
Phase 2 BSM grid inference script.

Generates events conditioned on a target (m_X, m_Y) mass point, loading
from the nearest available grid file in /pscratch/sd/l/lcondren/MCsim/wprime_signal/.

Conditioning vector (32-dim):
  [0:28]   normalised parton features (4 partons × 7 features)
  [28:32]  binary parton mask (always [1,1,1,1])

The 7th parton feature is mass / 600:
  slot 0  incoming beam-A  → 0
  slot 1  incoming beam-B  → 0
  slot 2  X                → m_X / 600
  slot 3  Y                → m_Y / 600

Output: {out_dir}/bsm_mX{m_X:04.0f}_mY{m_Y:04.0f}_rank{rank:02d}_of{world_size:02d}.npz
  keys: parts_truth, parts_gen, mask, mask_gen, parton_feat, mass_x, mass_y
"""

import os, sys, re, json, argparse, glob, time
import numpy as np

_GRID_DIR_DEFAULT = '/pscratch/sd/l/lcondren/MCsim/wprime_signal'

MAX_PARTONS = 4
PARTON_FEAT = 7
NUM_COND    = MAX_PARTONS * PARTON_FEAT + MAX_PARTONS  # 32
MASS_NORM   = 600.0


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--m_X',          type=float, required=True,
                   help='Target X mass in GeV (nearest grid point is used)')
    p.add_argument('--m_Y',          type=float, required=True,
                   help='Target Y mass in GeV (nearest grid point is used)')
    p.add_argument('--grid_dir',     default=_GRID_DIR_DEFAULT)
    p.add_argument('--ckpt_dir',     default=None,
                   help='Checkpoint root (default: {grid_dir}/checkpoints_bsm_grid)')
    p.add_argument('--run_name',     default='bsm_grid')
    p.add_argument('--stats_path',   default=None,
                   help='Normalisation stats (default: {grid_dir}/normalisation_stats.json)')
    p.add_argument('--out_dir',      default=None,
                   help='Output directory (default: {ckpt_dir}/{run_name}/infer)')
    p.add_argument('--rank',         type=int,
                   default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    p.add_argument('--world_size',   type=int,
                   default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)))
    p.add_argument('--gpu_id',       type=int, default=None)
    p.add_argument('--num_steps',    type=int, default=500)
    p.add_argument('--chunk_size',   type=int, default=200)
    p.add_argument('--n_total',      type=int, default=10000,
                   help='Total events to generate (split across ranks)')
    p.add_argument('--npart',        type=int, default=500)
    p.add_argument('--proj_dim',     type=int, default=128)
    p.add_argument('--num_layers',   type=int, default=8)
    p.add_argument('--num_gen_layers', type=int, default=2)
    p.add_argument('--use_truth_jet', action='store_true', default=False,
                   help='Condition particle generation on truth log_npart (not generated)')
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
import h5py

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.random.set_seed(42 + args.rank)

print(f'[rank {args.rank}/{args.world_size}] CUDA_VISIBLE_DEVICES={_gpu_id}  '
      f'visible TF GPUs: {len(gpus)}')

# ── Paths ─────────────────────────────────────────────────────────────────────

grid_dir   = args.grid_dir
ckpt_dir   = args.ckpt_dir   or os.path.join(grid_dir, 'checkpoints_bsm_grid')
stats_path = args.stats_path or os.path.join(grid_dir, 'normalisation_stats.json')
ckpt_path  = os.path.join(ckpt_dir, args.run_name, 'pet_pp.weights.h5')
out_dir    = args.out_dir    or os.path.join(ckpt_dir, args.run_name, 'infer')
os.makedirs(out_dir, exist_ok=True)

tag = f'mX{args.m_X:04.0f}_mY{args.m_Y:04.0f}'
out_file = os.path.join(out_dir,
                        f'bsm_{tag}_rank{args.rank:02d}_of{args.world_size:02d}.npz')

print(f'[rank {args.rank}] target: m_X={args.m_X}  m_Y={args.m_Y}')
print(f'[rank {args.rank}] checkpoint: {ckpt_path}')
print(f'[rank {args.rank}] output: {out_file}')

if os.path.exists(out_file):
    print(f'[rank {args.rank}] Output already exists, skipping.')
    sys.exit(0)

# ── Find nearest grid file ────────────────────────────────────────────────────

def _parse_masses_from_filename(fname):
    m = re.search(r'signal_mX(\d+)_mY(\d+)', os.path.basename(fname))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def find_nearest_signal_file(grid_dir, m_X, m_Y):
    files = sorted(glob.glob(f'{grid_dir}/signal_mX*.hdf5'))
    if not files:
        raise FileNotFoundError(f"No signal files found in {grid_dir}")
    best_file = None
    best_dist = float('inf')
    for f in files:
        fx, fy = _parse_masses_from_filename(f)
        if fx is None:
            continue
        dist = (fx - m_X) ** 2 + (fy - m_Y) ** 2
        if dist < best_dist:
            best_dist = dist
            best_file = f
    fx_best, fy_best = _parse_masses_from_filename(best_file)
    print(f'[rank {args.rank}] target ({m_X}, {m_Y}) → nearest grid ({fx_best}, {fy_best}): '
          f'{os.path.basename(best_file)}  (dist={best_dist**0.5:.1f} GeV)')
    return best_file, fx_best, fy_best


data_path, m_X_grid, m_Y_grid = find_nearest_signal_file(grid_dir, args.m_X, args.m_Y)

# ── Load normalisation stats ──────────────────────────────────────────────────

if not os.path.exists(stats_path):
    raise FileNotFoundError(
        f"Stats file not found: {stats_path}\n"
        f"Run bsm_grid_train.py to compute and save normalisation stats first.")

with open(stats_path) as fh:
    stats = json.load(fh)

cond_mean_raw = np.array(stats['cond_mean'], dtype=np.float32)
expected_cond = MAX_PARTONS * PARTON_FEAT  # 28
if len(cond_mean_raw) != expected_cond:
    raise ValueError(
        f"FATAL: Stats file '{stats_path}' has cond_mean with {len(cond_mean_raw)} dims, "
        f"expected {expected_cond}. Wrong stats file? Phase 1 stats have 36 dims.")

cond_mean = cond_mean_raw
cond_std  = np.array(stats['cond_std'],  dtype=np.float32)
jet_mean  = float(stats['jet_mean'][0])
jet_std   = float(stats['jet_std'][0])
part_mean = np.array(stats['part_mean'], dtype=np.float32)
part_std  = np.array(stats['part_std'],  dtype=np.float32)

print(f'[rank {args.rank}] stats loaded: cond_dims={len(cond_mean)}  '
      f'part_dims={len(part_mean)}  jet_mean={jet_mean:.3f}')

# ── Per-rank event slice ──────────────────────────────────────────────────────

n_per_rank = args.n_total // args.world_size
remainder  = args.n_total % args.world_size
my_n       = n_per_rank + (1 if args.rank < remainder else 0)
my_start   = args.rank * n_per_rank + min(args.rank, remainder)
my_end     = my_start + my_n

print(f'[rank {args.rank}] event range [{my_start}, {my_end}) = {my_n} events')

# ── Load data and build conditioning ─────────────────────────────────────────

with h5py.File(data_path, 'r') as f:
    n_avail = f['particle_features'].shape[0]
    s = min(my_start, n_avail)
    e = min(my_end,   n_avail)
    if s >= e:
        raise RuntimeError(
            f"[rank {args.rank}] Requested range [{my_start}, {my_end}) is beyond "
            f"available events ({n_avail}) in {data_path}. "
            f"Reduce --n_total or --world_size.")

    file_mx = float(f.attrs.get('mass_x', m_X_grid))
    file_my = float(f.attrs.get('mass_y', m_Y_grid))

    pf_raw   = f['particle_features'][s:e].astype(np.float32)
    part_raw = f['parton_features'][s:e].astype(np.float32)

N = len(pf_raw)
print(f'[rank {args.rank}] loaded {N} events from {os.path.basename(data_path)} '
      f'(file mass_x={file_mx}, mass_y={file_my})')

# Build 7-feature parton vector with mass as 7th feature
mass_col = np.zeros((N, MAX_PARTONS, 1), dtype=np.float32)
mass_col[:, 2, 0] = file_mx / MASS_NORM
mass_col[:, 3, 0] = file_my / MASS_NORM
part7 = np.concatenate([part_raw[:, :MAX_PARTONS, :], mass_col], axis=2)  # (N, 4, 7)

# Normalise conditioning
cond_raw  = part7.reshape(N, MAX_PARTONS * PARTON_FEAT)  # (N, 28)
cond_norm = (cond_raw - cond_mean) / cond_std             # (N, 28)
parton_mask = np.ones((N, MAX_PARTONS), dtype=np.float32) # always all-ones
cond = np.concatenate([cond_norm, parton_mask], axis=1)   # (N, 32)

# Truth particle info
mask_truth = pf_raw[:, :args.npart, 6].astype(np.float32)  # (N, npart)
X_raw      = pf_raw[:, :args.npart, :6]                     # (N, npart, 6)

npart     = mask_truth.sum(axis=1, keepdims=True)
log_npart = np.log(np.maximum(npart, 1.0))
jet_truth = (log_npart - jet_mean) / jet_std                # (N, 1)

print(f'[rank {args.rank}] mean truth npart={mask_truth.sum(axis=1).mean():.1f}  '
      f'mean n_partons={parton_mask.sum(axis=1).mean():.2f}')

# ── Load model ────────────────────────────────────────────────────────────────

_scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts)
from PET_pp_parton_vpar_bsm import PET_pp_parton_vpar_bsm

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

model = PET_pp_parton_vpar_bsm(
    num_feat=6, num_jet=1,
    max_partons=MAX_PARTONS,
    parton_feat=PARTON_FEAT,
    num_part=args.npart,
    projection_dim=args.proj_dim,
    local=True, K=5,
    num_layers=args.num_layers,
    num_gen_layers=args.num_gen_layers,
    drop_probability=0.0,
    simple=False, layer_scale=True, talking_head=False,
    mode='generator',
)
model.load_weights(ckpt_path)
print(f'[rank {args.rank}] Loaded {ckpt_path}')

# ── Generate ──────────────────────────────────────────────────────────────────

nsplit       = max(1, N // args.chunk_size)
actual_chunk = N // nsplit
print(f'[rank {args.rank}] generating {N} events  nsplit={nsplit} '
      f'({actual_chunk} events/chunk)  num_steps={args.num_steps}')

jets_in = jet_truth if args.use_truth_jet else None

t1 = time.perf_counter()
parts_gen, jets_gen = model.generate(
    cond=cond,
    jet_mean=jet_mean,
    jet_std=jet_std,
    nsplit=nsplit,
    num_steps=args.num_steps,
    jets=jets_in,
    use_tqdm=True,
)
dt = time.perf_counter() - t1
print(f'[rank {args.rank}] generated in {dt/60:.2f} min  ({dt/N*1000:.0f} ms/event)')

log_npart_gen = jets_gen[:, 0] * jet_std + jet_mean
npart_gen     = np.clip(np.round(np.exp(log_npart_gen)).astype(int), 1, args.npart)
mask_gen      = (np.arange(args.npart)[None, :] < npart_gen[:, None]).astype(np.float32)
parts_phys    = (parts_gen * part_std + part_mean) * mask_gen[:, :, None]
parts_phys[:, :, 5] = np.round(parts_phys[:, :, 5])

np.savez_compressed(out_file,
    parts_truth = X_raw,
    parts_gen   = parts_phys,
    mask        = mask_truth,
    mask_gen    = mask_gen,
    parton_feat = part7,       # (N, 4, 7) — includes mass feature
    mass_x      = np.float32(file_mx),
    mass_y      = np.float32(file_my),
)
print(f'[rank {args.rank}] saved → {out_file}')
print(f'[rank {args.rank}] done in {(time.perf_counter()-t1)/60:.2f} min total')
