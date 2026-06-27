#!/usr/bin/env python3
"""Phase 2 BSM grid inference — E020b: event-level cone_X conditioning.

Copied from infer_bsm_grid.py and extended for E020b:
  - Reads truth event features (MET) from holdout HDF5 data.
  - Normalizes using normalisation_stats_event_b.json.
  - Passes truth event features to model.generate() — stage-2 isolation test.

For a clean stage-2 evaluation pass --use_truth_jet to also bypass stage-1
multiplicity prediction.

Do NOT modify the original infer_bsm_grid.py.
"""

import os, sys, re, json, argparse, glob, time
import numpy as np

_GRID_DIR_DEFAULT = '/pscratch/sd/l/lcondren/MCsim/wprime_signal'

MAX_PARTONS    = 4
PARTON_FEAT    = 7
NUM_COND       = MAX_PARTONS * PARTON_FEAT + MAX_PARTONS  # 32
MASS_NORM      = 600.0
NUM_EVENT_FEAT = 2   # E020b: log(cone_pT_X+1), log(cone_mass_X+1)
R_CONE         = 1.0


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--m_X',              type=float, required=True)
    p.add_argument('--m_Y',              type=float, required=True)
    p.add_argument('--grid_dir',         default=_GRID_DIR_DEFAULT)
    p.add_argument('--ckpt_dir',         default=None)
    p.add_argument('--run_name',         default='bsm_grid_event_b')
    p.add_argument('--stats_path',       default=None)
    p.add_argument('--stats_event_path', default=None,
                   help='Event stats JSON (default: {grid_dir}/normalisation_stats_event_b.json)')
    p.add_argument('--out_dir',          default=None)
    p.add_argument('--rank',             type=int,
                   default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    p.add_argument('--world_size',       type=int,
                   default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)))
    p.add_argument('--gpu_id',           type=int, default=None)
    p.add_argument('--num_steps',        type=int, default=500)
    p.add_argument('--chunk_size',       type=int, default=200)
    p.add_argument('--n_total',          type=int, default=10000)
    p.add_argument('--npart',            type=int, default=500)
    p.add_argument('--proj_dim',         type=int, default=128)
    p.add_argument('--num_layers',       type=int, default=8)
    p.add_argument('--num_gen_layers',   type=int, default=2)
    p.add_argument('--use_truth_jet',    action='store_true', default=False,
                   help='Condition particle generation on truth log_npart (bypasses stage 1)')
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

grid_dir          = args.grid_dir
ckpt_dir          = args.ckpt_dir or os.path.join(grid_dir, 'checkpoints_bsm_grid')
stats_path        = args.stats_path or os.path.join(grid_dir, 'normalisation_stats.json')
stats_event_path  = (args.stats_event_path
                     or os.path.join(grid_dir, 'normalisation_stats_event_b.json'))
ckpt_path         = os.path.join(ckpt_dir, args.run_name, 'pet_pp.weights.h5')
out_dir           = args.out_dir or os.path.join(ckpt_dir, args.run_name, 'infer')
os.makedirs(out_dir, exist_ok=True)

tag      = f'mX{args.m_X:04.0f}_mY{args.m_Y:04.0f}'
out_file = os.path.join(out_dir, f'bsm_{tag}_rank{args.rank:02d}_of{args.world_size:02d}.npz')

print(f'[rank {args.rank}] target: m_X={args.m_X}  m_Y={args.m_Y}')
print(f'[rank {args.rank}] checkpoint: {ckpt_path}')
print(f'[rank {args.rank}] output: {out_file}')

if os.path.exists(out_file):
    print(f'[rank {args.rank}] Output already exists, skipping.')
    sys.exit(0)

# ── Find nearest grid file ────────────────────────────────────────────────────

def _parse_masses_from_filename(fname):
    m = re.search(r'signal_mX(\d+)_mY(\d+)', os.path.basename(fname))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def find_nearest_signal_file(grid_dir, m_X, m_Y):
    files = sorted(glob.glob(f'{grid_dir}/signal_mX*.hdf5'))
    if not files:
        raise FileNotFoundError(f"No signal files in {grid_dir}")
    best_file, best_dist = None, float('inf')
    for f in files:
        fx, fy = _parse_masses_from_filename(f)
        if fx is None: continue
        dist = (fx - m_X)**2 + (fy - m_Y)**2
        if dist < best_dist:
            best_dist = dist; best_file = f
    fx_b, fy_b = _parse_masses_from_filename(best_file)
    print(f'[rank {args.rank}] target ({m_X},{m_Y}) → nearest ({fx_b},{fy_b}): '
          f'{os.path.basename(best_file)}  (dist={best_dist**0.5:.1f} GeV)')
    return best_file, fx_b, fy_b


data_path, m_X_grid, m_Y_grid = find_nearest_signal_file(grid_dir, args.m_X, args.m_Y)

# ── Load normalisation stats ──────────────────────────────────────────────────

if not os.path.exists(stats_path):
    raise FileNotFoundError(f"Stats file not found: {stats_path}")
if not os.path.exists(stats_event_path):
    raise FileNotFoundError(
        f"Event stats not found: {stats_event_path}\n"
        f"Run bsm_grid_train_event_b.py first to compute them.")

with open(stats_path) as fh:
    stats = json.load(fh)
with open(stats_event_path) as fh:
    event_stats = json.load(fh)

cond_mean_raw = np.array(stats['cond_mean'], dtype=np.float32)
expected_cond = MAX_PARTONS * PARTON_FEAT
if len(cond_mean_raw) != expected_cond:
    raise ValueError(f"Stats file has {len(cond_mean_raw)} cond dims, expected {expected_cond}.")

cond_mean = cond_mean_raw
cond_std  = np.array(stats['cond_std'],  dtype=np.float32)
jet_mean  = float(stats['jet_mean'][0])
jet_std   = float(stats['jet_std'][0])
part_mean = np.array(stats['part_mean'], dtype=np.float32)
part_std  = np.array(stats['part_std'],  dtype=np.float32)

event_mean = np.array(event_stats['event_mean'], dtype=np.float32)
event_std  = np.array(event_stats['event_std'],  dtype=np.float32)

print(f'[rank {args.rank}] stats loaded: cond_dims={len(cond_mean)}  '
      f'part_dims={len(part_mean)}  jet_mean={jet_mean:.3f}')
print(f'[rank {args.rank}] event stats: mean={event_mean}  std={event_std}')

# ── Event feature helpers ─────────────────────────────────────────────────────

def _compute_event_raw_all7(pf_raw, part_raw, num_part):
    """Same as training — computes 7 log-space event features."""
    valid = pf_raw[:, :num_part, 6].astype(bool)
    pT    = np.exp(np.clip(pf_raw[:, :num_part, 3], -10, 10)) * valid
    sp    = pf_raw[:, :num_part, 1]
    cp    = pf_raw[:, :num_part, 2]
    eta   = pf_raw[:, :num_part, 0]
    phi   = np.arctan2(sp, cp)

    MET_x   = (pT * cp).sum(1)
    MET_y   = (pT * sp).sum(1)
    met_mag = np.sqrt(MET_x**2 + MET_y**2)
    met_phi = np.arctan2(MET_y, MET_x)
    feats   = [np.log1p(met_mag), np.sin(met_phi), np.cos(met_phi)]

    eta_clip = np.clip(eta, -8, 8)
    for slot in [2, 3]:
        pze    = np.clip(part_raw[:, slot, 3], -1 + 1e-7, 1 - 1e-7)
        eta_p  = 0.5 * np.log((1 + pze) / (1 - pze))
        phi_p  = np.arctan2(part_raw[:, slot, 1], part_raw[:, slot, 2])
        deta   = eta - eta_p[:, None]
        dphi   = phi - phi_p[:, None]
        dphi   = (dphi + np.pi) % (2 * np.pi) - np.pi
        dR     = np.sqrt(deta**2 + dphi**2)
        in_c   = (dR < R_CONE) & valid
        wt     = pT * in_c
        pT_c   = wt.sum(1)
        E_c    = (wt * np.cosh(eta_clip)).sum(1)
        px_c   = (wt * cp).sum(1)
        py_c   = (wt * sp).sum(1)
        pz_c   = (wt * np.sinh(eta_clip)).sum(1)
        m2     = np.maximum(E_c**2 - px_c**2 - py_c**2 - pz_c**2, 0.0)
        feats.append(np.log1p(pT_c))
        feats.append(np.log1p(np.sqrt(m2)))

    return np.stack(feats, axis=1).astype(np.float32)


def _assemble_event_feat(raw7):
    """Variant b: cone_X only (2 features)."""
    return raw7[:, 3:5]


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
            f"available events ({n_avail}). Reduce --n_total or --world_size.")

    file_mx  = float(f.attrs.get('mass_x', m_X_grid))
    file_my  = float(f.attrs.get('mass_y', m_Y_grid))
    pf_raw   = f['particle_features'][s:e].astype(np.float32)
    part_raw = f['parton_features'][s:e].astype(np.float32)

N = len(pf_raw)
print(f'[rank {args.rank}] loaded {N} events (file mass_x={file_mx}, mass_y={file_my})')

# Build 7-feature parton conditioning vector
mass_col = np.zeros((N, MAX_PARTONS, 1), dtype=np.float32)
mass_col[:, 2, 0] = file_mx / MASS_NORM
mass_col[:, 3, 0] = file_my / MASS_NORM
part7    = np.concatenate([part_raw[:, :MAX_PARTONS, :], mass_col], axis=2)  # (N,4,7)
cond_raw  = part7.reshape(N, MAX_PARTONS * PARTON_FEAT)
cond_norm = (cond_raw - cond_mean) / cond_std
parton_mask = np.ones((N, MAX_PARTONS), dtype=np.float32)
cond = np.concatenate([cond_norm, parton_mask], axis=1)  # (N, 32)

# Compute truth event features from raw particle data
raw7         = _compute_event_raw_all7(pf_raw, part_raw, args.npart)
event_raw    = _assemble_event_feat(raw7)                            # (N, 3)
event_feat   = (event_raw - event_mean) / event_std                 # (N, 3) normalized

print(f'[rank {args.rank}] event_feat sample[0]: {event_feat[0]}')
print(f'[rank {args.rank}] event_feat mean: {event_feat.mean(0)}  std: {event_feat.std(0)}')

# Truth particle info
mask_truth = pf_raw[:, :args.npart, 6].astype(np.float32)
X_raw      = pf_raw[:, :args.npart, :6]

npart     = mask_truth.sum(axis=1, keepdims=True)
log_npart = np.log(np.maximum(npart, 1.0))
jet_truth = (log_npart - jet_mean) / jet_std

print(f'[rank {args.rank}] mean truth npart={mask_truth.sum(axis=1).mean():.1f}')

# ── Load model ────────────────────────────────────────────────────────────────

_scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts)
from PET_pp_parton_vpar_bsm_event_b import PET_pp_parton_vpar_bsm_event_b

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

model = PET_pp_parton_vpar_bsm_event_b(
    num_feat=6, num_jet=1,
    max_partons=MAX_PARTONS,
    parton_feat=PARTON_FEAT,
    num_event_feat=NUM_EVENT_FEAT,
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
    event_feat=event_feat,
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
    parts_truth  = X_raw,
    parts_gen    = parts_phys,
    mask         = mask_truth,
    mask_gen     = mask_gen,
    parton_feat  = part7,
    mass_x       = np.float32(file_mx),
    mass_y       = np.float32(file_my),
    event_feat   = event_feat,
)
print(f'[rank {args.rank}] saved → {out_file}')
print(f'[rank {args.rank}] done in {(time.perf_counter()-t1)/60:.2f} min total')
