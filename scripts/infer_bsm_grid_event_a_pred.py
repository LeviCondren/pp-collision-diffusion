#!/usr/bin/env python3
"""Phase 2 BSM grid inference — E020a: event-level MET conditioning, predicted features.

Two-pass generation:
  Pass 1: Run E008 baseline (PET_pp_parton_vpar_bsm, no event conditioning)
          to obtain initial generated particles.
  Compute event features (MET) from Pass-1 particles.
  Pass 2: Run E020a (PET_pp_parton_vpar_bsm_event_a) conditioned on those
          predicted features to produce the final generated particles.

This contrasts with infer_bsm_grid_event_a.py which passes truth event features
(oracle / upper-bound test). The predicted variant tests the closed-loop scenario
relevant to production use.

E020a features: log1p(MET_mag), sin(MET_phi), cos(MET_phi) — NUM_EVENT_FEAT = 3
"""

import os, sys, re, json, argparse, glob, time
import numpy as np

_GRID_DIR_DEFAULT = '/pscratch/sd/l/lcondren/MCsim/wprime_signal'

MAX_PARTONS    = 4
PARTON_FEAT    = 7
NUM_COND       = MAX_PARTONS * PARTON_FEAT + MAX_PARTONS  # 32
MASS_NORM      = 600.0
NUM_EVENT_FEAT = 3   # E020a: log(MET_mag+1), sin(MET_phi), cos(MET_phi)
R_CONE         = 1.0


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--m_X',                type=float, required=True)
    p.add_argument('--m_Y',                type=float, required=True)
    p.add_argument('--grid_dir',           default=_GRID_DIR_DEFAULT)
    p.add_argument('--ckpt_dir',           default=None)
    p.add_argument('--run_name',           default='bsm_grid_event_a')
    p.add_argument('--baseline_run_name',  default='bsm_grid',
                   help='E008 baseline run name (for pass-1 checkpoint)')
    p.add_argument('--stats_path',         default=None)
    p.add_argument('--stats_event_path',   default=None,
                   help='Event stats JSON (default: {grid_dir}/normalisation_stats_event_a.json)')
    p.add_argument('--out_dir',            default=None)
    p.add_argument('--rank',               type=int,
                   default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    p.add_argument('--world_size',         type=int,
                   default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)))
    p.add_argument('--gpu_id',             type=int, default=None)
    p.add_argument('--num_steps',          type=int, default=500,
                   help='DDPM steps for Pass 2 (E020a final generation)')
    p.add_argument('--num_steps_baseline', type=int, default=100,
                   help='DDPM steps for Pass 1 (E008 baseline, to estimate event features)')
    p.add_argument('--chunk_size',         type=int, default=200)
    p.add_argument('--n_total',            type=int, default=5000)
    p.add_argument('--npart',              type=int, default=500)
    p.add_argument('--proj_dim',           type=int, default=128)
    p.add_argument('--num_layers',         type=int, default=8)
    p.add_argument('--num_gen_layers',     type=int, default=2)
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

grid_dir         = args.grid_dir
ckpt_dir         = args.ckpt_dir or os.path.join(grid_dir, 'checkpoints_bsm_grid')
stats_path       = args.stats_path or os.path.join(grid_dir, 'normalisation_stats.json')
stats_event_path = (args.stats_event_path
                    or os.path.join(grid_dir, 'normalisation_stats_event_a.json'))
ckpt_path          = os.path.join(ckpt_dir, args.run_name,          'pet_pp.weights.h5')
baseline_ckpt_path = os.path.join(ckpt_dir, args.baseline_run_name, 'pet_pp.weights.h5')
out_dir = args.out_dir or os.path.join(ckpt_dir, args.run_name, 'infer_holdout_pred')
os.makedirs(out_dir, exist_ok=True)

tag      = f'mX{args.m_X:04.0f}_mY{args.m_Y:04.0f}'
out_file = os.path.join(out_dir, f'bsm_{tag}_rank{args.rank:02d}_of{args.world_size:02d}.npz')

print(f'[rank {args.rank}] target: m_X={args.m_X}  m_Y={args.m_Y}')
print(f'[rank {args.rank}] baseline checkpoint: {baseline_ckpt_path}')
print(f'[rank {args.rank}] E020a checkpoint:    {ckpt_path}')
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

for p in [stats_path, stats_event_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Stats file not found: {p}")

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

def _compute_event_from_gen(parts_phys, mask_gen, part_raw, num_part):
    """Compute 7 event features from GENERATED (physical-space) particles.

    parts_phys: (N, npart, 6) — denormalized (eta, sin_phi, cos_phi, log_pT, pid, charge)
    mask_gen:   (N, npart)    — float32 mask (1=valid, 0=padding)
    part_raw:   (N, 4, 6)     — raw parton features (for cone axis)
    """
    valid   = mask_gen[:, :num_part].astype(bool)
    pT      = np.exp(np.clip(parts_phys[:, :num_part, 3], -10, 10)) * valid
    sp      = parts_phys[:, :num_part, 1]
    cp      = parts_phys[:, :num_part, 2]
    eta     = parts_phys[:, :num_part, 0]
    phi     = np.arctan2(sp, cp)

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
    """Variant a: MET only (3 features)."""
    return raw7[:, :3]


# ── Per-rank event slice ──────────────────────────────────────────────────────

n_per_rank = args.n_total // args.world_size
remainder  = args.n_total % args.world_size
my_n       = n_per_rank + (1 if args.rank < remainder else 0)
my_start   = args.rank * n_per_rank + min(args.rank, remainder)
my_end     = my_start + my_n

print(f'[rank {args.rank}] event range [{my_start}, {my_end}) = {my_n} events')

# ── Load data ─────────────────────────────────────────────────────────────────

with h5py.File(data_path, 'r') as f:
    n_avail = f['particle_features'].shape[0]
    s = min(my_start, n_avail)
    e = min(my_end,   n_avail)
    if s >= e:
        raise RuntimeError(
            f"[rank {args.rank}] Requested range [{my_start}, {my_end}) is beyond "
            f"available events ({n_avail}).")

    file_mx  = float(f.attrs.get('mass_x', m_X_grid))
    file_my  = float(f.attrs.get('mass_y', m_Y_grid))
    pf_raw   = f['particle_features'][s:e].astype(np.float32)
    part_raw = f['parton_features'][s:e].astype(np.float32)

N = len(pf_raw)
print(f'[rank {args.rank}] loaded {N} events (file mass_x={file_mx}, mass_y={file_my})')

# Build parton conditioning vector
mass_col = np.zeros((N, MAX_PARTONS, 1), dtype=np.float32)
mass_col[:, 2, 0] = file_mx / MASS_NORM
mass_col[:, 3, 0] = file_my / MASS_NORM
part7    = np.concatenate([part_raw[:, :MAX_PARTONS, :], mass_col], axis=2)  # (N,4,7)
cond_raw  = part7.reshape(N, MAX_PARTONS * PARTON_FEAT)
cond_norm = (cond_raw - cond_mean) / cond_std
parton_mask = np.ones((N, MAX_PARTONS), dtype=np.float32)
cond = np.concatenate([cond_norm, parton_mask], axis=1)  # (N, 32)

# Truth particle info (for comparison output only)
mask_truth = pf_raw[:, :args.npart, 6].astype(np.float32)
X_raw      = pf_raw[:, :args.npart, :6]

npart_truth = mask_truth.sum(axis=1, keepdims=True)
log_npart_truth = np.log(np.maximum(npart_truth, 1.0))
jet_truth = (log_npart_truth - jet_mean) / jet_std

print(f'[rank {args.rank}] mean truth npart={mask_truth.sum(axis=1).mean():.1f}')

# ── Load model architecture ───────────────────────────────────────────────────

_scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts)
from PET_pp_parton_vpar_bsm       import PET_pp_parton_vpar_bsm
from PET_pp_parton_vpar_bsm_event_a import PET_pp_parton_vpar_bsm_event_a

for p in [baseline_ckpt_path, ckpt_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f'Checkpoint not found: {p}')

# ── Pass 1: E008 baseline generation (no event conditioning) ─────────────────

print(f'\n[rank {args.rank}] === PASS 1: E008 baseline ({args.num_steps_baseline} steps) ===')
model_baseline = PET_pp_parton_vpar_bsm(
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
model_baseline.load_weights(baseline_ckpt_path)
print(f'[rank {args.rank}] Loaded baseline: {baseline_ckpt_path}')

nsplit = max(1, N // args.chunk_size)
t1 = time.perf_counter()
parts_gen_p1, jets_gen_p1 = model_baseline.generate(
    cond=cond,
    jet_mean=jet_mean,
    jet_std=jet_std,
    nsplit=nsplit,
    num_steps=args.num_steps_baseline,
    jets=None,
    use_tqdm=True,
)
dt1 = time.perf_counter() - t1
print(f'[rank {args.rank}] Pass 1 done in {dt1/60:.2f} min')

# Denormalize pass-1 particles
log_npart_p1 = jets_gen_p1[:, 0] * jet_std + jet_mean
npart_p1     = np.clip(np.round(np.exp(log_npart_p1)).astype(int), 1, args.npart)
mask_gen_p1  = (np.arange(args.npart)[None, :] < npart_p1[:, None]).astype(np.float32)
parts_phys_p1 = (parts_gen_p1 * part_std + part_mean) * mask_gen_p1[:, :, None]
parts_phys_p1[:, :, 5] = np.round(parts_phys_p1[:, :, 5])

print(f'[rank {args.rank}] Pass-1 mean npart={mask_gen_p1.sum(axis=1).mean():.1f}')

# Free baseline model memory
del model_baseline, parts_gen_p1, jets_gen_p1
tf.keras.backend.clear_session()

# ── Compute event features from Pass-1 particles ─────────────────────────────

raw7         = _compute_event_from_gen(parts_phys_p1, mask_gen_p1, part_raw, args.npart)
event_raw    = _assemble_event_feat(raw7)                            # (N, 3)
event_feat   = (event_raw - event_mean) / event_std                 # (N, 3) normalized

print(f'[rank {args.rank}] Predicted event_feat sample[0]: {event_feat[0]}')
print(f'[rank {args.rank}] Predicted event_feat mean: {event_feat.mean(0)}  '
      f'std: {event_feat.std(0)}')

# ── Pass 2: E020a generation conditioned on predicted event features ──────────

print(f'\n[rank {args.rank}] === PASS 2: E020a ({args.num_steps} steps) ===')
model = PET_pp_parton_vpar_bsm_event_a(
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
print(f'[rank {args.rank}] Loaded E020a: {ckpt_path}')

t2 = time.perf_counter()
parts_gen, jets_gen = model.generate(
    cond=cond,
    jet_mean=jet_mean,
    jet_std=jet_std,
    event_feat=event_feat,
    nsplit=nsplit,
    num_steps=args.num_steps,
    jets=None,
    use_tqdm=True,
)
dt2 = time.perf_counter() - t2
print(f'[rank {args.rank}] Pass 2 done in {dt2/60:.2f} min')

log_npart_gen = jets_gen[:, 0] * jet_std + jet_mean
npart_gen     = np.clip(np.round(np.exp(log_npart_gen)).astype(int), 1, args.npart)
mask_gen      = (np.arange(args.npart)[None, :] < npart_gen[:, None]).astype(np.float32)
parts_phys    = (parts_gen * part_std + part_mean) * mask_gen[:, :, None]
parts_phys[:, :, 5] = np.round(parts_phys[:, :, 5])

np.savez_compressed(out_file,
    parts_truth         = X_raw,
    parts_gen           = parts_phys,
    parts_gen_baseline  = parts_phys_p1,
    mask                = mask_truth,
    mask_gen            = mask_gen,
    mask_gen_baseline   = mask_gen_p1,
    parton_feat         = part7,
    mass_x              = np.float32(file_mx),
    mass_y              = np.float32(file_my),
    event_feat          = event_feat,
)
print(f'[rank {args.rank}] saved → {out_file}')
total = time.perf_counter() - t1
print(f'[rank {args.rank}] total {total/60:.2f} min  '
      f'(pass1={dt1/60:.2f} pass2={dt2/60:.2f})')
