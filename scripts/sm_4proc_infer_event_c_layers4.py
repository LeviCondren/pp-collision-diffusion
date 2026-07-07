#!/usr/bin/env python3
"""E027 inference on SM 4-process holdout.

Runs the E022-architecture model (sm_4proc_event_c_layers4) on holdout events
from one SM process (dijet/ttbar/wjets/zjets). Holdout slice: events
[holdout_start : holdout_start + n_total] (default holdout_start=490000, n_total=5000).

Usage (4 processes in parallel, one per GPU):
  python3 sm_4proc_infer_event_c_layers4.py --process dijet  --gpu_id 0 &
  python3 sm_4proc_infer_event_c_layers4.py --process ttbar  --gpu_id 1 &
  python3 sm_4proc_infer_event_c_layers4.py --process wjets  --gpu_id 2 &
  python3 sm_4proc_infer_event_c_layers4.py --process zjets  --gpu_id 3 &
  wait
"""

import os, sys, json, argparse, time
import numpy as np

_SM_DIR_DEFAULT = '/pscratch/sd/l/lcondren/MCsim/full_event_mixed'
SM_PROCESSES    = ['dijet', 'ttbar', 'wjets', 'zjets']

MAX_PARTONS    = 4
PARTON_FEAT    = 7
NUM_COND       = MAX_PARTONS * PARTON_FEAT + MAX_PARTONS  # 32
MASS_NORM      = 600.0
NUM_EVENT_FEAT = 7
R_CONE         = 1.0


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--process',            required=True, choices=SM_PROCESSES)
    p.add_argument('--sm_dir',             default=_SM_DIR_DEFAULT)
    p.add_argument('--ckpt_dir',           default=None)
    p.add_argument('--run_name',           default='sm_4proc_event_c_layers4')
    p.add_argument('--stats_path',         default=None)
    p.add_argument('--stats_event_path',   default=None)
    p.add_argument('--out_dir',            default=None)
    p.add_argument('--holdout_start',      type=int, default=490000)
    p.add_argument('--n_total',            type=int, default=5000)
    p.add_argument('--gpu_id',             type=int, default=0)
    p.add_argument('--num_steps',          type=int, default=500)
    p.add_argument('--chunk_size',         type=int, default=200)
    p.add_argument('--npart',              type=int, default=500)
    p.add_argument('--proj_dim',           type=int, default=128)
    p.add_argument('--num_layers',         type=int, default=8)
    p.add_argument('--num_gen_layers',     type=int, default=4)
    p.add_argument('--use_truth_jet',      action='store_true', default=False)
    return p.parse_args()


args = _parse()

os.environ['CUDA_VISIBLE_DEVICES']  = str(args.gpu_id)
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
tf.random.set_seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────

sm_dir           = args.sm_dir
ckpt_dir         = args.ckpt_dir or os.path.join(sm_dir, 'checkpoints_sm_4proc')
stats_path       = args.stats_path or os.path.join(sm_dir, 'normalisation_stats_sm4proc.json')
stats_event_path = (args.stats_event_path
                    or os.path.join(sm_dir, 'normalisation_stats_event_c_sm4proc.json'))
ckpt_path        = os.path.join(ckpt_dir, args.run_name, 'pet_pp.weights.h5')
out_dir          = args.out_dir or os.path.join(ckpt_dir, args.run_name, 'infer_holdout_truth')
os.makedirs(out_dir, exist_ok=True)

data_path = os.path.join(sm_dir, f'{args.process}.hdf5')
out_file  = os.path.join(out_dir, f'{args.process}.npz')

print(f'[{args.process}] checkpoint: {ckpt_path}')
print(f'[{args.process}] data: {data_path}  holdout [{args.holdout_start}:{args.holdout_start+args.n_total}]')
print(f'[{args.process}] output: {out_file}')

if os.path.exists(out_file):
    print(f'[{args.process}] Output already exists, skipping.')
    sys.exit(0)

# ── Load normalisation stats ──────────────────────────────────────────────────

if not os.path.exists(stats_path):
    raise FileNotFoundError(f"Stats not found: {stats_path}. Run training first.")
if not os.path.exists(stats_event_path):
    raise FileNotFoundError(f"Event stats not found: {stats_event_path}. Run training first.")

with open(stats_path) as fh:
    stats = json.load(fh)
with open(stats_event_path) as fh:
    event_stats = json.load(fh)

cond_mean = np.array(stats['cond_mean'], dtype=np.float32)
cond_std  = np.array(stats['cond_std'],  dtype=np.float32)
jet_mean  = float(stats['jet_mean'][0])
jet_std   = float(stats['jet_std'][0])
part_mean = np.array(stats['part_mean'], dtype=np.float32)
part_std  = np.array(stats['part_std'],  dtype=np.float32)
event_mean = np.array(event_stats['event_mean'], dtype=np.float32)
event_std  = np.array(event_stats['event_std'],  dtype=np.float32)

print(f'[{args.process}] stats loaded  event_mean={event_mean}')

# ── Event feature helpers ─────────────────────────────────────────────────────

def _compute_event_raw_all7(pf_raw, part_raw, num_part):
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

# ── Load holdout data ─────────────────────────────────────────────────────────

with h5py.File(data_path, 'r') as f:
    n_avail  = f['particle_features'].shape[0]
    s = min(args.holdout_start, n_avail)
    e = min(s + args.n_total,   n_avail)
    if s >= e:
        raise RuntimeError(
            f"[{args.process}] holdout range [{s},{e}) is empty "
            f"(file has {n_avail} events).")
    pf_raw   = f['particle_features'][s:e].astype(np.float32)
    part_raw = f['parton_features'][s:e].astype(np.float32)

N = len(pf_raw)
print(f'[{args.process}] loaded {N} holdout events')

# Build conditioning
mass_col    = np.zeros((N, MAX_PARTONS, 1), dtype=np.float32)   # SM: always zero
part7       = np.concatenate([part_raw[:, :MAX_PARTONS, :], mass_col], axis=2)
cond_raw    = part7.reshape(N, MAX_PARTONS * PARTON_FEAT)
cond_norm   = (cond_raw - cond_mean) / cond_std
parton_mask = np.ones((N, MAX_PARTONS), dtype=np.float32)
cond        = np.concatenate([cond_norm, parton_mask], axis=1)  # (N, 32)

# Truth event features
raw7       = _compute_event_raw_all7(pf_raw, part_raw, args.npart)
event_feat = (raw7 - event_mean) / event_std

print(f'[{args.process}] event_feat sample[0]: {event_feat[0]}')

# Truth particle info
mask_truth = pf_raw[:, :args.npart, 6].astype(np.float32)
X_raw      = pf_raw[:, :args.npart, :6]
npart      = mask_truth.sum(axis=1, keepdims=True)
log_npart  = np.log(np.maximum(npart, 1.0))
jet_truth  = (log_npart - jet_mean) / jet_std

print(f'[{args.process}] mean truth npart={mask_truth.sum(axis=1).mean():.1f}')

# ── Load model ────────────────────────────────────────────────────────────────

_scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts)
from PET_pp_parton_vpar_bsm_event_c_layers4 import PET_pp_parton_vpar_bsm_event_c

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

model = PET_pp_parton_vpar_bsm_event_c(
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
print(f'[{args.process}] Loaded {ckpt_path}')

# ── Generate ──────────────────────────────────────────────────────────────────

nsplit       = max(1, N // args.chunk_size)
actual_chunk = N // nsplit
print(f'[{args.process}] generating {N} events  nsplit={nsplit} '
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
print(f'[{args.process}] generated in {dt/60:.2f} min  ({dt/N*1000:.0f} ms/event)')

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
    process      = np.bytes_(args.process),
    event_feat   = event_feat,
)
print(f'[{args.process}] saved → {out_file}')
print(f'[{args.process}] done in {(time.perf_counter()-t1)/60:.2f} min total')
