#!/usr/bin/env python3
"""BSM grid inference — E038: three-pass composable-diffusion CFG for cone mass conditioning.

Copied from infer_asym.py (E035/E036/E037) and modified for E038:
  - Uses model.generate_composable_cfg() / DDPMSamplerComposableCFG instead of
    model.generate_asym_cfg() — the composable-diffusion formula that never
    references v_full, so it cannot double-count it the way generate_asym_cfg's
    v_full coefficient (gs_x+gs_y) does:
      v_no_x:  cone_mass_X zeroed (index 4 of 7-dim event vector)  — "Y-only" branch
      v_no_y:  cone_mass_Y zeroed (index 6 of 7-dim event vector)  — "X-only" branch
      v_null:  both cone masses zeroed
      v_guided = v_null + gs_x*(v_no_y - v_null) + gs_y*(v_no_x - v_null)
  - Everything else (data loading, stats, event-feature reconstruction, MET fix,
    output format) is unchanged from infer_asym.py.
  - Intended for use against an E038 checkpoint (raised cfg_drop_x_prob/y=0.35,
    giving v_null far more training exposure than E036's 1%) — see EXPERIMENTS.md.

Do NOT modify infer_asym.py (E035/E036/E037 canonical, generate_asym_cfg-based).

Changelog:
  E038 (2026-09-15): composable-diffusion CFG inference, paired with the E038
    training run (raised per-cone dropout).
"""

import os, sys, re, json, argparse, glob, time
import numpy as np

_GRID_DIR_DEFAULT = '/pscratch/sd/l/lcondren/MCsim/wprime_signal'

MAX_PARTONS    = 4
PARTON_FEAT    = 7
NUM_COND       = MAX_PARTONS * PARTON_FEAT + MAX_PARTONS  # 32
MASS_NORM      = 600.0
NUM_EVENT_FEAT = 7
R_CONE         = 1.0


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--m_X',               type=float, required=True)
    p.add_argument('--m_Y',               type=float, required=True)
    p.add_argument('--grid_dir',          default=_GRID_DIR_DEFAULT)
    p.add_argument('--ckpt_dir',          default=None)
    p.add_argument('--run_name',          default='bsm_grid_event_c_stage1_cfg_asym_e038')
    p.add_argument('--stats_path',        default=None,
                   help='8-dim stats JSON (default: {ckpt_dir}/normalisation_stats_event_c_stage1_cfg_asym.json). '
                        'Pass E034 stats path to reuse identical stats.')
    p.add_argument('--out_dir',           default=None)
    p.add_argument('--rank',              type=int,
                   default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    p.add_argument('--world_size',        type=int,
                   default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)))
    p.add_argument('--gpu_id',            type=int, default=None)
    p.add_argument('--num_steps',         type=int, default=500)
    p.add_argument('--chunk_size',        type=int, default=200)
    p.add_argument('--n_total',           type=int, default=10000)
    p.add_argument('--npart',             type=int, default=500)
    p.add_argument('--proj_dim',          type=int, default=128)
    p.add_argument('--num_layers',        type=int, default=8)
    p.add_argument('--num_gen_layers',    type=int, default=2)
    p.add_argument('--use_truth_jet',     action='store_true', default=False)
    p.add_argument('--use_true_event',    action='store_true', default=False)
    p.add_argument('--num_jet_steps',     type=int, default=None)
    p.add_argument('--guidance_scale_X',  type=float, default=1.5,
                   help='Composable-CFG guidance scale for cone_mass_X. '
                        '0=v_null, 1=additive approx of v_full, >1=extrapolation. (default 1.5)')
    p.add_argument('--guidance_scale_Y',  type=float, default=1.5,
                   help='Composable-CFG guidance scale for cone_mass_Y. (default 1.5)')
    p.add_argument('--use_true_cone',     action='store_true', default=False)
    p.add_argument('--num_jet_mlp',       type=int, default=512)
    p.add_argument('--stage1_only',       action='store_true', default=False)
    p.add_argument('--fix_met',           action='store_true', default=False)
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
ckpt_dir   = args.ckpt_dir or os.path.join(grid_dir, 'checkpoints_bsm_grid')
stats_path = (args.stats_path
              or os.path.join(ckpt_dir,
                              'normalisation_stats_event_c_stage1_cfg_asym.json'))
ckpt_path  = os.path.join(ckpt_dir, args.run_name, 'pet_pp.weights.h5')
out_dir    = args.out_dir or os.path.join(ckpt_dir, args.run_name, 'infer')
os.makedirs(out_dir, exist_ok=True)

tag      = f'mX{args.m_X:04.0f}_mY{args.m_Y:04.0f}'
out_file = os.path.join(out_dir, f'bsm_{tag}_rank{args.rank:02d}_of{args.world_size:02d}.npz')

print(f'[rank {args.rank}] target: m_X={args.m_X}  m_Y={args.m_Y}')
print(f'[rank {args.rank}] checkpoint: {ckpt_path}')
print(f'[rank {args.rank}] output: {out_file}')
print(f'[rank {args.rank}] guidance_scale_X={args.guidance_scale_X}  '
      f'guidance_scale_Y={args.guidance_scale_Y}')

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
    raise FileNotFoundError(
        f"Stats not found: {stats_path}\n"
        f"Run bsm_grid_train_event_c_stage1_cfg_asym.py first, or pass "
        f"--stats_path to reuse E034 stats (identical content).")

with open(stats_path) as fh:
    stats = json.load(fh)

cond_mean_raw = np.array(stats['cond_mean'], dtype=np.float32)
expected_cond = MAX_PARTONS * PARTON_FEAT
if len(cond_mean_raw) != expected_cond:
    raise ValueError(f"Stats file has {len(cond_mean_raw)} cond dims, expected {expected_cond}.")

cond_mean  = cond_mean_raw
cond_std   = np.array(stats['cond_std'],  dtype=np.float32)
jet_mean_a = np.array(stats['jet_mean'],  dtype=np.float32)
jet_std_a  = np.array(stats['jet_std'],   dtype=np.float32)
jet_mean   = float(jet_mean_a[0])
jet_std    = float(jet_std_a[0])
ev_mean    = jet_mean_a[1:]
ev_std     = jet_std_a[1:]
part_mean  = np.array(stats['part_mean'], dtype=np.float32)
part_std   = np.array(stats['part_std'],  dtype=np.float32)

print(f'[rank {args.rank}] stats loaded: cond_dims={len(cond_mean)}  '
      f'part_dims={len(part_mean)}  jet_mean={jet_mean:.3f}')
print(f'[rank {args.rank}] ev_mean={ev_mean}  ev_std={ev_std}')

# ── Event feature helpers (unchanged from E034) ───────────────────────────────

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


def _assemble_event_feat(raw7):
    return raw7


def fix_met(parts_phys, mask_gen, jets_gen, ev_mean, ev_std):
    """Post-process generated particle cloud to match stage-1 predicted MET (unchanged from E034)."""
    parts_out = parts_phys.copy()
    mask = mask_gen.astype(bool)

    log1p_met_pred = jets_gen[:, 1] * float(ev_std[0]) + float(ev_mean[0])
    met_mag_pred   = np.expm1(np.clip(log1p_met_pred, -10, 20))
    sp_pred = jets_gen[:, 2] * float(ev_std[1]) + float(ev_mean[1])
    cp_pred = jets_gen[:, 3] * float(ev_std[2]) + float(ev_mean[2])
    rn = np.sqrt(sp_pred**2 + cp_pred**2 + 1e-8)
    sp_pred = sp_pred / rn;  cp_pred = cp_pred / rn

    met_x_tgt = met_mag_pred * cp_pred
    met_y_tgt = met_mag_pred * sp_pred

    pT = np.exp(np.clip(parts_phys[:, :, 3], -10, 10)) * mask
    met_x_cur = (pT * parts_phys[:, :, 2]).sum(1)
    met_y_cur = (pT * parts_phys[:, :, 1]).sum(1)

    n_valid = mask.sum(1).clip(min=1).astype(float)
    d_px = ((met_x_tgt - met_x_cur) / n_valid)[:, None] * mask
    d_py = ((met_y_tgt - met_y_cur) / n_valid)[:, None] * mask

    px_new = pT * parts_phys[:, :, 2] + d_px
    py_new = pT * parts_phys[:, :, 1] + d_py
    pT_new = np.sqrt(px_new**2 + py_new**2)

    safe = (pT_new > 1e-2) & mask
    pT_safe = np.where(safe, pT_new, 1e-6)
    parts_out[:, :, 3] = np.where(safe, np.log(pT_safe),  parts_phys[:, :, 3])
    parts_out[:, :, 2] = np.where(safe, px_new / pT_safe, parts_phys[:, :, 2])
    parts_out[:, :, 1] = np.where(safe, py_new / pT_safe, parts_phys[:, :, 1])

    n_skipped = (~safe & mask).sum()
    if n_skipped:
        print(f'[fix_met] {n_skipped} particles skipped (corrected pT < 10 MeV)')
    return parts_out


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

mass_col = np.zeros((N, MAX_PARTONS, 1), dtype=np.float32)
mass_col[:, 2, 0] = file_mx / MASS_NORM
mass_col[:, 3, 0] = file_my / MASS_NORM
part7    = np.concatenate([part_raw[:, :MAX_PARTONS, :], mass_col], axis=2)
cond_raw  = part7.reshape(N, MAX_PARTONS * PARTON_FEAT)
cond_norm = (cond_raw - cond_mean) / cond_std
parton_mask = np.ones((N, MAX_PARTONS), dtype=np.float32)
cond = np.concatenate([cond_norm, parton_mask], axis=1)

raw7         = _compute_event_raw_all7(pf_raw, part_raw, args.npart)
event_raw    = _assemble_event_feat(raw7)
event_feat   = (event_raw - ev_mean) / ev_std

print(f'[rank {args.rank}] event_feat sample[0]: {event_feat[0]}')
print(f'[rank {args.rank}] event_feat mean: {event_feat.mean(0)}  std: {event_feat.std(0)}')

mask_truth = pf_raw[:, :args.npart, 6].astype(np.float32)
X_raw      = pf_raw[:, :args.npart, :6]

npart     = mask_truth.sum(axis=1, keepdims=True)
log_npart = np.log(np.maximum(npart, 1.0))
jet_s0    = (log_npart - jet_mean) / jet_std
# BUG FIX (inherited from infer_asym.py, not yet fixed there): jet_truth must be
# the full (N,8) [log_npart, 7 event feats] vector — generate_composable_cfg's
# `jets` param is indexed as jet[:,0:1] and jet[:,1:], so a (N,1)-only array
# (the original form of this line) silently made jet[:,1:] an empty (N,0)
# slice, i.e. --use_truth_jet never actually supplied truth event features.
jet_truth = np.concatenate([jet_s0, event_feat], axis=1)   # (N, 8)

print(f'[rank {args.rank}] mean truth npart={mask_truth.sum(axis=1).mean():.1f}')

# ── Load model ────────────────────────────────────────────────────────────────

_scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts)
from PET_pp_parton_vpar_bsm_event_c_stage1_cfg_asym import PET_pp_parton_vpar_bsm_event_c_stage1

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

model = PET_pp_parton_vpar_bsm_event_c_stage1(
    num_feat=6, num_jet=8,
    max_partons=MAX_PARTONS,
    parton_feat=PARTON_FEAT,
    num_event_feat=NUM_EVENT_FEAT,
    num_part=args.npart,
    projection_dim=args.proj_dim,
    num_jet_mlp=args.num_jet_mlp,
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

if args.stage1_only:
    from tqdm import tqdm as _tqdm
    jsteps   = args.num_jet_steps or 512
    splits   = np.array_split(cond, nsplit)
    jet_info = []
    for split in _tqdm(splits, desc=f'[rank {args.rank}] stage-1'):
        jet = model.DDPMSampler(split, model.ema_jet,
                                data_shape=[split.shape[0], 8],
                                w=0.0, num_steps=jsteps,
                                const_shape=[-1, 1]).numpy()
        jet_info.append(jet)
    jets_gen = np.concatenate(jet_info)
    dt = time.perf_counter() - t1
    print(f'[rank {args.rank}] stage-1 done in {dt/60:.2f} min')
    np.savez_compressed(out_file,
        parton_feat      = part7,
        mass_x           = np.float32(file_mx),
        mass_y           = np.float32(file_my),
        event_feat_truth = event_feat,
        jets_gen         = jets_gen,
    )
else:
    print(f'[rank {args.rank}] composable CFG  guidance_scale_X={args.guidance_scale_X}  '
          f'guidance_scale_Y={args.guidance_scale_Y}  '
          f'use_true_cone={args.use_true_cone}  use_true_event={args.use_true_event}')

    if args.use_true_event:
        # Standard (non-CFG) generation with truth event features
        parts_gen, jets_gen = model.generate(
            cond=cond,
            jet_mean=jet_mean,
            jet_std=jet_std,
            event_feat=event_feat,
            nsplit=nsplit,
            num_steps=args.num_steps,
            jets=jets_in,
            use_tqdm=True,
            num_jet_steps=args.num_jet_steps,
            use_true_event=True,
        )
    else:
        # Three-pass composable-diffusion CFG generation (E038)
        parts_gen, jets_gen = model.generate_composable_cfg(
            cond=cond,
            jet_mean=jet_mean,
            jet_std=jet_std,
            nsplit=nsplit,
            jets=jets_in,
            use_tqdm=True,
            num_steps=args.num_steps,
            num_jet_steps=args.num_jet_steps,
            guidance_scale_x=args.guidance_scale_X,
            guidance_scale_y=args.guidance_scale_Y,
            use_true_cone=args.use_true_cone,
            event_feat=event_feat if args.use_true_cone else None,
        )
    dt = time.perf_counter() - t1
    print(f'[rank {args.rank}] generated in {dt/60:.2f} min  ({dt/N*1000:.0f} ms/event)')

    log_npart_gen = jets_gen[:, 0] * jet_std + jet_mean
    npart_gen     = np.clip(np.round(np.exp(log_npart_gen)).astype(int), 1, args.npart)
    mask_gen      = (np.arange(args.npart)[None, :] < npart_gen[:, None]).astype(np.float32)
    parts_phys    = (parts_gen * part_std + part_mean) * mask_gen[:, :, None]
    parts_phys[:, :, 5] = np.round(parts_phys[:, :, 5])

    if args.fix_met:
        print(f'[rank {args.rank}] applying MET correction ...')
        parts_phys = fix_met(parts_phys, mask_gen, jets_gen, ev_mean, ev_std)

    np.savez_compressed(out_file,
        parts_truth         = X_raw,
        parts_gen           = parts_phys,
        mask                = mask_truth,
        mask_gen            = mask_gen,
        parton_feat         = part7,
        mass_x              = np.float32(file_mx),
        mass_y              = np.float32(file_my),
        event_feat_truth    = event_feat,
        jets_gen            = jets_gen,
        guidance_scale_X    = np.float32(args.guidance_scale_X),
        guidance_scale_Y    = np.float32(args.guidance_scale_Y),
    )
print(f'[rank {args.rank}] saved → {out_file}')
print(f'[rank {args.rank}] done in {(time.perf_counter()-t1)/60:.2f} min total')
