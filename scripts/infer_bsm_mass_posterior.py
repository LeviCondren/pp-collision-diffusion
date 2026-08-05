#!/usr/bin/env python3
"""
W' mass inference via diffusion model ELBO scoring.

For a set of observed particle-level events, estimates log p(x | theta) at
each mass hypothesis theta = (m_X, m_Y) on the W' signal grid using:

    score(x, theta) = -E_t[ ||v_theta(x_t, t | cond(theta)) - v_true||^2 ]

where t ~ U[0,1] and the expectation is approximated by Monte Carlo over
--n_t timestep samples.  Lower score = higher likelihood.

The conditioning cond(theta) uses the truth parton kinematics from the
observed events (fixed) with the hypothesized mass injected into parton
slots 2 and 3.  Both the particle (body+head) and jet (multiplicity)
terms are scored.

Typical usage — recover mass from a heldout grid point:
  python infer_bsm_mass_posterior.py \\
    --grid_dir /pub/lcondren/wprime_signal_mpi \\
    --run_name bsm_grid_event_c_layers4_mpi \\
    --obs_m_X 250 --obs_m_Y 250 \\
    --n_obs 2000 --n_t 200 --chunk 500 \\
    --out_dir ./mass_inference

Outputs:
  {out_dir}/posterior_{obs_tag}.npz   — arrays: mass_x, mass_y,
    mean_log_likelihood, per_event_scores (part + jet)
"""

import os, sys, re, json, glob, argparse, time
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── constants (must match training) ──────────────────────────────────────────
MAX_PARTONS    = 4
PARTON_FEAT    = 7
NUM_COND       = MAX_PARTONS * PARTON_FEAT + MAX_PARTONS   # 32
NUM_FEAT       = 6
MASS_NORM      = 600.0
NUM_EVENT_FEAT = 7
R_CONE         = 1.0
HELDOUT_POINTS = frozenset({(250, 250), (250, 300), (300, 250), (300, 300)})


# ── argument parsing ──────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description='W\' mass posterior via diffusion ELBO scoring')

    # data paths
    p.add_argument('--grid_dir',   required=True,
                   help='Directory with signal_mX*.hdf5 files')
    p.add_argument('--ckpt_dir',   default=None,
                   help='Checkpoint root (default: grid_dir/checkpoints_bsm_grid)')
    p.add_argument('--run_name',   default='bsm_grid_event_c_layers4_mpi',
                   help='Run subdirectory inside ckpt_dir')
    p.add_argument('--stats_path', default=None)
    p.add_argument('--stats_event_path', default=None)

    # observed data — which mass point to treat as "data"
    p.add_argument('--obs_m_X',  type=float, default=None,
                   help='True m_X of the observed events (from signal grid)')
    p.add_argument('--obs_m_Y',  type=float, default=None,
                   help='True m_Y of the observed events')
    p.add_argument('--obs_file', default=None,
                   help='Explicit HDF5 file for observed events (overrides obs_m_X/Y)')

    # scoring settings
    p.add_argument('--val_start', type=int, default=80000,
                   help='Start of validation slice in HDF5 files')
    p.add_argument('--n_obs',    type=int, default=2000,
                   help='Number of observed events to score')
    p.add_argument('--n_t',      type=int, default=200,
                   help='Monte Carlo timestep samples per event per hypothesis')
    p.add_argument('--chunk',    type=int, default=500,
                   help='Events per memory chunk during scoring')
    p.add_argument('--npart',    type=int, default=500)

    # model arch (must match training)
    p.add_argument('--proj_dim',       type=int, default=128)
    p.add_argument('--num_layers',     type=int, default=8)
    p.add_argument('--num_gen_layers', type=int, default=4,
                   help='4 for E031 (layers4), 2 for E032 (stage1)')
    p.add_argument('--num_jet',        type=int, default=1,
                   help='Jet output dimension: 1 for layers4 (E031), 8 for stage1 (E032)')
    p.add_argument('--num_jet_mlp',    type=int, default=256,
                   help='Stage-1 jet MLP width — use 512 for E032')

    # output
    p.add_argument('--out_dir',  default='./mass_inference')
    p.add_argument('--gpu_id',   type=int, default=0)

    return p.parse_args()


args = _parse()
NUM_JET = args.num_jet   # 1 for layers4 (E031), 8 for stage-1 (E032)

# ── GPU / TF setup ────────────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES']  = str(args.gpu_id)
os.environ['TF_GPU_ALLOCATOR']      = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
tf.random.set_seed(42)
print(f'Visible GPUs: {len(gpus)}')

sys.path.insert(0, _SCRIPT_DIR)

# ── paths ─────────────────────────────────────────────────────────────────────
grid_dir         = args.grid_dir
ckpt_dir         = args.ckpt_dir or os.path.join(grid_dir, 'checkpoints_bsm_grid')
stats_path       = args.stats_path or os.path.join(grid_dir, 'normalisation_stats.json')
stats_event_path = (args.stats_event_path
                    or os.path.join(grid_dir, 'normalisation_stats_event_c.json'))
ckpt_path        = os.path.join(ckpt_dir, args.run_name, 'pet_pp.weights.h5')
os.makedirs(args.out_dir, exist_ok=True)


# ── load normalisation stats ──────────────────────────────────────────────────
with open(stats_path) as fh:
    stats = json.load(fh)

part_mean = np.array(stats['part_mean'], dtype=np.float32)
part_std  = np.array(stats['part_std'],  dtype=np.float32)
cond_mean = np.array(stats['cond_mean'], dtype=np.float32)
cond_std  = np.array(stats['cond_std'],  dtype=np.float32)

if NUM_JET == 1:
    # layers4 (E031): single scalar log_npart; event features from separate file
    jet_mean_s  = float(stats['jet_mean'][0])
    jet_std_s   = float(stats['jet_std'][0])
    with open(stats_event_path) as fh:
        event_stats = json.load(fh)
    event_mean = np.array(event_stats['event_mean'], dtype=np.float32)
    event_std  = np.array(event_stats['event_std'],  dtype=np.float32)
    print(f'Stats loaded | part_dims={len(part_mean)}  jet_mean={jet_mean_s:.3f}')
else:
    # stage-1 (E032): 8-dim jet = [log_npart, 7 event features] from unified stats
    jet_mean_arr = np.array(stats['jet_mean'], dtype=np.float32)   # (8,)
    jet_std_arr  = np.array(stats['jet_std'],  dtype=np.float32)   # (8,)
    jet_mean_s   = float(jet_mean_arr[0])   # log_npart mean (scalar)
    jet_std_s    = float(jet_std_arr[0])    # log_npart std  (scalar)
    event_mean   = jet_mean_arr[1:]         # (7,)  event feature means
    event_std    = jet_std_arr[1:]          # (7,)  event feature stds
    print(f'Stats loaded | part_dims={len(part_mean)}  jet_dims={NUM_JET}  '
          f'jet_mean[0]={jet_mean_s:.3f}')


# ── event feature helpers (identical to training) ─────────────────────────────

def _compute_event_raw_all7(pf_raw, part_raw):
    P     = args.npart
    valid = pf_raw[:, :P, 6].astype(bool)
    pT    = np.exp(np.clip(pf_raw[:, :P, 3], -10, 10)) * valid
    sp    = pf_raw[:, :P, 1]
    cp    = pf_raw[:, :P, 2]
    eta   = pf_raw[:, :P, 0]
    phi   = np.arctan2(sp, cp)

    MET_x   = (pT * cp).sum(1)
    MET_y   = (pT * sp).sum(1)
    met_mag = np.sqrt(MET_x**2 + MET_y**2)
    met_phi = np.arctan2(MET_y, MET_x)
    feats   = [np.log1p(met_mag), np.sin(met_phi), np.cos(met_phi)]

    eta_clip = np.clip(eta, -8, 8)
    for slot in [2, 3]:
        pze   = np.clip(part_raw[:, slot, 3], -1 + 1e-7, 1 - 1e-7)
        eta_p = 0.5 * np.log((1 + pze) / (1 - pze))
        phi_p = np.arctan2(part_raw[:, slot, 1], part_raw[:, slot, 2])
        deta  = eta - eta_p[:, None]
        dphi  = phi - phi_p[:, None]
        dphi  = (dphi + np.pi) % (2 * np.pi) - np.pi
        dR    = np.sqrt(deta**2 + dphi**2)
        in_c  = (dR < R_CONE) & valid
        wt    = pT * in_c
        pT_c  = wt.sum(1)
        E_c   = (wt * np.cosh(eta_clip)).sum(1)
        px_c  = (wt * cp).sum(1)
        py_c  = (wt * sp).sum(1)
        pz_c  = (wt * np.sinh(eta_clip)).sum(1)
        m2    = np.maximum(E_c**2 - px_c**2 - py_c**2 - pz_c**2, 0.0)
        feats.append(np.log1p(pT_c))
        feats.append(np.log1p(np.sqrt(m2)))

    return np.stack(feats, axis=1).astype(np.float32)


# ── load observed events ──────────────────────────────────────────────────────

def _parse_masses(fname):
    m = re.search(r'signal_mX(\d+)_mY(\d+)', os.path.basename(fname))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _find_signal_file(grid_dir, m_X, m_Y):
    files = sorted(glob.glob(f'{grid_dir}/signal_mX*.hdf5'))
    best, best_d = None, float('inf')
    for f in files:
        fx, fy = _parse_masses(f)
        if fx is None:
            continue
        d = (fx - m_X)**2 + (fy - m_Y)**2
        if d < best_d:
            best_d = d; best = f
    return best


if args.obs_file:
    obs_path = args.obs_file
    mx_obs, my_obs = _parse_masses(obs_path)
    mx_obs = mx_obs or 0; my_obs = my_obs or 0
elif args.obs_m_X is not None and args.obs_m_Y is not None:
    obs_path = _find_signal_file(grid_dir, args.obs_m_X, args.obs_m_Y)
    mx_obs, my_obs = _parse_masses(obs_path)
    print(f'Observed file: {os.path.basename(obs_path)}  '
          f'(true mass: m_X={mx_obs}, m_Y={my_obs})')
else:
    raise ValueError('Provide --obs_m_X / --obs_m_Y or --obs_file')

import h5py
with h5py.File(obs_path, 'r') as f:
    n_avail  = f['particle_features'].shape[0]
    s        = min(args.val_start, n_avail)
    e        = min(s + args.n_obs,  n_avail)
    if s >= e:
        raise RuntimeError(f'val_start={args.val_start} exceeds file size {n_avail}')
    pf_raw   = f['particle_features'][s:e].astype(np.float32)
    part_raw = f['parton_features'][s:e].astype(np.float32)

N = len(pf_raw)
print(f'Observed events: {N}  (file rows {s}–{e})')

# Normalise particle features
mask_2d  = pf_raw[:, :args.npart, 6].astype(np.float32)       # (N, P)
x_raw    = pf_raw[:, :args.npart, :NUM_FEAT]
x_norm   = ((x_raw - part_mean) / part_std).astype(np.float32) * mask_2d[:, :, None]

# Normalise jet and event features
npart_truth  = mask_2d.sum(1, keepdims=True)                   # (N, 1)
log_npart    = np.log(np.maximum(npart_truth, 1.0))
event_raw    = _compute_event_raw_all7(pf_raw, part_raw)
event_feat   = ((event_raw - event_mean) / event_std).astype(np.float32)

if NUM_JET == 1:
    jet_norm = ((log_npart - jet_mean_s) / jet_std_s).astype(np.float32)  # (N, 1)
else:
    # stage-1: 8-dim = [log_npart_norm, event_feat]
    log_npart_n = ((log_npart - jet_mean_s) / jet_std_s).astype(np.float32)
    jet_norm    = np.concatenate([log_npart_n, event_feat], axis=1)        # (N, 8)

print(f'mean npart={mask_2d.sum(1).mean():.1f}  '
      f'jet_norm sample: {jet_norm[0,0]:.3f}  '
      f'event_feat[0]: {event_feat[0]}')


# ── build conditioning vector for a given mass hypothesis ─────────────────────

def build_cond(part_raw_obs, m_X_hyp, m_Y_hyp):
    """Use truth parton kinematics from obs events with hypothesized mass."""
    N_loc    = len(part_raw_obs)
    mass_col = np.zeros((N_loc, MAX_PARTONS, 1), dtype=np.float32)
    mass_col[:, 2, 0] = m_X_hyp / MASS_NORM
    mass_col[:, 3, 0] = m_Y_hyp / MASS_NORM
    part7    = np.concatenate([part_raw_obs[:, :MAX_PARTONS, :], mass_col], axis=2)
    cond_raw = part7.reshape(N_loc, MAX_PARTONS * PARTON_FEAT)
    cond_n   = ((cond_raw - cond_mean) / cond_std).astype(np.float32)
    pmask    = np.ones((N_loc, MAX_PARTONS), dtype=np.float32)
    return np.concatenate([cond_n, pmask], axis=1)   # (N, 32)


# ── load model ────────────────────────────────────────────────────────────────

if args.num_gen_layers == 4:
    from PET_pp_parton_vpar_bsm_event_c_layers4 import PET_pp_parton_vpar_bsm_event_c
else:
    from PET_pp_parton_vpar_bsm_event_c_stage1 import PET_pp_parton_vpar_bsm_event_c_stage1 as PET_pp_parton_vpar_bsm_event_c

_model_kwargs = dict(
    num_feat=NUM_FEAT, num_jet=NUM_JET,
    max_partons=MAX_PARTONS, parton_feat=PARTON_FEAT,
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
if NUM_JET > 1:
    _model_kwargs['num_jet_mlp'] = args.num_jet_mlp
model = PET_pp_parton_vpar_bsm_event_c(**_model_kwargs)
model.load_weights(ckpt_path)
print(f'Loaded weights: {ckpt_path}')


# ── per-chunk ELBO scoring ────────────────────────────────────────────────────

def score_chunk(x_n, jet_n, mask2, cond, evf, N_t):
    """Score a chunk of events over N_t timestep samples.

    Returns:
        part_scores (n_chunk,) — mean MSE for particle term
        jet_scores  (n_chunk,) — mean MSE for jet (multiplicity) term
    """
    nc      = len(x_n)
    mask3   = mask2[:, :, np.newaxis]

    x_tf   = tf.constant(x_n,   dtype=tf.float32)
    jet_tf = tf.constant(jet_n,  dtype=tf.float32)
    m2_tf  = tf.constant(mask2,  dtype=tf.float32)
    m3_tf  = tf.constant(mask3,  dtype=tf.float32)
    c_tf   = tf.constant(cond,   dtype=tf.float32)
    e_tf   = tf.constant(evf,    dtype=tf.float32)

    part_acc = np.zeros(nc, dtype=np.float64)
    jet_acc  = np.zeros(nc, dtype=np.float64)

    for _ in range(N_t):
        t = tf.random.uniform((nc, 1), dtype=tf.float32)
        _, alpha, sigma = model.get_logsnr_alpha_sigma(t)

        # ── particle term ──
        eps_part = tf.random.normal((nc, args.npart, NUM_FEAT), dtype=tf.float32) * m3_tf
        x_t      = alpha[:, None] * x_tf + eps_part * sigma[:, None]
        x_t_m    = x_t * m3_tf

        v_body = model.ema_body([x_t_m, x_t_m[:, :, :2], m3_tf, t], training=False)
        # stage-1 head takes only log_npart (1-dim); layers4 jet_tf is already 1-dim
        jet_head = jet_tf[:, :1] if NUM_JET > 1 else jet_tf
        v_pred = m3_tf * model.ema_head([v_body, jet_head, m3_tf, t, c_tf, e_tf], training=False)

        v_true = (alpha[:, None] * eps_part - sigma[:, None] * x_tf) * m3_tf
        sq     = tf.reduce_sum(tf.square(v_true - v_pred), axis=[1, 2])
        n_p    = tf.reduce_sum(m2_tf, axis=1) + 1e-8
        part_acc += (sq / n_p).numpy()

        # ── jet (multiplicity) term ──
        eps_jet    = tf.random.normal((nc, NUM_JET), dtype=tf.float32)
        jet_t      = alpha * jet_tf + eps_jet * sigma
        v_pred_jet = model.ema_jet([jet_t, t, c_tf], training=False)
        v_true_jet = alpha * eps_jet - sigma * jet_tf
        jet_acc += tf.reduce_mean(tf.square(v_true_jet - v_pred_jet), axis=1).numpy()

    return (part_acc / N_t).astype(np.float32), (jet_acc / N_t).astype(np.float32)


# ── discover all mass hypothesis files ───────────────────────────────────────

hyp_files = sorted(glob.glob(f'{grid_dir}/signal_mX*.hdf5'))
if not hyp_files:
    raise FileNotFoundError(f'No signal files in {grid_dir}')

hyp_masses = []
for f in hyp_files:
    mx, my = _parse_masses(f)
    if mx is not None:
        hyp_masses.append((mx, my))

print(f'\nScoring {N} observed events against {len(hyp_masses)} mass hypotheses '
      f'({args.n_t} timestep samples each)\n')


# ── main scoring loop ─────────────────────────────────────────────────────────

results_part = np.zeros(len(hyp_masses), dtype=np.float32)
results_jet  = np.zeros(len(hyp_masses), dtype=np.float32)

t0 = time.perf_counter()

for hi, (m_X_h, m_Y_h) in enumerate(hyp_masses):
    cond_h = build_cond(part_raw, m_X_h, m_Y_h)   # (N, 32) for this hypothesis

    chunk_part = []
    chunk_jet  = []
    for s in range(0, N, args.chunk):
        e_c = min(s + args.chunk, N)
        cp, cj = score_chunk(
            x_norm[s:e_c], jet_norm[s:e_c], mask_2d[s:e_c],
            cond_h[s:e_c], event_feat[s:e_c],
            args.n_t,
        )
        chunk_part.append(cp)
        chunk_jet.append(cj)

    part_mean_ev = np.concatenate(chunk_part).mean()
    jet_mean_ev  = np.concatenate(chunk_jet).mean()
    results_part[hi] = part_mean_ev
    results_jet[hi]  = jet_mean_ev
    total = part_mean_ev + jet_mean_ev

    elapsed = time.perf_counter() - t0
    eta     = elapsed / (hi + 1) * (len(hyp_masses) - hi - 1)
    heldout = (m_X_h, m_Y_h) in HELDOUT_POINTS
    flag    = ' [HELDOUT]' if heldout else ''
    print(f'  [{hi+1:3d}/{len(hyp_masses)}] m_X={m_X_h:5.0f}  m_Y={m_Y_h:5.0f}'
          f'  part={part_mean_ev:.4f}  jet={jet_mean_ev:.4f}'
          f'  total={total:.4f}{flag}'
          f'  ETA {eta/60:.1f}min')


# ── find best-fit mass and report ─────────────────────────────────────────────

total_scores  = results_part + results_jet
best_idx      = np.argmin(total_scores)
best_m_X, best_m_Y = hyp_masses[best_idx]
best_ll       = -total_scores[best_idx]

print(f'\n{"─"*60}')
print(f'True observed mass:  m_X={mx_obs}  m_Y={my_obs}')
print(f'Best-fit hypothesis: m_X={best_m_X}  m_Y={best_m_Y}  '
      f'(log-lik estimate: {best_ll:.4f})')
print(f'{"─"*60}')

# Top-5
sorted_idx = np.argsort(total_scores)
print('\nTop 5 hypotheses:')
for rank, idx in enumerate(sorted_idx[:5], 1):
    mx_r, my_r = hyp_masses[idx]
    print(f'  #{rank}: m_X={mx_r:5.0f}  m_Y={my_r:5.0f}  '
          f'log-lik={-total_scores[idx]:.4f}')

# ── save results ──────────────────────────────────────────────────────────────

obs_tag = f'mX{mx_obs}_mY{my_obs}' if mx_obs else 'custom'
out_path = os.path.join(args.out_dir, f'posterior_{obs_tag}.npz')

np.savez_compressed(
    out_path,
    mass_x              = np.array([m[0] for m in hyp_masses], dtype=np.float32),
    mass_y              = np.array([m[1] for m in hyp_masses], dtype=np.float32),
    part_scores         = results_part,
    jet_scores          = results_jet,
    total_scores        = total_scores,
    mean_log_likelihood = -total_scores,
    obs_m_X             = np.float32(mx_obs or 0),
    obs_m_Y             = np.float32(my_obs or 0),
    n_obs               = np.int32(N),
    n_t                 = np.int32(args.n_t),
)
print(f'\nSaved → {out_path}')
print(f'Total time: {(time.perf_counter()-t0)/60:.1f} min')
