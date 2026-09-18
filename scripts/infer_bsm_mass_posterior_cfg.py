#!/usr/bin/env python3
"""
W' mass posterior via ELBO — E034/E035 CFG variant.

Adapted from infer_bsm_mass_posterior.py for:
  - NPZ input (generated events from E035 inference sweep) instead of HDF5
  - PET_pp_parton_vpar_bsm_event_c_stage1_cfg architecture (num_jet=8, num_gen_layers=2)
  - normalisation_stats_event_c_stage1_cfg.json (unified 8-dim stats)

The "observed" events are generated particles from a given CFG guidance scale,
rather than truth particles. Event conditioning (e_tf to ema_head) uses the
stage-1 predicted event features stored in jets_gen[:, 1:] from the NPZ.

Outputs posterior_mX{mx}_mY{my}.npz with same format as infer_bsm_mass_posterior.py,
readable by plot_a022_mass_posterior.py.

Usage:
  python3 infer_bsm_mass_posterior_cfg.py \\
    --npz_dir  .../infer_cfg_sweep/s1p5 \\
    --grid_dir /pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi \\
    --ckpt_dir /pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/checkpoints_bsm_grid \\
    --run_name bsm_grid_event_c_stage1_cfg \\
    --obs_m_X 250 --obs_m_Y 250 \\
    --n_obs 2000 --n_t 200 --chunk 500 \\
    --out_dir ./mass_inference_e035_s1p5
"""

import os, sys, re, json, glob, argparse, time
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_PARTONS    = 4
PARTON_FEAT    = 7
NUM_FEAT       = 6
NUM_JET        = 8   # stage-1: log_npart + 7 event features
MASS_NORM      = 600.0
NUM_EVENT_FEAT = 7


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--npz_dir',    required=True,
                   help='Directory with bsm_mX????_mY????_rank00_of01.npz files '
                        '(e.g. infer_cfg_sweep/s1p5)')
    p.add_argument('--grid_dir',   required=True,
                   help='Directory with signal_mX*.hdf5 files (for hypothesis grid)')
    p.add_argument('--ckpt_dir',   default=None,
                   help='Checkpoint root (default: grid_dir/checkpoints_bsm_grid)')
    p.add_argument('--run_name',   default='bsm_grid_event_c_stage1_cfg')
    p.add_argument('--stats_path', default=None,
                   help='Stats JSON (default: ckpt_dir/normalisation_stats_event_c_stage1_cfg.json)')
    p.add_argument('--obs_m_X',    type=float, required=True,
                   help='True m_X of the observed (generated) events')
    p.add_argument('--obs_m_Y',    type=float, required=True,
                   help='True m_Y of the observed (generated) events')
    p.add_argument('--n_obs',      type=int, default=2000,
                   help='Number of generated events to score (default 2000)')
    p.add_argument('--n_t',        type=int, default=200,
                   help='Monte Carlo timestep samples per event per hypothesis')
    p.add_argument('--chunk',      type=int, default=500,
                   help='Events per memory chunk during scoring')
    p.add_argument('--npart',      type=int, default=500)
    p.add_argument('--proj_dim',   type=int, default=128)
    p.add_argument('--num_layers', type=int, default=8)
    p.add_argument('--num_gen_layers', type=int, default=2)
    p.add_argument('--num_jet_mlp',    type=int, default=512)
    p.add_argument('--out_dir',    default='./mass_inference_cfg')
    p.add_argument('--gpu_id',     type=int, default=0)
    p.add_argument('--max_minutes', type=float, default=None,
                   help='Stop scoring after this many minutes and exit cleanly '
                        '(checkpoint is flushed; self-resubmitting scripts can then '
                        'fire before SLURM kills the job). Default: no limit.')
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
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
tf.random.set_seed(42)
print(f'Visible GPUs: {len(gpus)}')

sys.path.insert(0, _SCRIPT_DIR)

# ── Paths ─────────────────────────────────────────────────────────────────────

grid_dir   = args.grid_dir
ckpt_dir   = args.ckpt_dir or os.path.join(grid_dir, 'checkpoints_bsm_grid')
stats_path = (args.stats_path
              or os.path.join(ckpt_dir, 'normalisation_stats_event_c_stage1_cfg.json'))
ckpt_path  = os.path.join(ckpt_dir, args.run_name, 'pet_pp.weights.h5')
os.makedirs(args.out_dir, exist_ok=True)

# ── Load normalisation stats ──────────────────────────────────────────────────

with open(stats_path) as fh:
    stats = json.load(fh)

part_mean    = np.array(stats['part_mean'], dtype=np.float32)
part_std     = np.array(stats['part_std'],  dtype=np.float32)
cond_mean    = np.array(stats['cond_mean'], dtype=np.float32)
cond_std     = np.array(stats['cond_std'],  dtype=np.float32)
jet_mean_arr = np.array(stats['jet_mean'],  dtype=np.float32)   # (8,)
jet_std_arr  = np.array(stats['jet_std'],   dtype=np.float32)   # (8,)
jet_mean_s   = float(jet_mean_arr[0])    # log_npart mean
jet_std_s    = float(jet_std_arr[0])     # log_npart std
event_mean   = jet_mean_arr[1:]          # (7,) event feature means
event_std    = jet_std_arr[1:]           # (7,) event feature stds

print(f'Stats loaded | part_dims={len(part_mean)}  '
      f'jet_mean[0]={jet_mean_s:.3f}  cond_dims={len(cond_mean)}')

# ── Load generated events from NPZ ───────────────────────────────────────────

mx_obs = int(args.obs_m_X)
my_obs = int(args.obs_m_Y)
npz_path = os.path.join(args.npz_dir,
    f'bsm_mX{mx_obs:04d}_mY{my_obs:04d}_rank00_of01.npz')
if not os.path.exists(npz_path):
    raise FileNotFoundError(f'NPZ not found: {npz_path}')

d       = np.load(npz_path)
n_avail = len(d['parts_gen'])
n_use   = min(args.n_obs, n_avail)
print(f'NPZ: {os.path.basename(npz_path)}  available={n_avail}  using={n_use}')

parts_gen  = d['parts_gen'][:n_use].astype(np.float32)    # (N, 500, 6) physical
mask_gen   = d['mask_gen'][:n_use].astype(np.float32)     # (N, 500)
parton_npz = d['parton_feat'][:n_use].astype(np.float32)  # (N, 4, 7) — col 6 is mass
jets_gen   = d['jets_gen'][:n_use].astype(np.float32)     # (N, 8) normalized

N = n_use

# Normalize generated particles back to training space
mask_2d = mask_gen[:, :args.npart]   # (N, P)
x_raw   = parts_gen[:, :args.npart, :]
x_norm  = ((x_raw - part_mean) / part_std) * mask_2d[:, :, None]

# Jet: use stage-1 predictions (jets_gen is already normalized)
jet_norm = jets_gen    # (N, 8): col 0=log_npart_norm, cols 1-7=event features norm

# Event features: use stage-1 predicted event features (cols 1-7 of jets_gen)
# These were the actual conditioning used during stage-2 generation
event_feat = jets_gen[:, 1:]   # (N, 7) normalized

# Parton features for conditioning: strip the mass column (col 6) added during inference
part_raw = parton_npz[:, :MAX_PARTONS, :6]   # (N, 4, 6)

print(f'N={N}  mean npart_gen={mask_2d.sum(1).mean():.1f}  '
      f'jets_gen[0,0]={jet_norm[0,0]:.3f}  '
      f'event_feat sample: {event_feat[0].round(3)}')


# ── Build conditioning for a mass hypothesis ──────────────────────────────────

def build_cond(part_raw_obs, m_X_hyp, m_Y_hyp):
    N_loc    = len(part_raw_obs)
    mass_col = np.zeros((N_loc, MAX_PARTONS, 1), dtype=np.float32)
    mass_col[:, 2, 0] = m_X_hyp / MASS_NORM
    mass_col[:, 3, 0] = m_Y_hyp / MASS_NORM
    part7    = np.concatenate([part_raw_obs[:, :MAX_PARTONS, :], mass_col], axis=2)
    cond_raw = part7.reshape(N_loc, MAX_PARTONS * PARTON_FEAT)
    cond_n   = ((cond_raw - cond_mean) / cond_std).astype(np.float32)
    pmask    = np.ones((N_loc, MAX_PARTONS), dtype=np.float32)
    return np.concatenate([cond_n, pmask], axis=1)   # (N, 32)


# ── Load model ────────────────────────────────────────────────────────────────

from PET_pp_parton_vpar_bsm_event_c_stage1_cfg import PET_pp_parton_vpar_bsm_event_c_stage1

model = PET_pp_parton_vpar_bsm_event_c_stage1(
    num_feat=NUM_FEAT, num_jet=NUM_JET,
    max_partons=MAX_PARTONS, parton_feat=PARTON_FEAT,
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
print(f'Loaded weights: {ckpt_path}')


# ── Per-chunk ELBO scoring ────────────────────────────────────────────────────

def score_chunk(x_n, jet_n, mask2, cond, evf, N_t):
    """Score a chunk of events over N_t timestep samples.

    Returns:
        part_scores (n_chunk,) — mean v-pred MSE for particle term
        jet_scores  (n_chunk,) — mean v-pred MSE for jet/event term
    """
    nc     = len(x_n)
    mask3  = mask2[:, :, np.newaxis]

    x_tf   = tf.constant(x_n,   dtype=tf.float32)
    jet_tf = tf.constant(jet_n,  dtype=tf.float32)   # (nc, 8)
    m2_tf  = tf.constant(mask2,  dtype=tf.float32)
    m3_tf  = tf.constant(mask3,  dtype=tf.float32)
    c_tf   = tf.constant(cond,   dtype=tf.float32)
    e_tf   = tf.constant(evf,    dtype=tf.float32)    # (nc, 7) event features

    part_acc = np.zeros(nc, dtype=np.float64)
    jet_acc  = np.zeros(nc, dtype=np.float64)

    for _ in range(N_t):
        t = tf.random.uniform((nc, 1), dtype=tf.float32)
        _, alpha, sigma = model.get_logsnr_alpha_sigma(t)

        # ── particle term ─────────────────────────────────────────────────────
        eps_part = tf.random.normal((nc, args.npart, NUM_FEAT), dtype=tf.float32) * m3_tf
        x_t      = alpha[:, None] * x_tf + eps_part * sigma[:, None]
        x_t_m    = x_t * m3_tf

        v_body = model.ema_body([x_t_m, x_t_m[:, :, :2], m3_tf, t], training=False)
        # Head takes log_npart only (first dim of jet_tf)
        jet_head = jet_tf[:, :1]
        v_pred   = m3_tf * model.ema_head(
            [v_body, jet_head, m3_tf, t, c_tf, e_tf], training=False)

        v_true = (alpha[:, None] * eps_part - sigma[:, None] * x_tf) * m3_tf
        sq     = tf.reduce_sum(tf.square(v_true - v_pred), axis=[1, 2])
        n_p    = tf.reduce_sum(m2_tf, axis=1) + 1e-8
        part_acc += (sq / n_p).numpy()

        # ── jet term (full 8-dim: log_npart + event features) ────────────────
        eps_jet    = tf.random.normal((nc, NUM_JET), dtype=tf.float32)
        jet_t      = alpha * jet_tf + eps_jet * sigma
        v_pred_jet = model.ema_jet([jet_t, t, c_tf], training=False)
        v_true_jet = alpha * eps_jet - sigma * jet_tf
        jet_acc   += tf.reduce_mean(tf.square(v_true_jet - v_pred_jet), axis=1).numpy()

    return (part_acc / N_t).astype(np.float32), (jet_acc / N_t).astype(np.float32)


# ── Discover all mass hypothesis files ───────────────────────────────────────

def _parse_masses(fname):
    m = re.search(r'signal_mX(\d+)_mY(\d+)', os.path.basename(fname))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

hyp_files  = sorted(glob.glob(f'{grid_dir}/signal_mX*.hdf5'))
hyp_masses = [(mx, my) for f in hyp_files
              for mx, my in [_parse_masses(f)] if mx is not None]

if not hyp_masses:
    raise FileNotFoundError(f'No signal_mX*.hdf5 found in {grid_dir}')

print(f'\nScoring {N} generated events  '
      f'(true mass: m_X={mx_obs}, m_Y={my_obs})  '
      f'against {len(hyp_masses)} hypotheses  '
      f'({args.n_t} timestep samples, chunk={args.chunk})\n')


# ── Checkpoint / resume ───────────────────────────────────────────────────────
# After each hypothesis the partial results are flushed to a checkpoint file.
# On the next invocation (self-resubmit) the completed hypotheses are skipped.

ckpt_path_run = os.path.join(args.out_dir,
                             f'_ckpt_mX{mx_obs}_mY{my_obs}.npz')

results_part = np.zeros(len(hyp_masses), dtype=np.float32)
results_jet  = np.zeros(len(hyp_masses), dtype=np.float32)
hi_start     = 0

if os.path.exists(ckpt_path_run):
    ck = np.load(ckpt_path_run)
    n_done = int(ck['n_done'])
    if n_done > 0 and n_done <= len(hyp_masses):
        results_part[:n_done] = ck['results_part'][:n_done]
        results_jet[:n_done]  = ck['results_jet'][:n_done]
        hi_start = n_done
        print(f'Resumed from checkpoint: {n_done}/{len(hyp_masses)} hypotheses already done')


# ── Main scoring loop ─────────────────────────────────────────────────────────

HELDOUT = frozenset({(250, 250), (250, 300), (300, 250), (300, 300)})

t0 = time.perf_counter()
for hi, (m_X_h, m_Y_h) in enumerate(hyp_masses):
    if hi < hi_start:
        continue   # already scored in a previous run

    cond_h = build_cond(part_raw, m_X_h, m_Y_h)

    cp_all, cj_all = [], []
    for s in range(0, N, args.chunk):
        e_c = min(s + args.chunk, N)
        cp, cj = score_chunk(
            x_norm[s:e_c], jet_norm[s:e_c], mask_2d[s:e_c],
            cond_h[s:e_c], event_feat[s:e_c],
            args.n_t,
        )
        cp_all.append(cp); cj_all.append(cj)

    part_mean_ev = np.concatenate(cp_all).mean()
    jet_mean_ev  = np.concatenate(cj_all).mean()
    results_part[hi] = part_mean_ev
    results_jet[hi]  = jet_mean_ev
    total   = part_mean_ev + jet_mean_ev
    elapsed = time.perf_counter() - t0
    n_this  = hi + 1 - hi_start          # hypotheses scored this run
    eta     = elapsed / n_this * (len(hyp_masses) - hi - 1) if n_this > 0 else 0
    flag    = ' [HELDOUT]' if (m_X_h, m_Y_h) in HELDOUT else ''
    print(f'  [{hi+1:3d}/{len(hyp_masses)}] m_X={m_X_h:5.0f}  m_Y={m_Y_h:5.0f}'
          f'  part={part_mean_ev:.4f}  jet={jet_mean_ev:.4f}'
          f'  total={total:.4f}{flag}'
          f'  ETA {eta/60:.1f}min')

    # Flush checkpoint after every hypothesis
    np.savez_compressed(ckpt_path_run,
                        results_part=results_part,
                        results_jet=results_jet,
                        n_done=np.int32(hi + 1))

    # Exit cleanly before wall time so the self-resubmitting bash script can fire
    if args.max_minutes is not None:
        wall_elapsed = (time.perf_counter() - t0) / 60.0
        if wall_elapsed >= args.max_minutes:
            print(f'\nReached max_minutes={args.max_minutes:.0f} min after '
                  f'{hi+1}/{len(hyp_masses)} hypotheses — exiting cleanly for resubmit.')
            sys.exit(0)


# ── Report and save ───────────────────────────────────────────────────────────

total_scores = results_part + results_jet
best_idx     = np.argmin(total_scores)
best_m_X, best_m_Y = hyp_masses[best_idx]

print(f'\n{"─"*60}')
print(f'True mass:  m_X={mx_obs}  m_Y={my_obs}')
print(f'Best-fit:   m_X={best_m_X}  m_Y={best_m_Y}'
      f'  (log-lik={-total_scores[best_idx]:.4f})')
sorted_idx = np.argsort(total_scores)
print('Top 5:')
for rank, idx in enumerate(sorted_idx[:5], 1):
    mx_r, my_r = hyp_masses[idx]
    hit = mx_r == mx_obs and my_r == my_obs
    print(f'  #{rank}: m_X={mx_r:5.0f}  m_Y={my_r:5.0f}  '
          f'log-lik={-total_scores[idx]:.4f}{"  ← TRUE" if hit else ""}')
print(f'{"─"*60}')

out_path = os.path.join(args.out_dir, f'posterior_mX{mx_obs}_mY{my_obs}.npz')
np.savez_compressed(
    out_path,
    mass_x              = np.array([m[0] for m in hyp_masses], dtype=np.float32),
    mass_y              = np.array([m[1] for m in hyp_masses], dtype=np.float32),
    part_scores         = results_part,
    jet_scores          = results_jet,
    total_scores        = total_scores,
    mean_log_likelihood = -total_scores,
    obs_m_X             = np.float32(mx_obs),
    obs_m_Y             = np.float32(my_obs),
    n_obs               = np.int32(N),
    n_t                 = np.int32(args.n_t),
)
print(f'Saved → {out_path}')
# Remove checkpoint now that the final result is on disk
if os.path.exists(ckpt_path_run):
    os.remove(ckpt_path_run)
    print(f'Checkpoint removed: {ckpt_path_run}')
print(f'Total time: {(time.perf_counter()-t0)/60:.1f} min')
