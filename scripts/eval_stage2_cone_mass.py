#!/usr/bin/env python3
"""
eval_stage2_cone_mass.py — Stage-2 cone mass diagnostic using truth conditioning.

Runs stage-2 particle generation conditioned on TRUTH log_npart and TRUTH event
features (no stage-1 sampling). Computes particle-level cone masses from the
generated particle cloud and plots generated vs truth for the 4 holdout mass points.

This isolates stage-2 generation quality from stage-1 prediction quality.

Needs a GPU — stage-2 at 500 steps is slow on CPU.

Usage:
  python3 eval_stage2_cone_mass.py \\
      --run_name bsm_grid_event_c_stage1_cfg \\
      --arch     stage1_cfg \\
      --ckpt_dir /pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid \\
      --grid_dir /pub/lcondren/wprime_signal_mpi \\
      --plot_out /pub/lcondren/wprime_signal_mpi/figures/stage2_cone_mass_e034.png
"""

import os, sys, glob, re, json, argparse, importlib, time
import numpy as np
from scipy.stats import wasserstein_distance

MAX_PARTONS = 4
PARTON_FEAT = 7
MASS_NORM   = 600.0
R_CONE      = 1.0
NUM_PART    = 500

HOLDOUT_MASS_POINTS = [(250, 250), (250, 300), (300, 250), (300, 300)]

STATS_NAME = {
    'stage1':     'normalisation_stats_event_c_stage1.json',
    'stage1_cfg': 'normalisation_stats_event_c_stage1_cfg.json',
}
ARCH_MODULE = {
    'stage1':     'PET_pp_parton_vpar_bsm_event_c_stage1',
    'stage1_cfg': 'PET_pp_parton_vpar_bsm_event_c_stage1_cfg',
}

# ── Args ──────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name',       default='bsm_grid_event_c_stage1_cfg')
    p.add_argument('--arch',           default=None,
                   help='stage1 or stage1_cfg (auto-detected from run_name)')
    p.add_argument('--ckpt_dir',       default='/pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid')
    p.add_argument('--grid_dir',       default='/pub/lcondren/wprime_signal_mpi')
    p.add_argument('--stats_path',     default=None)
    p.add_argument('--n_events',       type=int, default=1000)
    p.add_argument('--num_steps',      type=int, default=500,
                   help='DDPM steps for stage-2 (default 500, matching training).')
    p.add_argument('--chunk_size',     type=int, default=50,
                   help='Events per chunk for stage-2 generation.')
    p.add_argument('--proj_dim',       type=int, default=128)
    p.add_argument('--num_layers',     type=int, default=8)
    p.add_argument('--num_gen_layers', type=int, default=2)
    p.add_argument('--num_jet_mlp',    type=int, default=512)
    p.add_argument('--gpu',            type=int, default=0)
    p.add_argument('--plot_out',       default=None,
                   help='Output PNG path. Default: stage2_cone_mass_<run_name>.png')
    p.add_argument('--linear',         action='store_true',
                   help='Plot cone masses in linear GeV (expm1 of log1p values).')
    p.add_argument('--npz_out',        default=None,
                   help='Save generated/truth arrays to this .npz for fast replotting.')
    p.add_argument('--from_npz',       default=None,
                   help='Skip generation; load arrays from this .npz and replot.')
    return p.parse_args()

args = _parse()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['TF_GPU_ALLOCATOR']     = 'cuda_malloc_async'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print(f'[eval] visible GPUs: {len(gpus)}')

import h5py

# ── Arch / stats ──────────────────────────────────────────────────────────────

arch = args.arch or ('stage1_cfg' if 'cfg' in args.run_name else 'stage1')
print(f'[eval] arch={arch}  run={args.run_name}')

stats_path = args.stats_path or os.path.join(args.ckpt_dir, STATS_NAME[arch])
if not os.path.exists(stats_path):
    sys.exit(f'Stats not found: {stats_path}')

with open(stats_path) as fh:
    stats = json.load(fh)

cond_mean  = np.array(stats['cond_mean'], dtype=np.float32)
cond_std   = np.array(stats['cond_std'],  dtype=np.float32)
jet_mean_a = np.array(stats['jet_mean'],  dtype=np.float32)
jet_std_a  = np.array(stats['jet_std'],   dtype=np.float32)
jet_mean   = float(jet_mean_a[0])   # log_npart mean
jet_std    = float(jet_std_a[0])    # log_npart std
ev_mean    = jet_mean_a[1:]         # (7,) event feature means
ev_std     = jet_std_a[1:]          # (7,) event feature stds
part_mean  = np.array(stats['part_mean'], dtype=np.float32)
part_std   = np.array(stats['part_std'],  dtype=np.float32)

print(f'[eval] ev_mean[4,6] (cone_mass_X/Y): {ev_mean[4]:.3f}, {ev_mean[6]:.3f}')

# ── Event feature computation ─────────────────────────────────────────────────

def _compute_event_raw_all7(pf_raw, part_raw, num_part=NUM_PART):
    valid = pf_raw[:, :num_part, 6].astype(bool)
    pT    = np.exp(np.clip(pf_raw[:, :num_part, 3], -10, 10)) * valid
    sp    = pf_raw[:, :num_part, 1]
    cp    = pf_raw[:, :num_part, 2]
    eta   = pf_raw[:, :num_part, 0]
    phi   = np.arctan2(sp, cp)
    MET_x = (pT * cp).sum(1)
    MET_y = (pT * sp).sum(1)
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

def _cone_masses_from_gen(parts_phys, mask, part_raw):
    """Compute log1p(cone_mass) for X and Y cones from generated particle cloud.

    parts_phys : (N, max_part, 6) denormalized — [eta, sin_phi, cos_phi, log_pT, ...]
    mask       : (N, max_part) bool
    part_raw   : (N, n_parton, 6) raw parton features from HDF5
    Returns    : (cm_x, cm_y) each shape (N,), in log1p(GeV)
    """
    eta  = parts_phys[:, :, 0]
    sp   = parts_phys[:, :, 1]
    cp   = parts_phys[:, :, 2]
    logpT = parts_phys[:, :, 3]
    phi  = np.arctan2(sp, cp)
    pT   = np.exp(np.clip(logpT, -10, 10)) * mask
    eta_clip = np.clip(eta, -8, 8)

    results = []
    for slot in [2, 3]:
        pze   = np.clip(part_raw[:, slot, 3], -1 + 1e-7, 1 - 1e-7)
        eta_p = 0.5 * np.log((1 + pze) / (1 - pze))
        phi_p = np.arctan2(part_raw[:, slot, 1], part_raw[:, slot, 2])
        deta  = eta - eta_p[:, None]
        dphi  = phi - phi_p[:, None]
        dphi  = (dphi + np.pi) % (2 * np.pi) - np.pi
        dR    = np.sqrt(deta**2 + dphi**2)
        in_c  = (dR < R_CONE) & mask.astype(bool)
        wt    = pT * in_c
        E_c   = (wt * np.cosh(eta_clip)).sum(1)
        px_c  = (wt * cp).sum(1)
        py_c  = (wt * sp).sum(1)
        pz_c  = (wt * np.sinh(eta_clip)).sum(1)
        m2    = np.maximum(E_c**2 - px_c**2 - py_c**2 - pz_c**2, 0.0)
        results.append(np.log1p(np.sqrt(m2)))
    return results[0], results[1]

# ── HDF5 file finder ──────────────────────────────────────────────────────────

def _find_file(grid_dir, mx, my):
    pattern = os.path.join(grid_dir, f'signal_mX{mx:04d}_mY{my:04d}.hdf5')
    if os.path.exists(pattern):
        return pattern
    candidates = glob.glob(os.path.join(grid_dir, 'signal_mX*.hdf5'))
    def _dist(f):
        m = re.search(r'mX(\d+)_mY(\d+)', os.path.basename(f))
        if not m: return float('inf')
        return (int(m.group(1)) - mx)**2 + (int(m.group(2)) - my)**2
    return min(candidates, key=_dist, default=None)

# ── Load model ────────────────────────────────────────────────────────────────

ckpt_path = os.path.join(args.ckpt_dir, args.run_name, 'pet_pp.weights.h5')
if not os.path.exists(ckpt_path):
    sys.exit(f'Checkpoint not found: {ckpt_path}')

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts_dir)
mod = importlib.import_module(ARCH_MODULE[arch])
ModelCls = getattr(mod, 'PET_pp_parton_vpar_bsm_event_c_stage1')

model = ModelCls(
    num_feat=6, num_jet=8,
    max_partons=MAX_PARTONS, parton_feat=PARTON_FEAT, num_event_feat=7,
    num_part=NUM_PART, projection_dim=args.proj_dim, num_jet_mlp=args.num_jet_mlp,
    local=True, K=5, num_layers=args.num_layers, num_gen_layers=args.num_gen_layers,
    drop_probability=0.0, simple=False, layer_scale=True, talking_head=False,
    mode='generator',
)
model.load_weights(ckpt_path)
print(f'[eval] loaded {ckpt_path}')

# ── Per-mass-point generation ─────────────────────────────────────────────────

def rel_w1(gen, truth):
    r = np.ptp(truth)
    return wasserstein_distance(gen, truth) / r if r > 0 else float('nan')

results = []

for mx, my in HOLDOUT_MASS_POINTS:
    hdf5_path = _find_file(args.grid_dir, mx, my)
    label = f'({mx},{my})'
    print(f'\n[eval] {label}  file={os.path.basename(hdf5_path)}')

    with h5py.File(hdf5_path, 'r') as f:
        n_avail  = f['particle_features'].shape[0]
        n        = min(args.n_events, n_avail)
        pf_raw   = f['particle_features'][:n].astype(np.float32)
        part_raw = f['parton_features'][:n].astype(np.float32)
        file_mx  = float(f.attrs.get('mass_x', mx))
        file_my  = float(f.attrs.get('mass_y', my))

    N = len(pf_raw)

    # Parton conditioning
    mass_col = np.zeros((N, MAX_PARTONS, 1), dtype=np.float32)
    mass_col[:, 2, 0] = file_mx / MASS_NORM
    mass_col[:, 3, 0] = file_my / MASS_NORM
    part7     = np.concatenate([part_raw[:, :MAX_PARTONS, :], mass_col], axis=2)
    cond_raw  = part7.reshape(N, MAX_PARTONS * PARTON_FEAT)
    cond_norm = (cond_raw - cond_mean) / cond_std
    cond = np.concatenate([cond_norm, np.ones((N, MAX_PARTONS), np.float32)], axis=1)

    # Truth event features and log_npart
    raw7       = _compute_event_raw_all7(pf_raw, part_raw)
    event_feat = (raw7 - ev_mean) / ev_std  # (N, 7) normalized

    mask_truth = pf_raw[:, :NUM_PART, 6].astype(np.float32)
    npart      = mask_truth.sum(axis=1, keepdims=True)
    log_npart  = np.log(np.maximum(npart, 1.0))
    jet_truth  = (log_npart - jet_mean) / jet_std  # (N, 1)

    # Truth cone masses (from truth event features, denormalized)
    truth_cm_x = event_feat[:, 4] * ev_std[4] + ev_mean[4]  # log1p(GeV)
    truth_cm_y = event_feat[:, 6] * ev_std[6] + ev_mean[6]

    # Stage-2 generation with truth conditioning (no stage-1 sampling)
    nsplit = max(1, N // args.chunk_size)
    t0 = time.perf_counter()
    parts_gen, _ = model.generate(
        cond=cond,
        jet_mean=jet_mean,
        jet_std=jet_std,
        event_feat=event_feat,
        nsplit=nsplit,
        jets=jet_truth,          # truth log_npart → skip stage 1
        use_true_event=True,     # truth event features → skip stage-1 event output
        num_steps=args.num_steps,
        use_tqdm=True,
    )
    dt = time.perf_counter() - t0
    print(f'[eval] {label} generation done in {dt/60:.1f} min')

    # Denormalize generated particles
    parts_phys = parts_gen * part_std + part_mean  # (N, NUM_PART, 6)

    # Compute particle-level cone masses from generated cloud
    gen_cm_x, gen_cm_y = _cone_masses_from_gen(parts_phys, mask_truth, part_raw)

    w1_x = rel_w1(gen_cm_x, truth_cm_x)
    w1_y = rel_w1(gen_cm_y, truth_cm_y)
    print(f'[eval] {label}  cone_mass_X rel-W₁={w1_x:.4f}  cone_mass_Y rel-W₁={w1_y:.4f}')

    results.append({
        'label':       label,
        'gen_cm_x':    gen_cm_x,   'truth_cm_x': truth_cm_x,
        'gen_cm_y':    gen_cm_y,   'truth_cm_y': truth_cm_y,
        'w1_x':        w1_x,
        'w1_y':        w1_y,
        'dt':          dt,
    })

# ── Summary table ─────────────────────────────────────────────────────────────

SEP = '─' * 56
print(f'\n{SEP}')
print(f'  stage-2 cone mass  |  run: {args.run_name}')
print(f'  truth conditioning  |  {args.num_steps} steps  |  {args.n_events} events/pt')
print(SEP)
print(f'  {"mass point":<12}  {"cone_mass_X (rel-W₁)":>22}  {"cone_mass_Y (rel-W₁)":>22}')
print(SEP)
for r in results:
    print(f'  {r["label"]:<12}  {r["w1_x"]:>22.4f}  {r["w1_y"]:>22.4f}')
print(SEP)
mean_x = np.mean([r['w1_x'] for r in results])
mean_y = np.mean([r['w1_y'] for r in results])
print(f'  {"mean":<12}  {mean_x:>22.4f}  {mean_y:>22.4f}')
print(SEP)
total_t = sum(r['dt'] for r in results)
print(f'  Total time: {total_t/60:.1f} min\n')

# ── Save arrays for fast replotting ───────────────────────────────────────────

npz_out = args.npz_out
if npz_out is None:
    # Auto-name alongside plot output
    base = args.plot_out or f'stage2_cone_mass_{args.run_name}.png'
    npz_out = os.path.splitext(base)[0] + '_arrays.npz'

os.makedirs(os.path.dirname(os.path.abspath(npz_out)), exist_ok=True)
np.savez(npz_out, results=np.array(results, dtype=object))
print(f'[eval] arrays saved to {npz_out}')

# ── Plot ──────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _make_plot(results, args):
    try:
        epoch = json.load(open(os.path.join(args.ckpt_dir, args.run_name,
                                            'training_state.json'))).get('epochs_done', '?')
    except Exception:
        epoch = '?'

    linear = args.linear
    if linear:
        col_specs = [
            ('gen_cm_x', 'truth_cm_x', 'cone_mass_X (GeV)', 'w1_x'),
            ('gen_cm_y', 'truth_cm_y', 'cone_mass_Y (GeV)', 'w1_y'),
        ]
    else:
        col_specs = [
            ('gen_cm_x', 'truth_cm_x', 'log1p(cone_mass_X / GeV)', 'w1_x'),
            ('gen_cm_y', 'truth_cm_y', 'log1p(cone_mass_Y / GeV)', 'w1_y'),
        ]

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(9, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, r in enumerate(results):
        for col, (gk, tk, xlabel, w1k) in enumerate(col_specs):
            ax    = axes[row, col]
            gen   = np.expm1(r[gk]) if linear else r[gk]
            truth = np.expm1(r[tk]) if linear else r[tk]
            lo = min(np.percentile(truth, 0.5), np.percentile(gen, 0.5))
            hi = max(np.percentile(truth, 99.5), np.percentile(gen, 99.5))
            bins = np.linspace(lo, hi, 50)
            ax.hist(truth, bins=bins, histtype='step', density=True,
                    color='steelblue', linewidth=1.5, label='Truth')
            ax.hist(gen,   bins=bins, histtype='step', density=True,
                    color='tomato',    linewidth=1.5, label='Generated', linestyle='--')
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            mx_str, my_str = r['label'][1:-1].split(',')
            ax.set_title(f'mX={mx_str} mY={my_str}  rel-W₁={r[w1k]:.3f}', fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.legend(fontsize=7, frameon=False)

    scale_tag = 'linear' if linear else 'log1p'
    fig.suptitle(
        f'{args.run_name}  |  stage-2, truth conditioning  |  '
        f'{args.num_steps} steps  |  {args.n_events} events/pt  |  epoch {epoch}  |  {scale_tag}',
        fontsize=9,
    )
    fig.tight_layout()

    plot_out = args.plot_out or f'stage2_cone_mass_{args.run_name}.png'
    os.makedirs(os.path.dirname(os.path.abspath(plot_out)), exist_ok=True)
    fig.savefig(plot_out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[eval] plot saved to {plot_out}')

# Handle --from_npz replot path (deferred until stats are loaded for epoch lookup)
if args.from_npz:
    data = np.load(args.from_npz, allow_pickle=True)
    results = list(data['results'])
    _make_plot(results, args)
    sys.exit(0)

_make_plot(results, args)
