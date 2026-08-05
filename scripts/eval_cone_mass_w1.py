#!/usr/bin/env python3
"""
eval_cone_mass_w1.py — Quick cone-mass W₁ diagnostic (stage-1 only).

Runs stage-1 DDPM (no particle generation) for the 4 BSM holdout mass
points and prints relative-W₁ for log1p(cone_mass_X) and log1p(cone_mass_Y).
Also prints all 7 event-feature W₁ values for a full picture.

Typical runtime: ~30–60 s on CPU, ~5 s on a single GPU
at --n_events 2000 --num_jet_steps 100.

Usage:
  python3 eval_cone_mass_w1.py \\
      --run_name bsm_grid_event_c_stage1_cfg \\
      --arch     stage1_cfg \\
      --ckpt_dir /pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid \\
      --grid_dir /pub/lcondren/wprime_signal_mpi

  # E023 baseline
  python3 eval_cone_mass_w1.py \\
      --run_name bsm_grid_event_c_stage1_mpi \\
      --arch     stage1 \\
      --ckpt_dir /pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid \\
      --grid_dir /pub/lcondren/wprime_signal_mpi
"""

import os, sys, glob, re, json, argparse, importlib, time
import numpy as np
from scipy.stats import wasserstein_distance

# ── Constants (must match training) ──────────────────────────────────────────

MAX_PARTONS = 4
PARTON_FEAT = 7
MASS_NORM   = 600.0
R_CONE      = 1.0
NUM_PART    = 500   # used only for event-feature computation

HOLDOUT_MASS_POINTS = [(250, 250), (250, 300), (300, 250), (300, 300)]

FEAT_NAMES = [
    'log1p(MET)',
    'sin(MET_phi)',
    'cos(MET_phi)',
    'log1p(cone_pT_X)',
    'log1p(cone_mass_X)',   # index 4 ← target
    'log1p(cone_pT_Y)',
    'log1p(cone_mass_Y)',   # index 6 ← target
]

# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name',      default='bsm_grid_event_c_stage1_cfg')
    p.add_argument('--arch',          default=None,
                   help='Architecture module suffix: stage1 or stage1_cfg. '
                        'Auto-detected from run_name if not set.')
    p.add_argument('--ckpt_dir',      default='/pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid')
    p.add_argument('--grid_dir',      default='/pub/lcondren/wprime_signal_mpi')
    p.add_argument('--stats_path',    default=None,
                   help='Stats JSON path. Auto-detected from arch if not set.')
    p.add_argument('--n_events',      type=int, default=2000,
                   help='Events per mass point (default 2000).')
    p.add_argument('--num_jet_steps', type=int, default=500,
                   help='DDPM steps for stage-1 (default 500, matching training).')
    p.add_argument('--proj_dim',      type=int, default=128)
    p.add_argument('--num_layers',    type=int, default=8)
    p.add_argument('--num_gen_layers',type=int, default=2)
    p.add_argument('--num_jet_mlp',   type=int, default=512)
    p.add_argument('--gpu',           type=int, default=0,
                   help='GPU index to use (-1 = CPU only).')
    p.add_argument('--plot',          action='store_true', default=False,
                   help='Save a cone-mass overlay plot.')
    p.add_argument('--plot_out',      default=None,
                   help='Output PNG path (default: cone_mass_<run_name>.png).')
    return p.parse_args()

args = _parse()

# ── GPU / CPU selection ───────────────────────────────────────────────────────

if args.gpu < 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import tensorflow as tf

# ── Arch auto-detection ───────────────────────────────────────────────────────

arch = args.arch
if arch is None:
    arch = 'stage1_cfg' if 'cfg' in args.run_name else 'stage1'
    print(f'[eval] auto-detected arch={arch} from run_name={args.run_name}')

ARCH_MODULE = {
    'stage1':     'PET_pp_parton_vpar_bsm_event_c_stage1',
    'stage1_cfg': 'PET_pp_parton_vpar_bsm_event_c_stage1_cfg',
}
if arch not in ARCH_MODULE:
    raise ValueError(f'Unknown --arch {arch!r}. Use: stage1 or stage1_cfg')
module_name = ARCH_MODULE[arch]

# ── Stats path auto-detection ─────────────────────────────────────────────────

STATS_NAME = {
    'stage1':     'normalisation_stats_event_c_stage1.json',
    'stage1_cfg': 'normalisation_stats_event_c_stage1_cfg.json',
}
stats_path = args.stats_path or os.path.join(args.ckpt_dir, STATS_NAME[arch])
if not os.path.exists(stats_path):
    sys.exit(f'Stats not found: {stats_path}')

with open(stats_path) as fh:
    stats = json.load(fh)

cond_mean  = np.array(stats['cond_mean'], dtype=np.float32)
cond_std   = np.array(stats['cond_std'],  dtype=np.float32)
jet_mean_a = np.array(stats['jet_mean'],  dtype=np.float32)  # (8,)
jet_std_a  = np.array(stats['jet_std'],   dtype=np.float32)  # (8,)
jet_mean   = float(jet_mean_a[0])
jet_std    = float(jet_std_a[0])
ev_mean    = jet_mean_a[1:]   # (7,)
ev_std     = jet_std_a[1:]    # (7,)

print(f'[eval] stats loaded from {os.path.basename(stats_path)}')
print(f'[eval] ev_mean={ev_mean.round(3)}')
print(f'[eval] ev_std ={ev_std.round(3)}')

# ── Event feature computation (same formula as training) ─────────────────────

def _compute_event_raw_all7(pf_raw, part_raw, num_part=NUM_PART):
    """Compute 7-dim event features from raw HDF5 particle/parton data."""
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
    for slot in [2, 3]:  # X=parton slot 2, Y=parton slot 3
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

# ── HDF5 file finder ──────────────────────────────────────────────────────────

def _find_file(grid_dir, mx, my):
    pattern = os.path.join(grid_dir, f'signal_mX{mx:04d}_mY{my:04d}.hdf5')
    if os.path.exists(pattern):
        return pattern
    # fallback: glob
    candidates = glob.glob(os.path.join(grid_dir, 'signal_mX*.hdf5'))
    def _dist(f):
        m = re.search(r'mX(\d+)_mY(\d+)', os.path.basename(f))
        if not m: return float('inf')
        return (int(m.group(1)) - mx)**2 + (int(m.group(2)) - my)**2
    best = min(candidates, key=_dist, default=None)
    if best is None:
        sys.exit(f'No signal files found in {grid_dir}')
    return best

# ── Load model ────────────────────────────────────────────────────────────────

ckpt_path = os.path.join(args.ckpt_dir, args.run_name, 'pet_pp.weights.h5')
if not os.path.exists(ckpt_path):
    sys.exit(f'Checkpoint not found: {ckpt_path}')

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts_dir)
mod = importlib.import_module(module_name)
ModelCls = getattr(mod, 'PET_pp_parton_vpar_bsm_event_c_stage1')

model = ModelCls(
    num_feat=6, num_jet=8,
    max_partons=MAX_PARTONS,
    parton_feat=PARTON_FEAT,
    num_event_feat=7,
    num_part=500,
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
print(f'[eval] loaded checkpoint: {ckpt_path}')

# ── Per-mass-point evaluation ─────────────────────────────────────────────────

import h5py

def rel_w1(gen, truth):
    r = np.ptp(truth)
    return wasserstein_distance(gen, truth) / r if r > 0 else float('nan')

results = []

for mx, my in HOLDOUT_MASS_POINTS:
    hdf5_path = _find_file(args.grid_dir, mx, my)
    label = f'({mx},{my})'

    with h5py.File(hdf5_path, 'r') as f:
        n_avail = f['particle_features'].shape[0]
        n = min(args.n_events, n_avail)
        pf_raw   = f['particle_features'][:n].astype(np.float32)
        part_raw = f['parton_features'][:n].astype(np.float32)
        file_mx  = float(f.attrs.get('mass_x', mx))
        file_my  = float(f.attrs.get('mass_y', my))

    N = len(pf_raw)

    # Build parton conditioning
    mass_col = np.zeros((N, MAX_PARTONS, 1), dtype=np.float32)
    mass_col[:, 2, 0] = file_mx / MASS_NORM
    mass_col[:, 3, 0] = file_my / MASS_NORM
    part7     = np.concatenate([part_raw[:, :MAX_PARTONS, :], mass_col], axis=2)
    cond_raw  = part7.reshape(N, MAX_PARTONS * PARTON_FEAT)
    cond_norm = (cond_raw - cond_mean) / cond_std
    parton_mask = np.ones((N, MAX_PARTONS), dtype=np.float32)
    cond = np.concatenate([cond_norm, parton_mask], axis=1)  # (N, 32)

    # Truth event features
    raw7       = _compute_event_raw_all7(pf_raw, part_raw)
    event_feat = (raw7 - ev_mean) / ev_std  # normalized

    # Stage-1 DDPM
    t0 = time.perf_counter()
    jets_gen = model.DDPMSampler(
        cond, model.ema_jet,
        data_shape=[N, 8],
        w=0.0,
        num_steps=args.num_jet_steps,
        const_shape=[-1, 1],
    ).numpy()
    dt = time.perf_counter() - t0

    # jets_gen: (N, 8) — col 0 = log_npart, cols 1-7 = event features
    ef_gen  = jets_gen[:, 1:]   # (N, 7) normalized

    # Denormalize to physical space
    ef_gen_phys   = ef_gen  * ev_std + ev_mean
    ef_truth_phys = event_feat * ev_std + ev_mean

    # W₁ for all 7 features
    w1s = [rel_w1(ef_gen_phys[:, i], ef_truth_phys[:, i]) for i in range(7)]

    results.append({
        'label': label,
        'w1s': w1s,
        'dt': dt,
        'N': N,
        'cm_x_gen':   ef_gen_phys[:, 4],
        'cm_y_gen':   ef_gen_phys[:, 6],
        'cm_x_truth': ef_truth_phys[:, 4],
        'cm_y_truth': ef_truth_phys[:, 6],
    })
    print(f'[eval] {label} done in {dt:.1f}s  '
          f'cone_mass_X rel-W₁={w1s[4]:.4f}  cone_mass_Y rel-W₁={w1s[6]:.4f}')

# ── Summary table ─────────────────────────────────────────────────────────────

CONE_MASS_IDX = [4, 6]
SEP = '─' * 78

print(f'\n{SEP}')
print(f'  eval_cone_mass_w1  |  run: {args.run_name}  |  {args.num_jet_steps} steps  |  {args.n_events} events/mass-pt')
print(SEP)

# Header
col_w = 22
header_feats = [f'{n[:col_w]:>{col_w}}' for n in FEAT_NAMES]
print(f'  {"mass point":<12}  ' + '  '.join(header_feats))
print(f'  {"":12}  ' + '  '.join(['(rel-W₁)'.rjust(col_w)] * 7))
print(SEP)

for r in results:
    vals = '  '.join(
        f'\033[1;33m{r["w1s"][i]:>{col_w}.4f}\033[0m' if i in CONE_MASS_IDX
        else f'{r["w1s"][i]:>{col_w}.4f}'
        for i in range(7)
    )
    print(f'  {r["label"]:<12}  {vals}')

print(SEP)

# Mean across mass points
mean_w1s = np.mean([[r['w1s'][i] for i in range(7)] for r in results], axis=0)
vals_mean = '  '.join(
    f'\033[1;33m{mean_w1s[i]:>{col_w}.4f}\033[0m' if i in CONE_MASS_IDX
    else f'{mean_w1s[i]:>{col_w}.4f}'
    for i in range(7)
)
print(f'  {"mean":<12}  {vals_mean}')
print(SEP)

print(f'\n  Highlighted columns: cone_mass_X (index 4) and cone_mass_Y (index 6)')
total_t = sum(r['dt'] for r in results)
print(f'  Total stage-1 time: {total_t:.1f}s across {len(results)} mass points\n')

# ── Optional plot ─────────────────────────────────────────────────────────────

if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(9, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, r in enumerate(results):
        for col, (key_g, key_t, feat_label, feat_idx) in enumerate([
            ('cm_x_gen', 'cm_x_truth', 'log1p(cone_mass_X / GeV)', 4),
            ('cm_y_gen', 'cm_y_truth', 'log1p(cone_mass_Y / GeV)', 6),
        ]):
            ax = axes[row, col]
            gen   = r[key_g]
            truth = r[key_t]
            lo = min(np.percentile(truth, 0.5), np.percentile(gen, 0.5))
            hi = max(np.percentile(truth, 99.5), np.percentile(gen, 99.5))
            bins = np.linspace(lo, hi, 50)
            ax.hist(truth, bins=bins, histtype='step', density=True,
                    color='steelblue', linewidth=1.5, label='Truth')
            ax.hist(gen,   bins=bins, histtype='step', density=True,
                    color='tomato',    linewidth=1.5, label='Generated', linestyle='--')
            ax.set_xlabel(feat_label, fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.set_title(
                f'mX={r["label"].split(",")[0][1:]} mY={r["label"].split(",")[1][:-1]}  '
                f'rel-W₁={r["w1s"][feat_idx]:.3f}',
                fontsize=8,
            )
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.legend(fontsize=7, frameon=False)

    fig.suptitle(
        f'{args.run_name}  |  {args.num_jet_steps} steps  |  {args.n_events} events/pt  '
        f'|  epoch {json.load(open(os.path.join(args.ckpt_dir, args.run_name, "training_state.json"))).get("epochs_done","?")}',
        fontsize=9,
    )
    fig.tight_layout()

    plot_out = args.plot_out or f'cone_mass_{args.run_name}.png'
    fig.savefig(plot_out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[eval] plot saved to {plot_out}')
