#!/usr/bin/env python3
"""
analyze_cfg_sweep.py — E035 guidance scale sweep analysis.

Reads inference NPZs from infer_cfg_sweep/s{scale}/ and produces:
  1. Terminal summary table (W1 distances + FWHM for cone masses, per scale, per mass point)
  2. Matplotlib plots: cone mass overlays at each scale, W1 vs scale curves
  3. Stage-1 event feature W1 (cone mass X/Y, cone pT X/Y, MET log1p)
  4. Stage-2 particle-level W1 (eta, logpT, phi, multiplicity)
  5. FWHM of cone mass distributions (empirical, no Gaussian assumption)
  6. Summary CSV at out_dir/cfg_sweep_summary.csv

Usage (run on login node or interactive GPU):
  python3 analyze_cfg_sweep.py \\
      --sweep_dir /pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/checkpoints_bsm_grid/bsm_grid_event_c_stage1_cfg/infer_cfg_sweep \\
      --stats_path /pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/checkpoints_bsm_grid/normalisation_stats_event_c_stage1_cfg.json \\
      --out_dir /pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/checkpoints_bsm_grid/bsm_grid_event_c_stage1_cfg/plots_e035

Requires: numpy, scipy, matplotlib
"""

import os, sys, glob, json, argparse, re
import numpy as np
from scipy.stats import wasserstein_distance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants matching training ────────────────────────────────────────────────

SCALES     = [1.0, 1.5, 3.0, 5.0, 7.5, 10.0]
SCALE_STRS = ['1p0', '1p5', '3p0', '5p0', '7p5', '10p0']
HOLDOUTS   = [(250, 250), (250, 300), (300, 250), (300, 300)]

FEAT_NAMES_EV = [
    'log1p(MET)', 'sin(MET_phi)', 'cos(MET_phi)',
    'log1p(cone_pT_X)', 'log1p(cone_mass_X)',
    'log1p(cone_pT_Y)',  'log1p(cone_mass_Y)',
]
# Feature indices of interest in the 7-dim event vector
IDX_CM_X = 4   # log1p(cone_mass_X)
IDX_CM_Y = 6   # log1p(cone_mass_Y)
IDX_CPT_X = 3  # log1p(cone_pT_X)
IDX_CPT_Y = 5  # log1p(cone_pT_Y)
IDX_MET   = 0  # log1p(MET)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep_dir',  required=True,
                   help='Root sweep dir containing s1p0/, s1p5/, ... subdirs')
    p.add_argument('--stats_path', required=True,
                   help='normalisation_stats_event_c_stage1_cfg.json')
    p.add_argument('--out_dir',    default=None,
                   help='Output directory for plots and CSV '
                        '(default: sweep_dir/../plots_e035)')
    p.add_argument('--n_max',      type=int, default=5000,
                   help='Max events to use per NPZ (default 5000)')
    return p.parse_args()


args = _parse()
sweep_dir = args.sweep_dir
out_dir   = args.out_dir or os.path.join(os.path.dirname(sweep_dir), 'plots_e035')
os.makedirs(out_dir, exist_ok=True)

# ── Load normalisation stats ────────────────────────────────────────────────────

with open(args.stats_path) as fh:
    stats = json.load(fh)

ev_mean = np.array(stats['jet_mean'], dtype=np.float32)[1:]   # (7,) event feature means
ev_std  = np.array(stats['jet_std'],  dtype=np.float32)[1:]   # (7,) event feature stds
jet_mean = float(stats['jet_mean'][0])
jet_std  = float(stats['jet_std'][0])
part_mean = np.array(stats['part_mean'], dtype=np.float32)
part_std  = np.array(stats['part_std'],  dtype=np.float32)

print(f'[analyze] ev_mean = {ev_mean.round(3)}')
print(f'[analyze] ev_std  = {ev_std.round(3)}')


# ── Helper functions ───────────────────────────────────────────────────────────

def rel_w1(gen, truth):
    """Relative W1: W1(gen,truth) / range(truth)."""
    r = float(np.ptp(truth))
    if r < 1e-10:
        return float('nan')
    return float(wasserstein_distance(gen, truth)) / r


def abs_w1(gen, truth):
    """Absolute W1 (same units as input)."""
    return float(wasserstein_distance(gen, truth))


def fwhm(arr):
    """Empirical FWHM of a 1-D distribution (no Gaussian assumption)."""
    bins = np.linspace(np.percentile(arr, 0.5), np.percentile(arr, 99.5), 200)
    h, edges = np.histogram(arr, bins=bins)
    peak_idx = np.argmax(h)
    peak_val = h[peak_idx]
    half     = peak_val / 2.0
    centers  = 0.5 * (edges[:-1] + edges[1:])
    # left edge: last bin below peak that is <= half
    left_idx  = peak_idx
    for i in range(peak_idx, -1, -1):
        if h[i] <= half:
            left_idx = i
            break
    # right edge: first bin above peak that is <= half
    right_idx = peak_idx
    for i in range(peak_idx, len(h)):
        if h[i] <= half:
            right_idx = i
            break
    return float(centers[right_idx] - centers[left_idx])


def denorm_ev(ef_norm):
    """Denormalize 7-dim event features from normalised space."""
    return ef_norm * ev_std + ev_mean


def load_npz(path):
    d = np.load(path)
    return d


# ── Load all NPZs ──────────────────────────────────────────────────────────────

# Structure: results[scale_idx][mass_idx] = dict of arrays
print('[analyze] Loading NPZ files ...')

results = {}  # keyed by (scale_float, mx, my)
missing = []

for scale, sstr in zip(SCALES, SCALE_STRS):
    sdir = os.path.join(sweep_dir, f's{sstr}')
    for mx, my in HOLDOUTS:
        npz_path = os.path.join(sdir, f'bsm_mX{mx:04d}_mY{my:04d}_rank00_of01.npz')
        if not os.path.exists(npz_path):
            missing.append(npz_path)
            results[(scale, mx, my)] = None
            continue
        d = load_npz(npz_path)
        n = min(args.n_max, len(d['parts_truth']))
        results[(scale, mx, my)] = {
            'parts_truth':       d['parts_truth'][:n],
            'parts_gen':         d['parts_gen'][:n],
            'mask':              d['mask'][:n].astype(np.float32),
            'mask_gen':          d['mask_gen'][:n].astype(np.float32),
            'event_feat_truth':  d['event_feat_truth'][:n],   # (N,7) normalized
            'jets_gen':          d['jets_gen'][:n],            # (N,8) normalized
            'guidance_scale':    float(d['guidance_scale']),
        }
        print(f'  s={scale}  ({mx},{my})  n={n}  '
              f'mean_npart_truth={d["mask"][:n].sum(1).mean():.1f}  '
              f'mean_npart_gen={d["mask_gen"][:n].sum(1).mean():.1f}')

if missing:
    print(f'\n[analyze] WARNING: {len(missing)} NPZ(s) missing:')
    for p in missing:
        print(f'  {p}')
    print('[analyze] Run submit_e035_perlmutter_cfg_sweep.sh first, then re-run.')

available = {k: v for k, v in results.items() if v is not None}
if not available:
    sys.exit('[analyze] No NPZs found. Exiting.')


# ── Compute metrics ────────────────────────────────────────────────────────────

print('\n[analyze] Computing metrics ...')

metrics = {}  # (scale, mx, my) → dict

for (scale, mx, my), r in results.items():
    if r is None:
        metrics[(scale, mx, my)] = None
        continue

    # Event features (denormalized)
    ef_truth_phys = denorm_ev(r['event_feat_truth'])        # (N,7)
    ef_gen_phys   = denorm_ev(r['jets_gen'][:, 1:])         # (N,7); col 0 = log_npart

    # Stage-1 event feature W1 (relative)
    ev_rw1 = {feat: rel_w1(ef_gen_phys[:, i], ef_truth_phys[:, i])
              for i, feat in enumerate(FEAT_NAMES_EV)}

    # Stage-1 event feature W1 (absolute, physical units for cone masses)
    cm_x_truth_phys = np.expm1(np.clip(ef_truth_phys[:, IDX_CM_X], 0, 15))   # GeV
    cm_y_truth_phys = np.expm1(np.clip(ef_truth_phys[:, IDX_CM_Y], 0, 15))
    cm_x_gen_phys   = np.expm1(np.clip(ef_gen_phys[:, IDX_CM_X],   0, 15))
    cm_y_gen_phys   = np.expm1(np.clip(ef_gen_phys[:, IDX_CM_Y],   0, 15))

    # Particle-level features (flatten with mask)
    mt = r['mask'].astype(bool)    # (N, npart)
    mg = r['mask_gen'].astype(bool)

    eta_t   = r['parts_truth'][mt, 0]
    eta_g   = r['parts_gen'][mg, 0]
    logpT_t = r['parts_truth'][mt, 3]
    logpT_g = r['parts_gen'][mg, 3]
    phi_t   = np.arctan2(r['parts_truth'][mt, 1], r['parts_truth'][mt, 2])
    phi_g   = np.arctan2(r['parts_gen'][mg, 1],   r['parts_gen'][mg, 2])
    npart_t = mt.sum(axis=1).astype(float)
    npart_g = mg.sum(axis=1).astype(float)

    metrics[(scale, mx, my)] = {
        # Stage-1 event feature relative W1
        'rw1_cm_x':    ev_rw1['log1p(cone_mass_X)'],
        'rw1_cm_y':    ev_rw1['log1p(cone_mass_Y)'],
        'rw1_cpt_x':   ev_rw1['log1p(cone_pT_X)'],
        'rw1_cpt_y':   ev_rw1['log1p(cone_pT_Y)'],
        'rw1_met':     ev_rw1['log1p(MET)'],
        # Particle-level relative W1
        'rw1_eta':     rel_w1(eta_g, eta_t),
        'rw1_logpT':   rel_w1(logpT_g, logpT_t),
        'rw1_phi':     rel_w1(phi_g, phi_t),
        'rw1_npart':   rel_w1(npart_g, npart_t),
        # FWHM in GeV (physical cone masses)
        'fwhm_cm_x':   fwhm(cm_x_gen_phys),
        'fwhm_cm_y':   fwhm(cm_y_gen_phys),
        'fwhm_cm_x_truth': fwhm(cm_x_truth_phys),
        'fwhm_cm_y_truth': fwhm(cm_y_truth_phys),
        # Arrays for plotting
        '_cm_x_truth': cm_x_truth_phys,
        '_cm_y_truth': cm_y_truth_phys,
        '_cm_x_gen':   cm_x_gen_phys,
        '_cm_y_gen':   cm_y_gen_phys,
        '_npart_t':    npart_t,
        '_npart_g':    npart_g,
    }


# ── Terminal summary table ─────────────────────────────────────────────────────

METRIC_COLS = [
    ('rw1_cm_x',  'rel-W1 cm_X'),
    ('rw1_cm_y',  'rel-W1 cm_Y'),
    ('rw1_cpt_x', 'rel-W1 cpT_X'),
    ('rw1_cpt_y', 'rel-W1 cpT_Y'),
    ('rw1_met',   'rel-W1 MET'),
    ('rw1_eta',   'rel-W1 η'),
    ('rw1_logpT', 'rel-W1 logpT'),
    ('rw1_phi',   'rel-W1 φ'),
    ('rw1_npart', 'rel-W1 Npart'),
    ('fwhm_cm_x', 'FWHM cm_X [GeV]'),
    ('fwhm_cm_y', 'FWHM cm_Y [GeV]'),
]

print('\n' + '═' * 100)
print('  E035 — CFG guidance scale sweep  |  E034 epoch 43  |  5000 events / point  |  500 DDPM steps')
print('═' * 100)

# Mean over mass points per scale
print('\n  ── Mean over 4 holdout mass points ──\n')
header = f'  {"Scale":>6}  ' + '  '.join(f'{n:>15}' for _, n in METRIC_COLS)
print(header)
print('  ' + '─' * (len(header) - 2))

scale_means = {}
for scale in SCALES:
    row_vals = {}
    for key, name in METRIC_COLS:
        vals = [metrics[(scale, mx, my)][key]
                for mx, my in HOLDOUTS
                if metrics.get((scale, mx, my)) is not None]
        row_vals[key] = np.nanmean(vals) if vals else float('nan')
    scale_means[scale] = row_vals
    vals_str = '  '.join(f'{row_vals[k]:>15.4f}' for k, _ in METRIC_COLS)
    print(f'  {scale:>6.1f}  {vals_str}')

print('═' * 100)

# Per-mass-point breakdown per scale
print('\n  ── Per mass point breakdown ──\n')
for mx, my in HOLDOUTS:
    print(f'  Mass point ({mx},{my}):')
    print(f'  {"Scale":>6}  ' + '  '.join(f'{n:>15}' for _, n in METRIC_COLS))
    for scale in SCALES:
        m = metrics.get((scale, mx, my))
        if m is None:
            vals_str = '  '.join(f'{"MISSING":>15}' for _ in METRIC_COLS)
        else:
            vals_str = '  '.join(f'{m[k]:>15.4f}' for k, _ in METRIC_COLS)
        print(f'  {scale:>6.1f}  {vals_str}')
    print()


# ── Save CSV ───────────────────────────────────────────────────────────────────

csv_path = os.path.join(out_dir, 'cfg_sweep_summary.csv')
with open(csv_path, 'w') as fh:
    fh.write('scale,mx,my,' + ','.join(k for k, _ in METRIC_COLS) + '\n')
    for scale in SCALES:
        for mx, my in HOLDOUTS:
            m = metrics.get((scale, mx, my))
            if m is None:
                row = ','.join(['nan'] * len(METRIC_COLS))
            else:
                row = ','.join(f'{m[k]:.6f}' for k, _ in METRIC_COLS)
            fh.write(f'{scale},{mx},{my},{row}\n')
print(f'\n[analyze] CSV saved: {csv_path}')


# ── Plot 1: W1 vs guidance scale (mean over mass points) ──────────────────────

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('E035 — CFG guidance scale sweep  |  E034 epoch 43  |  Mean over 4 holdout mass points',
             fontsize=11)

plot_pairs = [
    (axes[0, 0], 'rw1_cm_x',  'rel-W₁  log1p(cone_mass_X)',  '#c0392b'),
    (axes[0, 1], 'rw1_cm_y',  'rel-W₁  log1p(cone_mass_Y)',  '#e67e22'),
    (axes[0, 2], 'rw1_met',   'rel-W₁  log1p(MET)',           '#8e44ad'),
    (axes[1, 0], 'rw1_cpt_x', 'rel-W₁  log1p(cone_pT_X)',    '#27ae60'),
    (axes[1, 1], 'rw1_cpt_y', 'rel-W₁  log1p(cone_pT_Y)',    '#2980b9'),
    (axes[1, 2], 'rw1_npart', 'rel-W₁  Npart (multiplicity)', '#7f8c8d'),
]

for ax, key, ylabel, color in plot_pairs:
    y_mean = [scale_means[s][key] for s in SCALES]
    # Per-mass-point scatter
    for mx, my in HOLDOUTS:
        y_pt = [metrics.get((s, mx, my), {}) or {} for s in SCALES]
        y_pt = [m.get(key, np.nan) if m else np.nan for m in y_pt]
        ax.plot(SCALES, y_pt, 'o--', color=color, alpha=0.3, linewidth=1,
                markersize=4, label=f'({mx},{my})')
    ax.plot(SCALES, y_mean, 'o-', color=color, linewidth=2.5, markersize=7,
            label='mean', zorder=5)
    ax.axvline(x=1.0, color='grey', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Guidance scale s', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xscale('log')
    ax.set_xticks(SCALES)
    ax.set_xticklabels([str(s) for s in SCALES], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
plot1_path = os.path.join(out_dir, 'e035_w1_vs_scale.png')
fig.savefig(plot1_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'[analyze] Plot 1 saved: {plot1_path}')


# ── Plot 2: FWHM vs guidance scale ────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle('E035 — Cone mass FWHM vs guidance scale  |  E034 epoch 43', fontsize=11)

for ax, key, key_truth, label, color in [
    (axes[0], 'fwhm_cm_x', 'fwhm_cm_x_truth', 'cone_mass_X [GeV]', '#c0392b'),
    (axes[1], 'fwhm_cm_y', 'fwhm_cm_y_truth', 'cone_mass_Y [GeV]', '#e67e22'),
]:
    for i, (mx, my) in enumerate(HOLDOUTS):
        truth_fwhm = np.nanmean([
            metrics[(s, mx, my)][key_truth]
            for s in SCALES
            if metrics.get((s, mx, my)) is not None
        ]) if any(metrics.get((s, mx, my)) for s in SCALES) else np.nan
        gen_fwhm = [
            metrics[(s, mx, my)][key] if metrics.get((s, mx, my)) else np.nan
            for s in SCALES
        ]
        pt_color = plt.cm.tab10(i)
        ax.plot(SCALES, gen_fwhm, 'o-', color=pt_color, linewidth=1.5,
                markersize=5, label=f'({mx},{my})', zorder=4)
        ax.axhline(y=truth_fwhm, color=pt_color, linewidth=1.0, linestyle='--',
                   alpha=0.6)

    ax.axvline(x=1.0, color='grey', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Guidance scale s', fontsize=9)
    ax.set_ylabel(f'FWHM  {label}', fontsize=9)
    ax.set_xscale('log')
    ax.set_xticks(SCALES)
    ax.set_xticklabels([str(s) for s in SCALES], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, frameon=False, title='solid=gen, dashed=truth', title_fontsize=7)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
plot2_path = os.path.join(out_dir, 'e035_fwhm_vs_scale.png')
fig.savefig(plot2_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'[analyze] Plot 2 saved: {plot2_path}')


# ── Plot 3: Cone mass overlays at each scale (mean over mass points) ──────────

n_scales = len(SCALES)
n_mass   = len(HOLDOUTS)

for col_feat, key_t, key_g, feat_label in [
    (IDX_CM_X, '_cm_x_truth', '_cm_x_gen', 'cone_mass_X [GeV]'),
    (IDX_CM_Y, '_cm_y_truth', '_cm_y_gen', 'cone_mass_Y [GeV]'),
]:
    fig, axes = plt.subplots(n_mass, n_scales, figsize=(3.5 * n_scales, 3 * n_mass),
                             sharex='row', sharey='row')
    fig.suptitle(f'E035 — {feat_label} distributions by guidance scale  |  E034 epoch 43',
                 fontsize=11)

    for row, (mx, my) in enumerate(HOLDOUTS):
        # Collect truth from s=1.0 (should be identical across scales)
        truth_ref = None
        for s in SCALES:
            m = metrics.get((s, mx, my))
            if m is not None and m[key_t] is not None:
                truth_ref = m[key_t]
                break

        for col, scale in enumerate(SCALES):
            ax = axes[row, col]
            m = metrics.get((scale, mx, my))

            if truth_ref is not None and len(truth_ref) > 0:
                lo = float(np.percentile(truth_ref, 0.5))
                hi = float(np.percentile(truth_ref, 99.5))
                bins = np.linspace(lo, hi, 50)
                ax.hist(truth_ref, bins=bins, histtype='step', density=True,
                        color='steelblue', linewidth=1.5, label='Truth')

            if m is not None:
                gen = m[key_g]
                lo2 = float(np.percentile(gen, 0.5)) if len(gen) > 0 else 0
                hi2 = float(np.percentile(gen, 99.5)) if len(gen) > 0 else 1
                bins2 = np.linspace(min(lo, lo2) if truth_ref is not None else lo2,
                                    max(hi, hi2) if truth_ref is not None else hi2, 50)
                ax.hist(gen, bins=bins2, histtype='step', density=True,
                        color='tomato', linewidth=1.5, linestyle='--', label='Gen')
                rw1 = m[f'rw1_cm_x' if 'X' in feat_label else 'rw1_cm_y']
                ax.text(0.97, 0.97, f'rel-W₁={rw1:.3f}',
                        transform=ax.transAxes, fontsize=7,
                        ha='right', va='top')

            if row == 0:
                ax.set_title(f's={scale}', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'({mx},{my})', fontsize=8)
            ax.set_xlabel(feat_label, fontsize=7)
            ax.tick_params(labelsize=7)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fname = f'e035_conemass_{"X" if "X" in feat_label else "Y"}_by_scale.png'
    plot3_path = os.path.join(out_dir, fname)
    fig.savefig(plot3_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[analyze] Plot 3 saved: {plot3_path}')


# ── Plot 4: Particle-level W1 vs scale ────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle('E035 — Particle-level distributions vs guidance scale  |  E034 epoch 43',
             fontsize=11)

for ax, key, ylabel, color in [
    (axes[0], 'rw1_eta',   'rel-W₁  η',     '#2c3e50'),
    (axes[1], 'rw1_logpT', 'rel-W₁  log pT', '#16a085'),
    (axes[2], 'rw1_phi',   'rel-W₁  φ',     '#8e44ad'),
]:
    y_mean = [scale_means[s][key] for s in SCALES]
    for mx, my in HOLDOUTS:
        y_pt = [metrics.get((s, mx, my), {}) or {} for s in HOLDOUTS]
        y_pt = [
            metrics[(s, mx, my)][key]
            if metrics.get((s, mx, my)) is not None else np.nan
            for s in SCALES
        ]
        ax.plot(SCALES, y_pt, 'o--', color=color, alpha=0.3, linewidth=1,
                markersize=4)
    ax.plot(SCALES, y_mean, 'o-', color=color, linewidth=2.5, markersize=7,
            label='mean', zorder=5)
    ax.axvline(x=1.0, color='grey', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Guidance scale s', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xscale('log')
    ax.set_xticks(SCALES)
    ax.set_xticklabels([str(s) for s in SCALES], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
plot4_path = os.path.join(out_dir, 'e035_particle_w1_vs_scale.png')
fig.savefig(plot4_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'[analyze] Plot 4 saved: {plot4_path}')

print(f'\n[analyze] Done. All outputs written to {out_dir}')
