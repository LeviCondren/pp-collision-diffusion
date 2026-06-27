#!/usr/bin/env python3
"""
Mass-conditioning diagnostic overlay for E008.

Overlays reconstructed event-level distributions for all available mass
points on a single set of axes so it is immediately visible whether the
model produces distinct outputs for different (mX, mY) inputs.

If the model uses mass conditioning correctly:
  - Generated curves (dotted) fan out with mass just as truth curves (solid) do.
If the model ignores conditioning:
  - Generated curves pile on top of each other regardless of mass.

A companion scatter plot shows W1(truth, generated) vs. mX+mY, separately
marking held-out (*) and trained (o) points so interpolation failure is
distinguishable from global architectural underfitting.

Usage:
    module load tensorflow/2.15.0
    python3 plot_e008_mass_overlay.py \
        [--infer_dirs DIR1 DIR2 ...] \
        [--out_dir PLOT_DIR] \
        [--n_events 5000]

Defaults scan both the ep019 holdout and trained-point directories.
"""

import os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance as _wass

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
_BASE = ('/pscratch/sd/l/lcondren/MCsim/wprime_signal/'
         'checkpoints_bsm_grid/bsm_grid')
p.add_argument('--infer_dirs', nargs='+', default=[
    os.path.join(_BASE, 'infer_holdout_ep019_5k'),
    os.path.join(_BASE, 'infer_trained_ep019_5k'),
])
p.add_argument('--out_dir',   default=os.path.join(_BASE, 'plots_mass_overlay'))
p.add_argument('--n_events',  type=int, default=5000)
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
N = args.n_events

# ── Load ──────────────────────────────────────────────────────────────────────
entries = []
for d in args.infer_dirs:
    if not os.path.isdir(d):
        print(f'  [skip] not found: {d}')
        continue
    src = 'holdout' if 'holdout' in os.path.basename(d) else 'trained'
    for path in sorted(glob.glob(os.path.join(d, 'bsm_mX*.npz'))):
        f = np.load(path)
        mX = float(f['mass_x']);  mY = float(f['mass_y'])
        entries.append({
            'key': f'({int(mX)},{int(mY)})',
            'src': src, 'mX': mX, 'mY': mY, 'mSum': mX + mY,
            'parts_truth': f['parts_truth'][:N],
            'parts_gen':   f['parts_gen'][:N],
            'mask':        f['mask'][:N],
            'mask_gen':    f['mask_gen'][:N],
        })
        print(f'  loaded {src} mX={int(mX)} mY={int(mY)}  ({path})')

assert entries, 'No bsm_mX*.npz files found in any --infer_dirs'

entries.sort(key=lambda e: (e['mSum'], e['mX']))

# ── Color map: viridis gradient by mX+mY ─────────────────────────────────────
all_msum = [e['mSum'] for e in entries]
_lo, _hi = min(all_msum), max(all_msum)
_cmap = cm.get_cmap('plasma')

def _color(mSum):
    t = (mSum - _lo) / max(_hi - _lo, 1.0)
    return _cmap(0.15 + 0.70 * t)

# ── Observables ───────────────────────────────────────────────────────────────
def _pT_m(parts, mask):
    return np.exp(np.clip(parts[:, :, 3], -10, 10)) * mask

def obs_mult(mask):         return mask.sum(axis=1)
def obs_HT(parts, mask):    return _pT_m(parts, mask).sum(axis=1)
def obs_MET(parts, mask):
    pT = _pT_m(parts, mask)
    return np.sqrt((pT * parts[:, :, 2]).sum(1)**2 +
                   (pT * parts[:, :, 1]).sum(1)**2)
def obs_lead_pT(parts, mask):
    return _pT_m(parts, mask).max(axis=1)
def obs_HT_top4(parts, mask):
    pT = _pT_m(parts, mask)
    top4 = np.sort(pT, axis=1)[:, -4:]
    return top4.sum(axis=1)

for e in entries:
    pt = e['parts_truth']; m = e['mask']
    pg = e['parts_gen'];   mg = e['mask_gen']
    e['mult_t']   = obs_mult(m);       e['mult_g']   = obs_mult(mg)
    e['HT_t']     = obs_HT(pt, m);    e['HT_g']     = obs_HT(pg, mg)
    e['MET_t']    = obs_MET(pt, m);   e['MET_g']    = obs_MET(pg, mg)
    e['leadpT_t'] = obs_lead_pT(pt, m); e['leadpT_g'] = obs_lead_pT(pg, mg)
    e['HT4_t']    = obs_HT_top4(pt, m); e['HT4_g']   = obs_HT_top4(pg, mg)

# ── Shared legend handles ─────────────────────────────────────────────────────
def _legend_handles(entries):
    pt_handles = [
        Line2D([0], [0], color=_color(e['mSum']), lw=2,
               label=f"{e['key']}{'*' if e['src']=='holdout' else ''}")
        for e in entries
    ]
    style_handles = [
        Line2D([0], [0], color='k', lw=2,   ls='-',  label='Truth (trained)'),
        Line2D([0], [0], color='k', lw=1.8, ls='--', label='Truth (holdout *)'),
        Line2D([0], [0], color='k', lw=1.4, ls=':',  label='Generated'),
    ]
    return pt_handles + style_handles

# ── Helper: overlay one observable on one axis ────────────────────────────────
def _overlay(ax, entries, key_t, key_g, bins, xlabel,
             log_x=False, title=None):
    for e in entries:
        col = _color(e['mSum'])
        ls_t = '--' if e['src'] == 'holdout' else '-'
        bw = bins[1] - bins[0]
        bc = 0.5 * (bins[:-1] + bins[1:])
        ht, _ = np.histogram(e[key_t], bins=bins)
        hg, _ = np.histogram(e[key_g], bins=bins)
        ht = ht / (ht.sum() * bw + 1e-12)
        hg = hg / (hg.sum() * bw + 1e-12)
        ax.step(bc, ht, where='mid', color=col, lw=1.8, ls=ls_t)
        ax.step(bc, hg, where='mid', color=col, lw=1.4, ls=':',  alpha=0.9)
    if log_x: ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('Density', fontsize=8)
    if title: ax.set_title(title, fontsize=9)

# ── Figure 1: 4-panel overlay ─────────────────────────────────────────────────
print('Overlay figure ...')

def _safe_max(arrs, pct=99.5):
    v = np.concatenate([a[np.isfinite(a)] for a in arrs])
    return float(np.percentile(v, pct))

ht_hi  = _safe_max([e['HT_t']     for e in entries] + [e['HT_g']     for e in entries])
met_hi = _safe_max([e['MET_t']    for e in entries] + [e['MET_g']    for e in entries])
lpt_hi = _safe_max([e['leadpT_t'] for e in entries] + [e['leadpT_g'] for e in entries])
ht4_hi = _safe_max([e['HT4_t']   for e in entries] + [e['HT4_g']    for e in entries])

bins_HT    = np.linspace(0, ht_hi  * 1.05, 60)
bins_MET   = np.linspace(0, met_hi * 1.05, 60)
bins_leadpT= np.logspace(np.log10(1.0), np.log10(lpt_hi + 1), 60)
bins_HT4   = np.linspace(0, ht4_hi * 1.05, 60)

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle(
    'E008 mass-conditioning overlay — Truth (solid=trained, dashed=holdout*) vs Generated (dotted)\n'
    'Curves should fan out by mass if conditioning is working',
    fontsize=10
)

_overlay(axes1[0, 0], entries, 'HT_t',     'HT_g',     bins_HT,    r'Scalar $H_T$ [GeV]',      title=r'Scalar $H_T$')
_overlay(axes1[0, 1], entries, 'MET_t',    'MET_g',    bins_MET,   r'MET [GeV]',               title='Missing $E_T$')
_overlay(axes1[1, 0], entries, 'leadpT_t', 'leadpT_g', bins_leadpT,r'Leading particle $p_T$ [GeV]',
         log_x=True, title=r'Leading particle $p_T$')
_overlay(axes1[1, 1], entries, 'HT4_t',    'HT4_g',    bins_HT4,  r'Top-4 particle $\Sigma p_T$ [GeV]',
         title=r'Top-4 $\Sigma p_T$')

axes1[0, 0].legend(handles=_legend_handles(entries), fontsize=7, loc='upper right', ncol=2)

fig1.tight_layout()
out1 = os.path.join(args.out_dir, 'mass_conditioning_overlay.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f'  -> {out1}')

# ── Figure 2: W1 vs mX+mY scatter ────────────────────────────────────────────
print('W1 vs mass scatter ...')

obs_defs = [
    ('HT_t',     'HT_g',     r'$H_T$ W₁ [GeV]'),
    ('MET_t',    'MET_g',    r'MET W₁ [GeV]'),
    ('leadpT_t', 'leadpT_g', r'Lead $p_T$ W₁ [GeV]'),
    ('HT4_t',    'HT4_g',   r'Top-4 $p_T$ W₁ [GeV]'),
]

fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
fig2.suptitle(
    'W₁(truth, generated) vs. total mass — E008 conditioning diagnostic\n'
    'Circles = trained grid points,  Triangles = held-out points (*)',
    fontsize=10
)

for ax, (kt, kg, ylabel) in zip(axes2, obs_defs):
    for e in entries:
        col  = _color(e['mSum'])
        mark = 'v' if e['src'] == 'holdout' else 'o'
        t_arr = e[kt][np.isfinite(e[kt])]
        g_arr = e[kg][np.isfinite(e[kg])]
        if len(t_arr) < 2 or len(g_arr) < 2: continue
        w1 = _wass(t_arr, g_arr)
        ax.scatter(e['mSum'], w1, color=col, marker=mark, s=100, zorder=3,
                   edgecolors='k', linewidths=0.5)
        ax.annotate(e['key'], (e['mSum'], w1),
                    textcoords='offset points', xytext=(5, 3), fontsize=7)
    ax.set_xlabel(r'$m_X + m_Y$ [GeV]', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)

proxy = [Line2D([0],[0], marker='o', color='gray', ls='', ms=9,
                markeredgecolor='k', label='Trained'),
         Line2D([0],[0], marker='v', color='gray', ls='', ms=9,
                markeredgecolor='k', label='Holdout *')]
axes2[0].legend(handles=proxy, fontsize=8)

fig2.tight_layout()
out2 = os.path.join(args.out_dir, 'w1_vs_mass.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'  -> {out2}')

# ── Summary table ─────────────────────────────────────────────────────────────
print('\n=== W1 summary ===')
hdr = f"{'Point':>14}  {'Src':>8}  {'mSum':>6}  {'HT W1':>8}  {'MET W1':>8}  {'LpT W1':>8}  {'HT4 W1':>8}"
print(hdr)
print('-' * len(hdr))
for e in entries:
    def _w1(kt, kg):
        t = e[kt][np.isfinite(e[kt])]; g = e[kg][np.isfinite(e[kg])]
        return _wass(t, g) if len(t)>1 and len(g)>1 else float('nan')
    print(f"{e['key']:>14}  {e['src']:>8}  {e['mSum']:>6.0f}"
          f"  {_w1('HT_t','HT_g'):>8.2f}"
          f"  {_w1('MET_t','MET_g'):>8.2f}"
          f"  {_w1('leadpT_t','leadpT_g'):>8.2f}"
          f"  {_w1('HT4_t','HT4_g'):>8.2f}")
