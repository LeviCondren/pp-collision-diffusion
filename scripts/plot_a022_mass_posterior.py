#!/usr/bin/env python3
"""
Plot A022 mass posterior inference results.

Produces two figures:
  1. 2x2 grid of 2D ELBO likelihood heatmaps over the 12x12 mass hypothesis grid
  2. 4x2 grid of 1D profile likelihoods (marginalised over orthogonal mass axis)

Usage:
    python plot_a022_mass_posterior.py \
        --infer_dir /pub/lcondren/wprime_signal_mpi/mass_inference_e031_e054 \
        --out_dir   /pub/lcondren/wprime_signal_mpi/mass_inference_e031_e054
"""

import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p = argparse.ArgumentParser()
p.add_argument('--infer_dir', default='/pub/lcondren/wprime_signal_mpi/mass_inference_e031_e054')
p.add_argument('--out_dir',   default=None)
args = p.parse_args()
OUT_DIR = args.out_dir or args.infer_dir
os.makedirs(OUT_DIR, exist_ok=True)

GRID_VALS = np.array([50,100,150,200,250,300,350,400,450,500,550,600], dtype=float)
N_GRID    = len(GRID_VALS)
CMAP      = 'plasma'   # perceptually uniform, CVD-safe

# ── Load ─────────────────────────────────────────────────────────────────────
def load(path):
    d  = np.load(path)
    mx = d['mass_x']; my = d['mass_y']; ll = d['mean_log_likelihood']
    true_mx = float(d['obs_m_X']); true_my = float(d['obs_m_Y'])

    # Build grid[iy, ix] = LL at (m_X=GRID_VALS[ix], m_Y=GRID_VALS[iy])
    grid = np.full((N_GRID, N_GRID), np.nan)
    for ix, gx in enumerate(GRID_VALS):
        for iy, gy in enumerate(GRID_VALS):
            mask = (mx == gx) & (my == gy)
            if mask.any():
                grid[iy, ix] = ll[mask][0]

    idx_best  = np.argmax(ll)
    true_ll   = ll[(mx == true_mx) & (my == true_my)][0]
    rank      = int(np.sum(ll > true_ll)) + 1
    ll_range  = float(ll.max() - ll.min())

    return dict(
        grid=grid,
        true_mx=true_mx, true_my=true_my,
        pred_mx=float(mx[idx_best]), pred_my=float(my[idx_best]),
        true_ll=true_ll, best_ll=float(ll[idx_best]),
        delta_ll=float(ll[idx_best] - true_ll),
        rank=rank, n_hyp=len(ll), ll_range=ll_range,
        n_t=int(float(d['n_t'])), n_obs=int(float(d['n_obs'])),
    )

files = sorted(glob.glob(os.path.join(args.infer_dir, 'posterior_*.npz')))
assert files, f'No posterior_*.npz found in {args.infer_dir}'
data = [load(f) for f in files]

for r in data:
    hit = r['pred_mx'] == r['true_mx'] and r['pred_my'] == r['true_my']
    print(f"True ({r['true_mx']:.0f},{r['true_my']:.0f})  Argmax ({r['pred_mx']:.0f},{r['pred_my']:.0f})  "
          f"Rank {r['rank']}/{r['n_hyp']}  ΔLL={r['delta_ll']:.4f}  range={r['ll_range']:.4f}  "
          f"{'HIT' if hit else 'MISS'}")

# ── Shared style helpers ──────────────────────────────────────────────────────
GOLD  = '#f5c842'
RED   = '#e8432d'
BLUE  = '#1a56db'

def _tick_fmt(ax):
    ax.set_xticks(GRID_VALS[::2])
    ax.set_yticks(GRID_VALS[::2])
    ax.tick_params(labelsize=8)

# ── Figure 1: 2D heatmaps ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.patch.set_facecolor('#f8f9fc')
for ax in axes.flat:
    ax.set_facecolor('#f8f9fc')

fig.suptitle(
    'W′ mass posterior — ELBO likelihood surface\n'
    'E031 epoch 54/200 · E031 snapshot · n_obs=500 · n_t=25–50 · 144 hypotheses',
    fontsize=12, fontweight='bold', y=0.995, color='#111827'
)

for ax, r in zip(axes.flat, data):
    # Relative LL so argmax = 0; makes colorbar meaningful across panels
    g_rel = r['grid'] - r['best_ll']

    im = ax.pcolormesh(GRID_VALS, GRID_VALS, g_rel, cmap=CMAP,
                       vmin=np.nanmin(g_rel), vmax=0, rasterized=True)

    # True mass — gold star
    ax.scatter(r['true_mx'], r['true_my'], marker='*', s=320,
               color=GOLD, edgecolors='#333', linewidths=0.8, zorder=10)

    # Argmax — red diamond (only if different from true)
    correct = (r['pred_mx'] == r['true_mx'] and r['pred_my'] == r['true_my'])
    if not correct:
        ax.scatter(r['pred_mx'], r['pred_my'], marker='D', s=130,
                   color=RED, edgecolors='#333', linewidths=0.8, zorder=9)

    ax.set_xlabel(r'Hypothesized $m_X$ [GeV]', fontsize=9)
    ax.set_ylabel(r'Hypothesized $m_Y$ [GeV]', fontsize=9)
    _tick_fmt(ax)

    rank_color = '#2d7a2d' if correct else '#c0392b'
    ax.set_title(
        rf'True: $m_X$={r["true_mx"]:.0f}, $m_Y$={r["true_my"]:.0f} GeV'
        + f'\nArgmax: ({r["pred_mx"]:.0f}, {r["pred_my"]:.0f}) '
        + f' — Rank '
        + r'$\mathbf{' + str(r["rank"]) + r'}$'
        + f'/{r["n_hyp"]}   '
        + f'ΔLL = {r["delta_ll"]:.4f}',
        fontsize=9, color='#111827'
    )

    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label(r'$\Delta\log\mathcal{L}$ (rel. argmax)', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

# Shared legend
legend_handles = [
    Line2D([0],[0], marker='*', color='w', markerfacecolor=GOLD,
           markeredgecolor='#333', markersize=14, label='True mass'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor=RED,
           markeredgecolor='#333', markersize=9,  label='Argmax (if ≠ true)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=0.9)

fig.tight_layout(rect=[0, 0.05, 1, 0.98])
out1 = os.path.join(OUT_DIR, 'posterior_heatmaps.png')
fig.savefig(out1, dpi=150, bbox_inches='tight', facecolor='#f8f9fc')
plt.close(fig)
print(f'\nSaved: {out1}')

# ── Figure 2: 1D profile likelihoods ─────────────────────────────────────────
fig2, axes2 = plt.subplots(4, 2, figsize=(12, 14))
fig2.patch.set_facecolor('#f8f9fc')

fig2.suptitle(
    'Profile likelihoods — max over orthogonal axis\n'
    'E031 epoch 54/200',
    fontsize=12, fontweight='bold', y=1.002, color='#111827'
)

AXIS_LABELS = [r'$m_X$ hypothesis [GeV]', r'$m_Y$ hypothesis [GeV]']
COL_TITLES  = [r'Profile over $m_X$  (max$_{m_Y}$)', r'Profile over $m_Y$  (max$_{m_X}$)']

for row, r in enumerate(data):
    g = r['grid']
    # profile_mx[ix] = max_{m_Y} LL at m_X=GRID_VALS[ix]
    profile_mx = np.nanmax(g, axis=0)
    # profile_my[iy] = max_{m_X} LL at m_Y=GRID_VALS[iy]
    profile_my = np.nanmax(g, axis=1)

    label = (rf"True $m_X$={r['true_mx']:.0f}, "
             rf"$m_Y$={r['true_my']:.0f} GeV"
             f"   (rank {r['rank']}/{r['n_hyp']})")

    for col, (profile, true_val, xlab) in enumerate([
        (profile_mx, r['true_mx'], AXIS_LABELS[0]),
        (profile_my, r['true_my'], AXIS_LABELS[1]),
    ]):
        ax = axes2[row, col]
        ax.set_facecolor('#f8f9fc')

        ax.plot(GRID_VALS, profile, color=BLUE, lw=1.8, marker='o', ms=4,
                markerfacecolor='white', markeredgecolor=BLUE, markeredgewidth=1.2)

        # Shade under curve lightly
        ax.fill_between(GRID_VALS, np.nanmin(profile), profile,
                        color=BLUE, alpha=0.08)

        # True mass vertical line
        ax.axvline(true_val, color=GOLD, lw=2.0, ls='--', zorder=5,
                   label='True mass')

        # Profile argmax
        pmax_idx = int(np.nanargmax(profile))
        ax.scatter(GRID_VALS[pmax_idx], profile[pmax_idx], color=RED, s=80,
                   zorder=6, label='Argmax', edgecolors='#333', linewidths=0.7)

        ax.set_ylabel(r'Profile $\log\mathcal{L}$', fontsize=8)
        if row == 3:
            ax.set_xlabel(xlab, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, lw=0.35, alpha=0.45, color='#b0b8c8')

        if row == 0:
            ax.set_title(COL_TITLES[col], fontsize=10, fontweight='semibold', color='#111827')

        if col == 0:
            ax.text(0.02, 0.97, label, transform=ax.transAxes,
                    fontsize=7.5, va='top', ha='left', color='#222',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.75, ec='none'))

        if row == 0 and col == 0:
            ax.legend(fontsize=8, framealpha=0.85)

fig2.tight_layout()
out2 = os.path.join(OUT_DIR, 'posterior_profiles.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#f8f9fc')
plt.close(fig2)
print(f'Saved: {out2}')
