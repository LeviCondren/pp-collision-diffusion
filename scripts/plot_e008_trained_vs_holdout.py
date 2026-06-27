#!/usr/bin/env python3
"""
Diagnostic: compare MET and jet-mass quality at TRAINED vs HELD-OUT grid points.

Loads two NPZ directories:
  --holdout_dir   infer_holdout_ep019_5k/   (the 4 held-out mass points)
  --trained_dir   infer_trained_ep019_5k/   (4 nearby trained mass points)

Produces:
  1. Side-by-side MET distributions (trained vs held-out, 2×4 grid)
  2. Leading jet mass distributions
  3. Sub-leading jet mass distributions
  4. Per-particle eta and pT (sanity check)
  5. Summary table: W1 for MET, m_J1, m_J2 at every point

Usage:
    module load tensorflow/2.15.0   # provides pyjet
    python3 plot_e008_trained_vs_holdout.py \\
        --holdout_dir .../infer_holdout_ep019_5k \\
        --trained_dir .../infer_trained_ep019_5k \\
        --out_dir     .../plots_ep019_trained_vs_holdout
"""

import os, sys, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance as _wass
from pyjet import cluster, DTYPE_PTEPM

p = argparse.ArgumentParser()
p.add_argument('--holdout_dir', default=(
    '/pscratch/sd/l/lcondren/MCsim/wprime_signal/'
    'checkpoints_bsm_grid/bsm_grid/infer_holdout_ep019_5k'))
p.add_argument('--trained_dir', default=(
    '/pscratch/sd/l/lcondren/MCsim/wprime_signal/'
    'checkpoints_bsm_grid/bsm_grid/infer_trained_ep019_5k'))
p.add_argument('--out_dir', default=None)
p.add_argument('--n_events', type=int, default=5000)
args = p.parse_args()

OUT_DIR = args.out_dir or os.path.join(
    os.path.dirname(args.trained_dir), 'plots_ep019_trained_vs_holdout')
os.makedirs(OUT_DIR, exist_ok=True)
N = args.n_events

# ── Load NPZs ─────────────────────────────────────────────────────────────────

def load_dir(d):
    files = sorted(glob.glob(os.path.join(d, 'bsm_mX*.npz')))
    assert files, f'No bsm_mX*.npz in {d}'
    pts = {}
    for path in files:
        data = np.load(path)
        mX = float(data['mass_x']); mY = float(data['mass_y'])
        key = f'({int(mX)},{int(mY)})'
        pts[key] = {
            'parts_truth': data['parts_truth'][:N],
            'parts_gen':   data['parts_gen'][:N],
            'mask':        data['mask'][:N],
            'mask_gen':    data['mask_gen'][:N],
            'mX': mX, 'mY': mY,
        }
        print(f'  {key}: truth npart={data["mask"][:N].sum(1).mean():.1f}  '
              f'gen npart={data["mask_gen"][:N].sum(1).mean():.1f}')
    return pts

print('Loading holdout points:')
holdout = load_dir(args.holdout_dir)
print('Loading trained points:')
trained = load_dir(args.trained_dir)

# ── Helpers ───────────────────────────────────────────────────────────────────

R_JET = 0.4; PT_MIN = 20.0

def _obs_MET(parts, mask):
    pT = np.exp(np.clip(parts[:, :, 3], -10, 10)) * mask
    return np.sqrt((pT * parts[:, :, 2]).sum(1)**2 +
                   (pT * parts[:, :, 1]).sum(1)**2)

def _to_psj(parts_ev, mask_bool):
    p = parts_ev[mask_bool]; pT = np.exp(np.clip(p[:, 3], -10, 10))
    ok = np.isfinite(pT) & (pT > 0.01); p, pT = p[ok], pT[ok]
    arr = np.zeros(len(p), dtype=DTYPE_PTEPM)
    arr['pT'] = pT; arr['eta'] = p[:, 0]; arr['mass'] = 0.0
    arr['phi'] = np.arctan2(p[:, 1], p[:, 2]); return arr

def _jet_masses(parts_arr, mask_arr, n=None):
    """Return arrays of leading and sub-leading jet masses."""
    n = n or len(parts_arr)
    m1, m2 = [], []
    for i in range(min(n, len(parts_arr))):
        arr = _to_psj(parts_arr[i], mask_arr[i].astype(bool))
        if len(arr) < 2: continue
        jets = cluster(arr, R=R_JET, p=-1).inclusive_jets(ptmin=PT_MIN)
        jets.sort(key=lambda j: j.pt, reverse=True)
        if len(jets) >= 1: m1.append(jets[0].mass)
        if len(jets) >= 2: m2.append(jets[1].mass)
    return np.array(m1), np.array(m2)

def _w1(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0: return float('nan')
    return float(_wass(a, b))

COLORS = {'truth': '#1f77b4', 'gen': '#ff7f0e'}

# ── Compute all observables ───────────────────────────────────────────────────

def compute_obs(pts, label):
    results = {}
    for key, r in pts.items():
        print(f'  {label} {key}: clustering jets ...', flush=True)
        met_t = _obs_MET(r['parts_truth'], r['mask'])
        met_g = _obs_MET(r['parts_gen'],   r['mask_gen'])
        m1_t, m2_t = _jet_masses(r['parts_truth'], r['mask'])
        m1_g, m2_g = _jet_masses(r['parts_gen'],   r['mask_gen'])
        results[key] = {
            'met_t': met_t, 'met_g': met_g,
            'm1_t': m1_t, 'm1_g': m1_g,
            'm2_t': m2_t, 'm2_g': m2_g,
            'mX': r['mX'], 'mY': r['mY'],
            # for sanity check
            'eta_t': r['parts_truth'][r['mask'].astype(bool), 0],
            'eta_g': r['parts_gen'][r['mask_gen'].astype(bool), 0],
            'pT_t':  np.exp(np.clip(r['parts_truth'][r['mask'].astype(bool), 3], -10, 10)),
            'pT_g':  np.exp(np.clip(r['parts_gen'][r['mask_gen'].astype(bool), 3], -10, 10)),
        }
    return results

print('Computing observables...')
obs_holdout = compute_obs(holdout, 'holdout')
obs_trained = compute_obs(trained, 'trained')

# ── W1 table ─────────────────────────────────────────────────────────────────

print('\n' + '='*72)
print(f'{"Point":<14} {"Type":<10} {"MET W1":>10} {"m_J1 W1":>10} {"m_J2 W1":>10}')
print('-'*72)

table_rows = []
for label, obs_dict in [('trained', obs_trained), ('holdout', obs_holdout)]:
    for key, r in sorted(obs_dict.items()):
        w_met = _w1(r['met_t'], r['met_g'])
        w_m1  = _w1(r['m1_t'], r['m1_g'])
        w_m2  = _w1(r['m2_t'], r['m2_g'])
        print(f'{key:<14} {label:<10} {w_met:>10.2f} {w_m1:>10.2f} {w_m2:>10.2f}')
        table_rows.append((key, label, w_met, w_m1, w_m2))

print('='*72)

# ── Plot 1: MET distributions ─────────────────────────────────────────────────

print('\nPlot 1: MET ...')
all_pts  = sorted(set(list(obs_trained.keys()) + list(obs_holdout.keys())))
n_cols   = len(all_pts)

fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8), constrained_layout=True)
fig.suptitle(f'MET distribution — trained (top) vs held-out (bottom)\n'
             f'E008 checkpoint (~24 epochs), 5k events, 500 steps', fontsize=10)

for col, key in enumerate(all_pts):
    for row, (obs_dict, row_label) in enumerate(
            [(obs_trained, 'trained'), (obs_holdout, 'holdout')]):
        ax = axes[row, col]
        if key not in obs_dict:
            ax.text(0.5, 0.5, 'not run', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            ax.set_title(f'{key}\n({row_label})', fontsize=8)
            continue
        r = obs_dict[key]
        met_t, met_g = r['met_t'], r['met_g']
        hi = max(np.percentile(met_t, 99), np.percentile(met_g, 99)) * 1.05
        bins = np.linspace(0, hi, 60)
        ax.hist(met_t, bins=bins, density=True, histtype='step', lw=1.8,
                color=COLORS['truth'], label='Truth')
        ax.hist(met_g, bins=bins, density=True, histtype='step', lw=1.8,
                color=COLORS['gen'], label='Gen', ls='--')
        w = _w1(met_t, met_g)
        ax.set_xlabel('MET [GeV]', fontsize=8)
        ax.set_title(f'{key} ({row_label})\nW₁={w:.1f} GeV', fontsize=8)
        if col == 0 and row == 0: ax.legend(fontsize=7)

out = f'{OUT_DIR}/met_trained_vs_holdout.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'  -> {out}')

# ── Plot 2: Leading jet mass ──────────────────────────────────────────────────

print('Plot 2: leading jet mass ...')
fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8), constrained_layout=True)
fig.suptitle('Leading jet mass — trained (top) vs held-out (bottom)', fontsize=10)

for col, key in enumerate(all_pts):
    for row, (obs_dict, row_label) in enumerate(
            [(obs_trained, 'trained'), (obs_holdout, 'holdout')]):
        ax = axes[row, col]
        if key not in obs_dict:
            ax.text(0.5, 0.5, 'not run', ha='center', va='center',
                    transform=ax.transAxes); continue
        r = obs_dict[key]
        m1_t, m1_g = r['m1_t'], r['m1_g']
        ref = m1_t if len(m1_t) else m1_g
        if len(ref) == 0:
            ax.text(0.5, 0.5, 'no jets', ha='center', va='center',
                    transform=ax.transAxes); continue
        hi = min(float(np.percentile(ref, 99)) * 1.05, 250)
        bins = np.linspace(0, hi, 50)
        if len(m1_t): ax.hist(m1_t, bins=bins, density=True, histtype='step',
                               lw=1.8, color=COLORS['truth'], label='Truth')
        if len(m1_g): ax.hist(m1_g, bins=bins, density=True, histtype='step',
                               lw=1.8, color=COLORS['gen'], label='Gen', ls='--')
        w = _w1(m1_t, m1_g)
        ax.set_xlabel('Leading jet mass [GeV]', fontsize=8)
        ax.set_title(f'{key} ({row_label})\nW₁={w:.1f} GeV', fontsize=8)
        if col == 0 and row == 0: ax.legend(fontsize=7)

out = f'{OUT_DIR}/jet_mass_leading_trained_vs_holdout.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'  -> {out}')

# ── Plot 3: Sub-leading jet mass ──────────────────────────────────────────────

print('Plot 3: sub-leading jet mass ...')
fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8), constrained_layout=True)
fig.suptitle('Sub-leading jet mass — trained (top) vs held-out (bottom)', fontsize=10)

for col, key in enumerate(all_pts):
    for row, (obs_dict, row_label) in enumerate(
            [(obs_trained, 'trained'), (obs_holdout, 'holdout')]):
        ax = axes[row, col]
        if key not in obs_dict:
            ax.text(0.5, 0.5, 'not run', ha='center', va='center',
                    transform=ax.transAxes); continue
        r = obs_dict[key]
        m2_t, m2_g = r['m2_t'], r['m2_g']
        ref = m2_t if len(m2_t) else m2_g
        if len(ref) == 0:
            ax.text(0.5, 0.5, 'no jets', ha='center', va='center',
                    transform=ax.transAxes); continue
        hi = min(float(np.percentile(ref, 99)) * 1.05, 200)
        bins = np.linspace(0, hi, 50)
        if len(m2_t): ax.hist(m2_t, bins=bins, density=True, histtype='step',
                               lw=1.8, color=COLORS['truth'], label='Truth')
        if len(m2_g): ax.hist(m2_g, bins=bins, density=True, histtype='step',
                               lw=1.8, color=COLORS['gen'], label='Gen', ls='--')
        w = _w1(m2_t, m2_g)
        ax.set_xlabel('Sub-leading jet mass [GeV]', fontsize=8)
        ax.set_title(f'{key} ({row_label})\nW₁={w:.1f} GeV', fontsize=8)
        if col == 0 and row == 0: ax.legend(fontsize=7)

out = f'{OUT_DIR}/jet_mass_sublead_trained_vs_holdout.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'  -> {out}')

# ── Plot 4: Sanity check — eta and pT marginals ───────────────────────────────

print('Plot 4: eta/pT sanity check ...')
all_obs = list(obs_trained.items()) + list(obs_holdout.items())
fig, axes = plt.subplots(len(all_obs), 2,
                         figsize=(10, 3.5*len(all_obs)), constrained_layout=True)
if len(all_obs) == 1: axes = axes[np.newaxis, :]
fig.suptitle('Per-particle η and pT sanity check (all 8 points)', fontsize=10)

for row, (key, r) in enumerate(all_obs):
    typ = 'trained' if key in obs_trained else 'holdout'
    ax_eta, ax_pT = axes[row, 0], axes[row, 1]

    bins_eta = np.linspace(-5, 5, 55)
    ax_eta.hist(r['eta_t'], bins=bins_eta, density=True, histtype='step',
                lw=1.5, color=COLORS['truth'], label='Truth')
    ax_eta.hist(r['eta_g'], bins=bins_eta, density=True, histtype='step',
                lw=1.5, color=COLORS['gen'], label='Gen', ls='--')
    ax_eta.set_xlabel(r'Particle $\eta$', fontsize=8)
    ax_eta.set_ylabel(f'{key}\n({typ})', fontsize=7)
    ax_eta.set_title(f'W₁={_w1(r["eta_t"],r["eta_g"]):.4f}', fontsize=8)
    if row == 0: ax_eta.legend(fontsize=7)

    pT_t = r['pT_t'][np.isfinite(r['pT_t']) & (r['pT_t'] > 0)]
    pT_g = r['pT_g'][np.isfinite(r['pT_g']) & (r['pT_g'] > 0)]
    hi = max(np.percentile(pT_t, 99.5), np.percentile(pT_g, 99.5) if len(pT_g) else 1)
    bins_pT = np.logspace(np.log10(0.3), np.log10(hi+1), 55)
    ax_pT.hist(pT_t, bins=bins_pT, density=True, histtype='step',
               lw=1.5, color=COLORS['truth'])
    ax_pT.hist(pT_g, bins=bins_pT, density=True, histtype='step',
               lw=1.5, color=COLORS['gen'], ls='--')
    ax_pT.set_xscale('log'); ax_pT.set_xlabel(r'Particle $p_T$ [GeV]', fontsize=8)
    ax_pT.set_title(f'W₁={_w1(pT_t,pT_g):.3f} GeV', fontsize=8)

out = f'{OUT_DIR}/sanity_eta_pT.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'  -> {out}')

# ── Final summary table ───────────────────────────────────────────────────────

print('\n' + '='*72)
print('SUMMARY TABLE')
print(f'{"Point":<14} {"Type":<10} {"MET W1 [GeV]":>14} {"m_J1 W1 [GeV]":>14} {"m_J2 W1 [GeV]":>14}')
print('-'*72)
for key, label, w_met, w_m1, w_m2 in sorted(table_rows, key=lambda x: (x[1], x[0])):
    print(f'{key:<14} {label:<10} {w_met:>14.2f} {w_m1:>14.2f} {w_m2:>14.2f}')
print('='*72)
print(f'\nAll plots saved to {OUT_DIR}')
