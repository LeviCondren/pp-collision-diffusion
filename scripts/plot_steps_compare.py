#!/usr/bin/env python3
"""
Compare 50-step vs 500-step DDPM inference quality.
Overlays truth / gen@50steps / gen@500steps on the same axes.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance as _wass

BASE = '/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/checkpoints_pet_pp/parton_v1_1node'
DIR_50  = f'{BASE}/infer_steps_compare_50'
DIR_500 = f'{BASE}/infer_steps_compare'
OUT_DIR = f'{BASE}/plots_steps_compare'
os.makedirs(OUT_DIR, exist_ok=True)

PROCS = ['dijet', 'zjets']
PROC_LABEL = {'dijet': 'Dijet', 'zjets': 'Z+jets'}

C_TRUTH = '#1f77b4'
C_50    = '#ff7f0e'
C_500   = '#2ca02c'

# ── helpers ───────────────────────────────────────────────────────────────────
def _jsd(a, b, bins):
    eps = 1e-10
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    ha = ha.astype(float) + eps;  hb = hb.astype(float) + eps
    ha /= ha.sum();                hb /= hb.sum()
    m = 0.5 * (ha + hb)
    return float(np.clip(0.5*(ha*np.log(ha/m)).sum() + 0.5*(hb*np.log(hb/m)).sum(), 0, None))

def _score(ax, jsd50, w1_50, jsd500, w1_500, units=''):
    u = f' {units}' if units else ''
    txt = (f'50-step:  JSD={jsd50:.4f}  W₁={w1_50:.3g}{u}\n'
           f'500-step: JSD={jsd500:.4f}  W₁={w1_500:.3g}{u}')
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, fontsize=6.5,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='none'))

def _load(directory, proc):
    path = f'{directory}/{proc}_rank00_of01.npz'
    d = np.load(path)
    return {
        'parts_truth': d['parts_truth'],
        'parts_gen':   d['parts_gen'],
        'mask':        d['mask'][:, :, None].astype(np.float32) if d['mask'].ndim == 2 else d['mask'].astype(np.float32),
        'mask_gen':    d['mask_gen'][:, :, None].astype(np.float32) if d['mask_gen'].ndim == 2 else d['mask_gen'].astype(np.float32),
    }

data = {}
for proc in PROCS:
    data[proc] = {
        'truth_ref': _load(DIR_50,  proc),   # truth is the same in both; use one
        'd50':       _load(DIR_50,  proc),
        'd500':      _load(DIR_500, proc),
    }
    n50  = len(data[proc]['d50']['parts_truth'])
    n500 = len(data[proc]['d500']['parts_truth'])
    print(f'{proc}: {n50} events (50-step), {n500} events (500-step)')

# ── Plot 1: Particle-level distributions ─────────────────────────────────────
print('Plotting particle distributions ...')
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle('50-step vs 500-step DDPM — Particle distributions  (1000 events each)', fontsize=11)

OBS_PART = [
    (r'$\eta$',        'eta',  lambda p, m: p[m, 0],                       np.linspace(-5, 5, 60),    ''),
    (r'$\phi$ [rad]',  'phi',  lambda p, m: np.arctan2(p[m, 1], p[m, 2]),  np.linspace(-np.pi, np.pi, 60), 'rad'),
    (r'$p_T$ [GeV]',   'pT',   lambda p, m: np.exp(np.clip(p[m, 3], -10, 10)), None, 'GeV'),
    ('Charge',         'chg',  lambda p, m: p[m, 5],                       np.linspace(-2, 2, 41),    ''),
]

for row, proc in enumerate(PROCS):
    r50  = data[proc]['d50']
    r500 = data[proc]['d500']
    m_t   = r50['mask'][:, :, 0].astype(bool)
    m_g50 = r50['mask_gen'][:, :, 0].astype(bool)
    m_g500= r500['mask_gen'][:, :, 0].astype(bool)

    for col, (xlabel, key, fn, bins, units) in enumerate(OBS_PART):
        ax = axes[row, col]
        t   = fn(r50['parts_truth'], m_t)
        g50 = fn(r50['parts_gen'],   m_g50)
        g50 = g50[np.isfinite(g50)]
        g500= fn(r500['parts_gen'],  m_g500)
        g500= g500[np.isfinite(g500)]

        if bins is None:
            hi = max(np.percentile(t, 99), np.percentile(g50, 99) if len(g50) else 1,
                     np.percentile(g500, 99) if len(g500) else 1)
            bins = np.logspace(np.log10(0.3), np.log10(hi + 1), 60)
            ax.set_xscale('log')

        ax.hist(t,    bins=bins, density=True, histtype='step', lw=2.0, color=C_TRUTH, label='Truth')
        ax.hist(g50,  bins=bins, density=True, histtype='step', lw=1.5, color=C_50,    label='Gen 50-step',  ls='--')
        ax.hist(g500, bins=bins, density=True, histtype='step', lw=1.5, color=C_500,   label='Gen 500-step', ls=':')
        ax.set_xlabel(xlabel, fontsize=9)
        if col == 0:
            ax.set_ylabel(PROC_LABEL[proc], fontsize=10)
        if row == 0:
            ax.legend(fontsize=7)
        jsd50  = _jsd(t, g50,  bins)
        jsd500 = _jsd(t, g500, bins)
        w50  = _wass(t, g50)  if len(g50)  > 1 else float('nan')
        w500 = _wass(t, g500) if len(g500) > 1 else float('nan')
        _score(ax, jsd50, w50, jsd500, w500, units)

fig.tight_layout()
out = f'{OUT_DIR}/particle_dists_compare.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {out}')

# ── Plot 2: Global event observables ─────────────────────────────────────────
print('Plotting global event observables ...')

def _pT_m(parts, mask): return np.exp(np.clip(parts[:, :, 3], -10, 10)) * mask[:, :, 0]
def _mult(mask):  return mask[:, :, 0].sum(axis=1)
def _HT(p, m):    return _pT_m(p, m).sum(axis=1)
def _MET(p, m):
    pT = _pT_m(p, m)
    return np.sqrt((pT * p[:, :, 2]).sum(1)**2 + (pT * p[:, :, 1]).sum(1)**2)
def _sph(p, m):
    pT = _pT_m(p, m); px = pT*p[:, :, 2]; py = pT*p[:, :, 1]
    Sxx = (px**2).sum(1); Syy = (py**2).sum(1); Sxy = (px*py).sum(1)
    d = np.clip(Sxx+Syy, 1e-8, None)
    det = (Sxx*Syy - Sxy**2) / d**2
    lam = (1. - np.sqrt(np.clip(1. - 4.*det, 0, None))) / 2.
    sph = np.clip(2.*lam, 0, 1); sph[d < 1e-6] = 0.0; return sph
def _eta_flat(p, m): return p[:, :, 0][m[:, :, 0] > 0]

OBS_EVT = [
    ('Multiplicity',   lambda p, m: _mult(m),  None,  ''),
    ('HT [GeV]',       _HT,                    None,  'GeV'),
    ('MET [GeV]',      _MET,                   None,  'GeV'),
    (r'Sphericity $S_T$', _sph,                np.linspace(0, 1, 50), ''),
    (r'Flat $\eta$',   _eta_flat,              np.linspace(-5, 5, 55), ''),
]

fig, axes = plt.subplots(2, 5, figsize=(25, 8))
fig.suptitle('50-step vs 500-step DDPM — Global event observables  (1000 events each)', fontsize=11)

for row, proc in enumerate(PROCS):
    r50  = data[proc]['d50']
    r500 = data[proc]['d500']

    for col, (xlabel, fn, bins, units) in enumerate(OBS_EVT):
        ax = axes[row, col]
        t    = fn(r50['parts_truth'], r50['mask'])
        g50  = fn(r50['parts_gen'],   r50['mask_gen'])
        g500 = fn(r500['parts_gen'],  r500['mask_gen'])
        t    = t[np.isfinite(t)];  g50 = g50[np.isfinite(g50)];  g500 = g500[np.isfinite(g500)]

        if bins is None:
            all_v = np.concatenate([t, g50, g500])
            if fn == _mult:
                bins = np.arange(0, int(all_v.max()) + 2, max(1, int(all_v.max())//60))
            else:
                hi = np.percentile(all_v, 99)
                bins = np.linspace(0, max(hi, 1e-3), 60)

        ax.hist(t,    bins=bins, density=True, histtype='step', lw=2.0, color=C_TRUTH, label='Truth')
        ax.hist(g50,  bins=bins, density=True, histtype='step', lw=1.5, color=C_50,    label='Gen 50-step',  ls='--')
        ax.hist(g500, bins=bins, density=True, histtype='step', lw=1.5, color=C_500,   label='Gen 500-step', ls=':')
        ax.set_xlabel(xlabel, fontsize=9)
        if col == 0:
            ax.set_ylabel(PROC_LABEL[proc], fontsize=10)
        if row == 0:
            ax.set_title(xlabel, fontsize=9)
            ax.legend(fontsize=7)
        if len(g50) > 1 and len(g500) > 1:
            _score(ax, _jsd(t, g50, bins), _wass(t, g50),
                       _jsd(t, g500, bins), _wass(t, g500), units)

fig.tight_layout()
out = f'{OUT_DIR}/global_obs_compare.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {out}')

print(f'\nDone. Plots saved to {OUT_DIR}')
