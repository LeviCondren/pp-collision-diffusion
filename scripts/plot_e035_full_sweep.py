#!/usr/bin/env python3
"""
plot_e035_full_sweep.py — Full inference plots for E035 CFG guidance scale sweep.

For every observable, produces one figure with:
    rows    = 4 holdout mass points  (250,250) / (250,300) / (300,250) / (300,300)
    columns = 6 guidance scales      s = 1.0 / 1.5 / 3.0 / 5.0 / 7.5 / 10.0

Truth (blue solid) vs Generated (orange dashed) overlaid; W₁ annotated per panel.

Figures written:
    01_eta.png              particle η
    02_logpT.png            particle log pT
    03_phi.png              particle φ
    04_charge.png           particle charge
    05_multiplicity.png     per-event particle count
    06_HT.png               scalar HT
    07_MET_particles.png    MET reconstructed from particle cloud
    08_sphericity.png       2D sphericity
    09_cone_mass_X.png      stage-1 cone_mass_X (physical GeV)
    10_cone_mass_Y.png      stage-1 cone_mass_Y (physical GeV)
    11_cone_pT_X.png        stage-1 cone_pT_X
    12_cone_pT_Y.png        stage-1 cone_pT_Y
    13_MET_stage1.png       stage-1 log1p(MET) (physical GeV)

Usage:
    python3 plot_e035_full_sweep.py \\
        --sweep_dir  .../infer_cfg_sweep \\
        --stats_path .../normalisation_stats_event_c_stage1_cfg.json \\
        --out_dir    .../plots_e035
"""

import os, sys, json, argparse
import numpy as np
from scipy.stats import wasserstein_distance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────

SCALES     = [1.0, 1.5, 3.0, 5.0, 7.5, 10.0]
SCALE_STRS = ['1p0', '1p5', '3p0', '5p0', '7p5', '10p0']
HOLDOUTS   = [(250, 250), (250, 300), (300, 250), (300, 300)]

C_TRUTH = '#1f77b4'
C_GEN   = '#ff7f0e'

# Stage-1 event feature order (index into jets_gen[:, 1:] / event_feat_truth)
# ['log1p(MET)', 'sin(MET_phi)', 'cos(MET_phi)',
#  'log1p(cone_pT_X)', 'log1p(cone_mass_X)',
#  'log1p(cone_pT_Y)', 'log1p(cone_mass_Y)']
IDX_MET_LOG = 0
IDX_CPT_X   = 3
IDX_CM_X    = 4
IDX_CPT_Y   = 5
IDX_CM_Y    = 6


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep_dir',  required=True)
    p.add_argument('--stats_path', required=True)
    p.add_argument('--out_dir',    default=None)
    p.add_argument('--n_max',      type=int, default=5000)
    return p.parse_args()


args   = _parse()
out_dir = args.out_dir or os.path.join(os.path.dirname(args.sweep_dir), 'plots_e035')
os.makedirs(out_dir, exist_ok=True)

# ── Normalisation stats ───────────────────────────────────────────────────────

with open(args.stats_path) as fh:
    stats = json.load(fh)
ev_mean = np.array(stats['jet_mean'], dtype=np.float32)[1:]   # (7,)
ev_std  = np.array(stats['jet_std'],  dtype=np.float32)[1:]


# ── Helper: W1 + JSD score annotation ────────────────────────────────────────

def _w1(a, b):
    try:
        return float(wasserstein_distance(a, b))
    except Exception:
        return float('nan')


def _score(ax, w1, pos=(0.97, 0.97)):
    ax.text(*pos, f'W₁={w1:.3g}',
            transform=ax.transAxes, fontsize=6.5,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, ec='none'))


# ── Load all NPZs into per-(scale, mass) dicts of 1-D observable arrays ──────

print('Loading NPZs and computing observables...')

# We'll store 1-D arrays for every observable for every (scale, mx, my)
# Keys: (scale_float, mx, my)
obs = {}   # (s, mx, my) -> dict of arrays

for scale, sstr in zip(SCALES, SCALE_STRS):
    sdir = os.path.join(args.sweep_dir, f's{sstr}')
    for mx, my in HOLDOUTS:
        path = os.path.join(sdir, f'bsm_mX{mx:04d}_mY{my:04d}_rank00_of01.npz')
        if not os.path.exists(path):
            print(f'  MISSING: s={scale} ({mx},{my})')
            obs[(scale, mx, my)] = None
            continue

        d = np.load(path)
        n = min(args.n_max, len(d['parts_truth']))

        pt  = d['parts_truth'][:n]          # (N, npart, 6)
        pg  = d['parts_gen'][:n]
        mt  = d['mask'][:n].astype(bool)    # (N, npart)
        mg  = d['mask_gen'][:n].astype(bool)
        eft = d['event_feat_truth'][:n]      # (N, 7) normalised
        jg  = d['jets_gen'][:n]             # (N, 8) normalised

        # Denorm event features
        ef_t = eft * ev_std + ev_mean        # (N, 7) physical log-space
        ef_g = jg[:, 1:] * ev_std + ev_mean

        # Physical cone masses / pTs
        def _expm1c(x): return np.expm1(np.clip(x, 0, 15))

        # Particle-level 1-D arrays (flattened over valid particles)
        def _flat(arr, mask): return arr[mask]

        eta_t  = _flat(pt[:, :, 0], mt)
        eta_g  = _flat(pg[:, :, 0], mg)
        pT_t   = np.exp(np.clip(_flat(pt[:, :, 3], mt), -10, 10))
        pT_g_raw = np.exp(np.clip(_flat(pg[:, :, 3], mg), -10, 10))
        pT_g   = pT_g_raw[np.isfinite(pT_g_raw) & (pT_g_raw > 0)]
        phi_t  = np.arctan2(_flat(pt[:, :, 1], mt), _flat(pt[:, :, 2], mt))
        phi_g  = np.arctan2(_flat(pg[:, :, 1], mg), _flat(pg[:, :, 2], mg))
        chg_t  = _flat(pt[:, :, 5], mt)
        chg_g  = _flat(pg[:, :, 5], mg)

        # Event-level observables computed from particle cloud
        def _pT_ev(p, m):
            return np.exp(np.clip(p[:, :, 3], -10, 10)) * m

        pTe_t  = _pT_ev(pt, mt)
        pTe_g  = _pT_ev(pg, mg)
        mult_t = mt.sum(axis=1).astype(float)
        mult_g = mg.sum(axis=1).astype(float)
        HT_t   = pTe_t.sum(axis=1)
        HT_g   = pTe_g.sum(axis=1)

        MET_t  = np.sqrt(
            (pTe_t * pt[:, :, 2]).sum(1)**2 +
            (pTe_t * pt[:, :, 1]).sum(1)**2)
        MET_g  = np.sqrt(
            (pTe_g * pg[:, :, 2]).sum(1)**2 +
            (pTe_g * pg[:, :, 1]).sum(1)**2)

        def _sph(p, m):
            pT  = _pT_ev(p, m)
            px  = pT * p[:, :, 2]; py = pT * p[:, :, 1]
            Sxx = (px**2).sum(1); Syy = (py**2).sum(1); Sxy = (px*py).sum(1)
            denom = np.clip(Sxx + Syy, 1e-8, None)
            det  = (Sxx*Syy - Sxy**2) / denom**2
            lam  = (1. - np.sqrt(np.clip(1. - 4.*det, 0., None))) / 2.
            return np.clip(2.*lam, 0., 1.)

        sph_t = _sph(pt, mt)
        sph_g = _sph(pg, mg)

        obs[(scale, mx, my)] = dict(
            # particle-level
            eta_t=eta_t, eta_g=eta_g,
            pT_t=pT_t,   pT_g=pT_g,
            phi_t=phi_t, phi_g=phi_g,
            chg_t=chg_t, chg_g=chg_g,
            # event-level from cloud
            mult_t=mult_t, mult_g=mult_g,
            HT_t=HT_t,   HT_g=HT_g,
            MET_t=MET_t,  MET_g=MET_g,
            sph_t=sph_t,  sph_g=sph_g,
            # stage-1 event features (physical log-space)
            ef_t=ef_t, ef_g=ef_g,
        )
        print(f'  s={scale:4.1f}  ({mx},{my})  '
              f'npart_t={mult_t.mean():.0f}  npart_g={mult_g.mean():.0f}')

print('Done loading.\n')


# ── Generic grid plotter ──────────────────────────────────────────────────────

def make_grid(key_t, key_g, bins_fn, xlabel, title, fname,
              log_x=False, xscale_log=False):
    """
    One figure: rows=mass points, cols=scales.
    bins_fn(truth_array) -> bin edges (uses truth from s=1.0 as reference).
    """
    n_rows = len(HOLDOUTS)
    n_cols = len(SCALES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 2.8 * n_rows),
                             sharex='row', sharey='row')

    fig.suptitle(
        f'E035 — {title}  |  E034 epoch 43  |  5k events per point\n'
        f'Truth (blue) vs Generated (orange)  —  W₁ annotated',
        fontsize=9, y=1.01)

    for row, (mx, my) in enumerate(HOLDOUTS):
        # Reference bins from s=1.0 truth
        ref = obs.get((1.0, mx, my))
        bins = bins_fn(ref[key_t] if ref else np.array([0., 1.])) if ref else np.linspace(0, 1, 30)

        for col, scale in enumerate(SCALES):
            ax = axes[row, col]
            r  = obs.get((scale, mx, my))

            if ref is not None:
                ax.hist(ref[key_t], bins=bins, density=True,
                        histtype='step', lw=1.5, color=C_TRUTH, label='Truth')

            if r is not None:
                arr_g = r[key_g]
                ax.hist(arr_g, bins=bins, density=True,
                        histtype='step', lw=1.5, color=C_GEN,
                        linestyle='--', label='Gen')
                w1 = _w1(ref[key_t], arr_g) if ref is not None else float('nan')
                _score(ax, w1)

            if xscale_log:
                ax.set_xscale('log')
            if row == 0:
                ax.set_title(f's = {scale}', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'({mx},{my})\nDensity', fontsize=7)
            if row == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=7)
            ax.tick_params(labelsize=6)
            if row == 0 and col == 0:
                ax.legend(fontsize=6, frameon=False)

    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def _pct_bins(arr, lo=0.5, hi=99.5, n=60):
    a, b = np.percentile(arr, lo), np.percentile(arr, hi)
    return np.linspace(a, b, n + 1)

def _log_bins(arr, lo=0.5, hi=99.5, n=60):
    a = max(np.percentile(arr, lo), 1e-3)
    b = np.percentile(arr, hi)
    return np.logspace(np.log10(a), np.log10(max(b, a*10)), n + 1)


# ── Stage-1 event feature grid helper ────────────────────────────────────────

def make_ev_grid(feat_idx, transform, xlabel, title, fname):
    """Grid for a stage-1 event feature (denorm → physical via transform)."""
    n_rows = len(HOLDOUTS)
    n_cols = len(SCALES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 2.8 * n_rows),
                             sharex='row', sharey='row')
    fig.suptitle(
        f'E035 — {title}  |  E034 epoch 43  |  Stage-1 predictions\n'
        f'Truth (blue) vs Generated (orange)  —  W₁ annotated',
        fontsize=9, y=1.01)

    for row, (mx, my) in enumerate(HOLDOUTS):
        ref = obs.get((1.0, mx, my))
        ref_arr = transform(ref['ef_t'][:, feat_idx]) if ref else np.array([0., 1.])
        bins = _pct_bins(ref_arr)

        for col, scale in enumerate(SCALES):
            ax = axes[row, col]
            r  = obs.get((scale, mx, my))

            if ref is not None:
                ax.hist(ref_arr, bins=bins, density=True,
                        histtype='step', lw=1.5, color=C_TRUTH, label='Truth')

            if r is not None:
                gen_arr = transform(r['ef_g'][:, feat_idx])
                ax.hist(gen_arr, bins=bins, density=True,
                        histtype='step', lw=1.5, color=C_GEN,
                        linestyle='--', label='Gen')
                w1 = _w1(ref_arr, gen_arr) if ref is not None else float('nan')
                _score(ax, w1)

            if row == 0:
                ax.set_title(f's = {scale}', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'({mx},{my})\nDensity', fontsize=7)
            if row == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=7)
            ax.tick_params(labelsize=6)
            if row == 0 and col == 0:
                ax.legend(fontsize=6, frameon=False)

    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ── Generate all figures ──────────────────────────────────────────────────────

print('Generating figures...\n')

# 01 — particle η
make_grid('eta_t', 'eta_g',
          lambda a: np.linspace(-5, 5, 61),
          r'Particle $\eta$', r'Particle $\eta$',
          '01_eta.png')

# 02 — particle log pT (plot pT on log scale)
make_grid('pT_t', 'pT_g',
          lambda a: _log_bins(a[a > 0], lo=1, hi=99, n=60),
          r'Particle $p_T$ [GeV]', r'Particle $p_T$ spectrum',
          '02_pT.png', xscale_log=True)

# 03 — particle φ
make_grid('phi_t', 'phi_g',
          lambda a: np.linspace(-np.pi, np.pi, 61),
          r'Particle $\phi$ [rad]', r'Particle $\phi$',
          '03_phi.png')

# 04 — particle charge
make_grid('chg_t', 'chg_g',
          lambda a: np.linspace(-2, 2, 41),
          'Particle charge', 'Particle charge',
          '04_charge.png')

# 05 — multiplicity
make_grid('mult_t', 'mult_g',
          lambda a: np.linspace(max(0, a.min()-5), a.max()+5, 61),
          'Particle multiplicity', 'Event particle multiplicity',
          '05_multiplicity.png')

# 06 — scalar HT
make_grid('HT_t', 'HT_g',
          lambda a: _pct_bins(a, lo=1, hi=99),
          r'Scalar $H_T$ [GeV]', r'Scalar $H_T$',
          '06_HT.png')

# 07 — MET from particle cloud
make_grid('MET_t', 'MET_g',
          lambda a: _pct_bins(a, lo=0.5, hi=99.5),
          'MET [GeV]  (from particle cloud)', 'MET (particle cloud sum)',
          '07_MET_particles.png')

# 08 — sphericity
make_grid('sph_t', 'sph_g',
          lambda a: np.linspace(0, 1, 51),
          r'Sphericity $S_T$', r'2-D Sphericity $S_T$',
          '08_sphericity.png')

# 09 — cone mass X  (stage-1 feature)
make_ev_grid(IDX_CM_X, np.expm1,
             r'Cone mass $m_X$ [GeV]', r'Stage-1 cone mass $m_X$',
             '09_cone_mass_X.png')

# 10 — cone mass Y  (stage-1 feature)
make_ev_grid(IDX_CM_Y, np.expm1,
             r'Cone mass $m_Y$ [GeV]', r'Stage-1 cone mass $m_Y$',
             '10_cone_mass_Y.png')

# 11 — cone pT X
make_ev_grid(IDX_CPT_X, np.expm1,
             r'Cone $p_T^X$ [GeV]', r'Stage-1 cone $p_T^X$',
             '11_cone_pT_X.png')

# 12 — cone pT Y
make_ev_grid(IDX_CPT_Y, np.expm1,
             r'Cone $p_T^Y$ [GeV]', r'Stage-1 cone $p_T^Y$',
             '12_cone_pT_Y.png')

# 13 — stage-1 MET (log1p space → physical)
make_ev_grid(IDX_MET_LOG, np.expm1,
             r'Stage-1 MET [GeV]', r'Stage-1 MET (log1p decoded)',
             '13_MET_stage1.png')

print(f'\nAll figures written to {out_dir}')
