#!/usr/bin/env python3
"""
Diagnostic plots and Wasserstein comparison for MET post-processing (E024).

Computes before/after Wasserstein distances for all required observables and
produces histogram overlays (truth, generated, generated+pp).

Usage:
  python3 plot_met_postprocessing.py \
      --infer_dir     .../bsm_grid_event_c/infer_holdout_truth \
      --infer_pp_dir  .../bsm_grid_event_c/infer_holdout_truth_metpp \
      --out_dir       .../figures/E024_met_postprocessing
"""

import argparse, glob, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

R_CONE = 1.0

# ── Observable definitions ─────────────────────────────────────────────────────
# (key, xlabel, bins_fn)
_CATALOGUE_EVT = [
    ('MET_magnitude', r'MET magnitude [GeV]',
     lambda t, g: np.linspace(0, max(np.percentile(t,99), np.percentile(g,99))*1.1, 60)),
    ('MET_phi',       r'MET $\phi$ [rad]',
     lambda t, g: np.linspace(-np.pi, np.pi, 60)),
    ('cone_pT_X',     r'Cone $p_T$ X [GeV]',
     lambda t, g: np.linspace(0, max(np.percentile(t,99), np.percentile(g,99))*1.1, 60)),
    ('cone_pT_Y',     r'Cone $p_T$ Y [GeV]',
     lambda t, g: np.linspace(0, max(np.percentile(t,99), np.percentile(g,99))*1.1, 60)),
    ('cone_mass_X',   r'Cone mass X [GeV]',
     lambda t, g: np.linspace(0, max(np.percentile(t,99), np.percentile(g,99))*1.1, 60)),
    ('cone_mass_Y',   r'Cone mass Y [GeV]',
     lambda t, g: np.linspace(0, max(np.percentile(t,99), np.percentile(g,99))*1.1, 60)),
    ('multiplicity',  r'Event multiplicity (N valid)',
     lambda t, g: np.arange(int(min(t.min(), g.min())), int(max(t.max(), g.max()))+2)),
]
_CATALOGUE_PART = [
    ('eta_particle',   r'Particle $\eta$',
     lambda t, g: np.linspace(-5, 5, 60)),
    ('logpT_particle', r'Particle $\log p_T$ [GeV]',
     lambda t, g: np.linspace(
         max(-3, min(np.percentile(t[np.isfinite(t)], 1), np.percentile(g[np.isfinite(g)], 1))),
         min(8,  max(np.percentile(t[np.isfinite(t)], 99), np.percentile(g[np.isfinite(g)], 99))),
         60)),
    ('phi_particle',   r'Particle $\phi$ [rad]',
     lambda t, g: np.linspace(-np.pi, np.pi, 60)),
]


# ── Physics helpers ────────────────────────────────────────────────────────────

def met_xy(parts, mask):
    pT = np.exp(parts[:, :, 3]) * mask
    return (pT * parts[:, :, 2]).sum(1), (pT * parts[:, :, 1]).sum(1)


def cone_obs(parts, mask, parton_feat, boson_slot):
    float_mask = mask.astype(np.float32)
    valid = mask.astype(bool)
    pT  = np.exp(np.clip(parts[:, :, 3], -10, 10)) * float_mask
    sp  = parts[:, :, 1]
    cp  = parts[:, :, 2]
    eta = np.clip(parts[:, :, 0], -8, 8)
    phi = np.arctan2(sp, cp)

    pze   = np.clip(parton_feat[:, boson_slot, 3], -1+1e-7, 1-1e-7)
    eta_p = 0.5 * np.log((1+pze)/(1-pze))
    phi_p = np.arctan2(parton_feat[:, boson_slot, 1], parton_feat[:, boson_slot, 2])

    deta = eta - eta_p[:, None]
    dphi = (phi - phi_p[:, None] + np.pi) % (2*np.pi) - np.pi
    in_c = (np.sqrt(deta**2 + dphi**2) < R_CONE) & valid

    wt   = pT * in_c
    pT_c = wt.sum(1)
    E_c  = (wt * np.cosh(eta)).sum(1)
    px_c = (wt * cp).sum(1)
    py_c = (wt * sp).sum(1)
    pz_c = (wt * np.sinh(eta)).sum(1)
    m2   = np.maximum(E_c**2 - px_c**2 - py_c**2 - pz_c**2, 0.0)
    return pT_c, np.sqrt(m2)


def compute_all_obs(parts, mask, parton_feat):
    MET_x, MET_y = met_xy(parts, mask)
    obs = {
        'MET_magnitude': np.sqrt(MET_x**2 + MET_y**2),
        'MET_phi':       np.arctan2(MET_y, MET_x),
        'multiplicity':  mask.sum(axis=1),
    }
    pT_X, m_X = cone_obs(parts, mask, parton_feat, boson_slot=2)
    pT_Y, m_Y = cone_obs(parts, mask, parton_feat, boson_slot=3)
    obs['cone_pT_X']   = pT_X
    obs['cone_pT_Y']   = pT_Y
    obs['cone_mass_X'] = m_X
    obs['cone_mass_Y'] = m_Y

    valid = mask.astype(bool)
    obs['eta_particle']   = parts[:, :, 0][valid]
    obs['logpT_particle'] = parts[:, :, 3][valid]
    phi_flat = np.arctan2(parts[:, :, 1][valid], parts[:, :, 2][valid])
    obs['phi_particle']   = phi_flat
    return obs


def rel_w1(gen, truth):
    r = np.ptp(truth)
    return wasserstein_distance(gen, truth) / r if r > 0 else np.nan


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--infer_dir',    required=True)
    pa.add_argument('--infer_pp_dir', required=True)
    pa.add_argument('--out_dir',      required=True)
    pa.add_argument('--n_events', type=int, default=5000)
    args = pa.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    npz_files = sorted(glob.glob(
        os.path.join(args.infer_dir, 'bsm_mX*_rank00_of01.npz')))
    print(f'Found {len(npz_files)} mass point(s)')

    all_rows = []

    for npz_path in npz_files:
        fname   = os.path.basename(npz_path)
        parts_s = fname.replace('_rank00_of01.npz', '').split('_')
        mX      = int(parts_s[1].replace('mX', '').lstrip('0') or '0')
        mY      = int(parts_s[2].replace('mY', '').lstrip('0') or '0')
        label   = f'mX={mX} mY={mY}'
        mass_key = f'{mX}_{mY}'

        pp_fname = fname.replace('_rank00_of01.npz', '_metpp_rank00_of01.npz')
        pp_path  = os.path.join(args.infer_pp_dir, pp_fname)

        print(f'\n{"="*60}  {label}')

        d    = np.load(npz_path)
        N    = min(args.n_events, len(d['parts_truth']))
        pt   = d['parts_truth'][:N].astype(np.float32)
        pg   = d['parts_gen'][:N].astype(np.float32)
        mask = d['mask'][:N].astype(np.float32)
        mkg  = d['mask_gen'][:N].astype(np.float32)
        pfeat = d['parton_feat'][:N].astype(np.float32)

        d_pp  = np.load(pp_path)
        pg_pp = d_pp['parts_gen'][:N].astype(np.float32)

        obs_truth = compute_all_obs(pt,   mask, pfeat)
        obs_gen   = compute_all_obs(pg,   mkg,  pfeat)
        obs_pp    = compute_all_obs(pg_pp, mkg,  pfeat)

        out_subdir = os.path.join(args.out_dir, mass_key)
        os.makedirs(out_subdir, exist_ok=True)

        all_obs = [(k, xl, bf) for k, xl, bf in _CATALOGUE_EVT] + \
                  [(k, xl, bf) for k, xl, bf in _CATALOGUE_PART]

        for obs_key, xlabel, bins_fn in all_obs:
            t  = obs_truth[obs_key]
            g  = obs_gen[obs_key]
            gp = obs_pp[obs_key]

            t_fin  = t[np.isfinite(t)]
            g_fin  = g[np.isfinite(g)]
            gp_fin = gp[np.isfinite(gp)]

            w1_before = rel_w1(g_fin,  t_fin)
            w1_after  = rel_w1(gp_fin, t_fin)
            change    = w1_after - w1_before
            print(f'  {obs_key:<22}  before={w1_before:.4f}  '
                  f'after={w1_after:.4f}  Δ={change:+.4f}')

            all_rows.append({
                'mass_point': mass_key,
                'observable': obs_key,
                'w1_before':  w1_before,
                'w1_after':   w1_after,
                'delta':      change,
            })

            bins = bins_fn(t_fin, g_fin)
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            ax.hist(t_fin,  bins=bins, histtype='step', density=True,
                    lw=2.2, color='#333333', label='Truth')
            ax.hist(g_fin,  bins=bins, histtype='step', density=True,
                    lw=1.6, color='steelblue', label='Generated (before)', ls='--')
            ax.hist(gp_fin, bins=bins, histtype='step', density=True,
                    lw=1.6, color='tomato', label='Generated + MET pp', ls='-.')
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title(f'{label}  |  {obs_key}\n'
                         f'before={w1_before:.4f}  after={w1_after:.4f}',
                         fontsize=8)
            ax.legend(fontsize=7.5, framealpha=0.85)
            fig.tight_layout()
            safe = obs_key.replace('/', '_')
            fig.savefig(os.path.join(out_subdir, f'{safe}.png'), dpi=150)
            plt.close(fig)

    # ── Summary CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, 'wasserstein_summary.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['mass_point','observable',
                                            'w1_before','w1_after','delta'])
        w.writeheader()
        w.writerows(all_rows)
    print(f'\nWasserstein table -> {csv_path}')

    # ── Aggregate summary print ────────────────────────────────────────────────
    print('\n' + '='*75)
    print('AGGREGATE (mean over 4 mass points):')
    print('='*75)
    obs_keys = [k for k, _, _ in _CATALOGUE_EVT] + \
               [k for k, _, _ in _CATALOGUE_PART]
    print(f'  {"Observable":<22}  {"W1 before":>10}  {"W1 after":>10}  {"Delta":>10}')
    print('  ' + '-'*60)
    for ok in obs_keys:
        rows_ok = [r for r in all_rows if r['observable'] == ok]
        mb = np.nanmean([r['w1_before'] for r in rows_ok])
        ma = np.nanmean([r['w1_after']  for r in rows_ok])
        md = np.nanmean([r['delta']     for r in rows_ok])
        print(f'  {ok:<22}  {mb:>10.4f}  {ma:>10.4f}  {md:>+10.4f}')
    print('='*75)

    print(f'\nAll done. Figures in {args.out_dir}')


if __name__ == '__main__':
    main()
