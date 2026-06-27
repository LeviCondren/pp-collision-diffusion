#!/usr/bin/env python3
"""
compare_event_conditioning.py — E021 analysis

Compares three event-level conditioning variants (E020a/b/c) against the baseline
(E010, plain BSM grid without event token) at the 4 BSM holdout mass points.
Tests whether adding a learned event token to the cross-attention KV set improves
generation of event-level observables.

Observables:
  Event-level (per event, shape N):
    MET_mag, MET_phi
    cone_pT_X, cone_mass_X  (R=1.0 cone around X-boson parton direction)
    cone_pT_Y, cone_mass_Y  (R=1.0 cone around Y-boson parton direction)
  Sanity check (per particle, flattened):
    eta_particle, logpT_particle  (should be similar across all variants)

Outputs:
  <out_dir>/<observable>_<mass_point>.pdf  — per (grid point, observable) overlay
  <out_dir>/wasserstein_table.csv          — W1 per (observable, mass_point) per variant

Usage:
    python3 compare_event_conditioning.py \\
        --baseline_dir  .../bsm_grid/infer_holdout_5k \\
        --e020a_dir     .../bsm_grid_event_a/infer_holdout \\
        --e020b_dir     .../bsm_grid_event_b/infer_holdout \\
        --e020c_dir     .../bsm_grid_event_c/infer_holdout \\
        --out_dir       figures/E020_event_conditioning_comparison

The baseline directory should contain bsm_mX*.npz files for the 4 holdout
mass points (250,250), (250,300), (300,250), (300,300).  E020* directories
should contain NPZs from inference runs with the same mass points.

If a variant directory does not exist or is missing a mass point, that variant
is silently skipped for that point and noted in the output.
"""

import os, glob, argparse, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance as _wass

# Must match the training scripts (bsm_grid_train_event_*.py)
R_CONE = 1.0

# Variant metadata: (arg-key, display-label, line-color)
VARIANTS = [
    ('baseline', 'Baseline (E010)',  '#1f77b4'),
    ('e020a',    'E020a MET (3)',    '#ff7f0e'),
    ('e020b',    'E020b cone_X (2)', '#2ca02c'),
    ('e020c',    'E020c all-7',      '#d62728'),
]
TRUTH_COLOR = '#333333'
TRUTH_LABEL = 'Truth'

_CKPT = '/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid'

# ── Observable catalogue ───────────────────────────────────────────────────────
# Each entry: (key, x-axis label, units string, bins_fn(truth_array) -> edges)

_OBS_CATALOGUE = [
    ('MET_mag',
     r'MET magnitude [GeV]', 'GeV',
     lambda v: np.linspace(0, max(float(np.percentile(v, 99)) * 1.1, 1.0), 60)),

    ('MET_phi',
     r'MET $\phi$ [rad]', 'rad',
     lambda v: np.linspace(-np.pi, np.pi, 60)),

    ('cone_pT_X',
     r'Cone $p_T$ (X boson) [GeV]', 'GeV',
     lambda v: np.linspace(0, max(float(np.percentile(v, 99)) * 1.1, 1.0), 60)),

    ('cone_mass_X',
     r'Cone mass (X boson) [GeV]', 'GeV',
     lambda v: np.linspace(0, max(float(np.percentile(v, 99)) * 1.1, 1.0), 60)),

    ('cone_pT_Y',
     r'Cone $p_T$ (Y boson) [GeV]', 'GeV',
     lambda v: np.linspace(0, max(float(np.percentile(v, 99)) * 1.1, 1.0), 60)),

    ('cone_mass_Y',
     r'Cone mass (Y boson) [GeV]', 'GeV',
     lambda v: np.linspace(0, max(float(np.percentile(v, 99)) * 1.1, 1.0), 60)),

    ('eta_particle',
     r'Particle $\eta$', '',
     lambda v: np.linspace(-5, 5, 60)),

    ('logpT_particle',
     r'Particle $\log p_T$', '',
     lambda v: np.linspace(
         max(-3.0, float(np.percentile(v[np.isfinite(v)], 1))),
         min(8.0,  float(np.percentile(v[np.isfinite(v)], 99))),
         60)),
]
OBS_KEYS = [o[0] for o in _OBS_CATALOGUE]


# ── Event observable computation ───────────────────────────────────────────────

def _compute_obs(parts, mask, parton_feat):
    """Compute all event-level and per-particle observables.

    Parameters
    ----------
    parts       : (N, npart, 6)  raw particle features
                  [eta, sin_phi, cos_phi, log_pT, pid, charge]
    mask        : (N, npart)     float/bool, 1 = valid particle
    parton_feat : (N, 4, 7)      raw parton features
                  feature[1]=sin_phi, feature[2]=cos_phi, feature[3]=pz/p
                  slot 2 = X boson, slot 3 = Y boson; feature[6] = mass/600

    Returns
    -------
    dict obs_key -> np.ndarray
      Event-level keys have shape (N,).
      Per-particle keys ('eta_particle', 'logpT_particle') are 1-D flattened
      arrays of length = total valid particles.
    """
    float_mask = mask.astype(np.float32)
    valid      = mask.astype(bool)

    pT  = np.exp(np.clip(parts[:, :, 3], -10, 10)) * float_mask
    sp  = parts[:, :, 1]
    cp  = parts[:, :, 2]
    eta = parts[:, :, 0]
    phi = np.arctan2(sp, cp)

    # MET
    MET_x   = (pT * cp).sum(1)
    MET_y   = (pT * sp).sum(1)
    MET_mag = np.sqrt(MET_x**2 + MET_y**2)
    MET_phi = np.arctan2(MET_y, MET_x)

    obs = {'MET_mag': MET_mag, 'MET_phi': MET_phi}

    # Parton-cone observables
    eta_clip = np.clip(eta, -8, 8)
    for slot, suffix in [(2, 'X'), (3, 'Y')]:
        pze   = np.clip(parton_feat[:, slot, 3], -1 + 1e-7, 1 - 1e-7)
        eta_p = 0.5 * np.log((1 + pze) / (1 - pze))
        phi_p = np.arctan2(parton_feat[:, slot, 1], parton_feat[:, slot, 2])

        deta = eta - eta_p[:, None]
        dphi = phi - phi_p[:, None]
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dR   = np.sqrt(deta**2 + dphi**2)
        in_c = (dR < R_CONE) & valid

        wt   = pT * in_c
        pT_c = wt.sum(1)
        E_c  = (wt * np.cosh(eta_clip)).sum(1)
        px_c = (wt * cp).sum(1)
        py_c = (wt * sp).sum(1)
        pz_c = (wt * np.sinh(eta_clip)).sum(1)
        m2   = np.maximum(E_c**2 - px_c**2 - py_c**2 - pz_c**2, 0.0)

        obs[f'cone_pT_{suffix}']   = pT_c
        obs[f'cone_mass_{suffix}'] = np.sqrt(m2)

    # Per-particle sanity-check distributions (flattened)
    obs['eta_particle']   = eta[valid]
    obs['logpT_particle'] = parts[:, :, 3][valid]

    return obs


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_dir(npz_dir, n_events):
    """Load all bsm_mX*.npz files from a directory.

    Returns dict {mass_key: {arrays...}} or {} if directory/files are missing.
    """
    if not os.path.isdir(npz_dir):
        print(f'  [WARN] directory not found, skipping: {npz_dir}')
        return {}
    files = sorted(glob.glob(os.path.join(npz_dir, 'bsm_mX*.npz')))
    if not files:
        print(f'  [WARN] no bsm_mX*.npz files in {npz_dir}')
        return {}
    result = {}
    for path in files:
        d   = np.load(path)
        mX  = float(d['mass_x'])
        mY  = float(d['mass_y'])
        key = f'mX{int(mX):04d}_mY{int(mY):04d}'
        N   = min(n_events, len(d['parts_truth']))
        result[key] = {
            'parts_truth':  d['parts_truth'][:N].astype(np.float32),
            'parts_gen':    d['parts_gen'][:N].astype(np.float32),
            'mask':         d['mask'][:N].astype(np.float32),
            'mask_gen':     d['mask_gen'][:N].astype(np.float32),
            'parton_feat':  d['parton_feat'][:N].astype(np.float32),
            'mX': mX, 'mY': mY,
            'label': rf"$m_X$={int(mX)} GeV, $m_Y$={int(mY)} GeV",
        }
    print(f'  loaded {len(result)} mass points from {os.path.basename(npz_dir)}')
    return result


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description='Compare E020a/b/c event conditioning against baseline (E010).')
    p.add_argument('--baseline_dir',
                   default=f'{_CKPT}/bsm_grid/infer_holdout_5k',
                   help='Inference output dir for baseline (E010)')
    p.add_argument('--e020a_dir',
                   default=f'{_CKPT}/bsm_grid_event_a/infer_holdout',
                   help='Inference output dir for E020a (MET conditioning)')
    p.add_argument('--e020b_dir',
                   default=f'{_CKPT}/bsm_grid_event_b/infer_holdout',
                   help='Inference output dir for E020b (cone_X conditioning)')
    p.add_argument('--e020c_dir',
                   default=f'{_CKPT}/bsm_grid_event_c/infer_holdout',
                   help='Inference output dir for E020c (all-7 conditioning)')
    p.add_argument('--out_dir',
                   default='figures/E020_event_conditioning_comparison',
                   help='Output directory for plots and CSV')
    p.add_argument('--n_events', type=int, default=5000,
                   help='Max events to load per mass point')
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = _parse()
    os.makedirs(args.out_dir, exist_ok=True)

    var_dirs = {
        'baseline': args.baseline_dir,
        'e020a':    args.e020a_dir,
        'e020b':    args.e020b_dir,
        'e020c':    args.e020c_dir,
    }

    # ── Load all variant data ──────────────────────────────────────────────────
    print('Loading inference outputs ...')
    all_data = {}
    for vkey, vdir in var_dirs.items():
        print(f'  {vkey}: {vdir}')
        all_data[vkey] = _load_dir(vdir, args.n_events)

    if not all_data['baseline']:
        raise RuntimeError(
            f"Baseline directory is empty or missing: {args.baseline_dir}\n"
            "Run inference for E010 first.")

    mass_keys = sorted(all_data['baseline'].keys())
    print(f'\nMass points (from baseline): {mass_keys}')

    # ── Compute observables for truth and each variant ─────────────────────────
    print('\nComputing observables ...')
    # truth_obs[mass_key][obs_key]        -> array
    # gen_obs[variant_key][mass_key][obs_key] -> array
    truth_obs = {}
    gen_obs   = {vkey: {} for vkey, _, _ in VARIANTS}

    for mk in mass_keys:
        r_base = all_data['baseline'][mk]
        truth_obs[mk] = _compute_obs(
            r_base['parts_truth'], r_base['mask'], r_base['parton_feat'])
        print(f'  {mk}  truth MET_mag={truth_obs[mk]["MET_mag"].mean():.1f} GeV  '
              f'N={len(r_base["parts_truth"])}')

        for vkey, vlabel, _ in VARIANTS:
            if mk not in all_data[vkey]:
                print(f'  [skip] {vkey} missing {mk}')
                continue
            r = all_data[vkey][mk]
            gen_obs[vkey][mk] = _compute_obs(
                r['parts_gen'], r['mask_gen'], r['parton_feat'])

    # ── Plots: one PDF per (mass_key, observable) ─────────────────────────────
    print('\nGenerating overlay plots ...')
    n_plots = 0
    for mk in mass_keys:
        pt_label = all_data['baseline'][mk]['label']
        for obs_key, obs_xlabel, obs_units, bins_fn in _OBS_CATALOGUE:
            arr_t = truth_obs[mk].get(obs_key)
            if arr_t is None or len(arr_t) == 0:
                continue

            arr_t_finite = arr_t[np.isfinite(arr_t)]
            if len(arr_t_finite) == 0:
                continue

            bins = bins_fn(arr_t_finite)

            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            ax.set_title(pt_label, fontsize=9)

            ax.hist(arr_t_finite, bins=bins, density=True, histtype='step',
                    lw=2.2, color=TRUTH_COLOR, label=TRUTH_LABEL, zorder=5)

            for vkey, vlabel, vcolor in VARIANTS:
                if mk not in gen_obs.get(vkey, {}):
                    continue
                arr_g = gen_obs[vkey][mk].get(obs_key)
                if arr_g is None or len(arr_g) == 0:
                    continue
                arr_g_finite = arr_g[np.isfinite(arr_g)]
                ax.hist(arr_g_finite, bins=bins, density=True, histtype='step',
                        lw=1.5, color=vcolor, label=vlabel, ls='--')

            u_str = f' [{obs_units}]' if obs_units else ''
            ax.set_xlabel(obs_xlabel, fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=7.5, framealpha=0.85)
            fig.tight_layout()

            safe_key = obs_key.replace('/', '_')
            out_path = os.path.join(args.out_dir, f'{safe_key}_{mk}.pdf')
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            n_plots += 1

    print(f'  {n_plots} figures written to {args.out_dir}/')

    # ── Wasserstein distance CSV ───────────────────────────────────────────────
    print('\nComputing Wasserstein distances ...')
    csv_path = os.path.join(args.out_dir, 'wasserstein_table.csv')
    var_keys = [vkey for vkey, _, _ in VARIANTS]

    # Collect rows first so we can also print a summary
    rows = []
    for mk in mass_keys:
        for obs_key, obs_xlabel, obs_units, _ in _OBS_CATALOGUE:
            arr_t = truth_obs[mk].get(obs_key)
            if arr_t is None or len(arr_t) == 0:
                continue
            arr_t_finite = arr_t[np.isfinite(arr_t)]
            row = {'mass_point': mk, 'observable': obs_key}
            for vkey in var_keys:
                if mk not in gen_obs.get(vkey, {}):
                    row[vkey] = None
                    continue
                arr_g = gen_obs[vkey][mk].get(obs_key, np.array([]))
                arr_g_finite = arr_g[np.isfinite(arr_g)] if len(arr_g) else arr_g
                if len(arr_t_finite) == 0 or len(arr_g_finite) == 0:
                    row[vkey] = None
                else:
                    row[vkey] = float(_wass(arr_t_finite, arr_g_finite))
            rows.append(row)

    # Write CSV
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['mass_point', 'observable'] + var_keys)
        for row in rows:
            vals = [row['mass_point'], row['observable']]
            for vkey in var_keys:
                v = row.get(vkey)
                vals.append(f'{v:.6f}' if v is not None else '')
            writer.writerow(vals)

    print(f'  -> {csv_path}')

    # Print summary table to stdout
    _print_summary(rows, mass_keys, var_keys)

    print(f'\nDone. All outputs in: {args.out_dir}')


def _print_summary(rows, mass_keys, var_keys):
    """Print a compact summary of W1 distances to stdout."""
    print('\n' + '='*80)
    print('Wasserstein W1 summary (generated vs truth):')
    print('='*80)
    header = f"{'Observable':<22}" + ''.join(f'{k:<14}' for k in var_keys)
    for mk in mass_keys:
        print(f'\n  {mk}')
        print(f"  {'Observable':<20}" + ''.join(f'  {k:<12}' for k in var_keys))
        print('  ' + '-' * (20 + 14 * len(var_keys)))
        mk_rows = [r for r in rows if r['mass_point'] == mk]
        for row in mk_rows:
            line = f"  {row['observable']:<20}"
            for vkey in var_keys:
                v = row.get(vkey)
                line += f"  {v:.4f}      " if v is not None else f"  {'N/A':<12}"
            print(line)
    print('='*80 + '\n')


if __name__ == '__main__':
    main()
