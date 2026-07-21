#!/usr/bin/env python3
"""
Locality-scale diagnostics for E020c / E022 holdout inference.

Compares generated vs truth for large-R fat-jet observables and event-level
4-momentum sums.  Uses anti-kT R=1.0 fat jets (two leading jets per event).

Observable set (all new relative to plot_e008_bsm_holdout.py):
  Fat-jet (R=1.0):  J1 pT, J1 eta, J1 mass, J2 pT, J2 eta, J2 mass,
                    delta_phi(J1,J2), m(J1+J2), tau32(J1), tau32(J2)
  Event sums:       sum_E, sum_px, sum_py, sum_pz
  (sum_pT / HT is already in the standard set; excluded here.)

Output: {out_base}/{mX}_{mY}/{observable}.png  +  summary.csv per grid point

Usage (on Perlmutter, login node):
  PYTHONPATH=/global/u2/l/lcondren/.local/perlmutter/tensorflow2.15.0/lib/python3.9/site-packages \\
  /global/common/software/nersc9/tensorflow/2.15.0/bin/python3.9 \\
      plot_e020c_locality_diagnostics.py \\
      --infer_dir /pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid/bsm_grid_event_c/infer_holdout_truth \\
      --run_label E020c
"""
import argparse, os, glob, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from pyjet import cluster, DTYPE_PTEPM
except ImportError:
    sys.exit(
        "pyjet not found. Invoke with:\n"
        "  PYTHONPATH=/global/u2/l/lcondren/.local/perlmutter/tensorflow2.15.0/"
        "lib/python3.9/site-packages \\\n"
        "  /global/common/software/nersc9/tensorflow/2.15.0/bin/python3.9 "
        "plot_e020c_locality_diagnostics.py ..."
    )

from scipy.stats import wasserstein_distance

R_JET  = 1.0
PT_MIN = 20.0   # GeV, minimum leading-jet pT to cluster
BETA   = 1.0    # N-subjettiness angular exponent

# ── helpers ───────────────────────────────────────────────────────────────────

def _dphi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _tau_N(jet, N, beta=BETA, R=R_JET):
    """N-subjettiness using N exclusive kT sub-axes inside fat jet."""
    cs = jet.constituents()
    if len(cs) < N:
        return np.nan
    c_pT  = np.array([c.pt  for c in cs])
    c_eta = np.array([c.eta for c in cs])
    c_phi = np.array([c.phi for c in cs])
    arr = np.zeros(len(cs), dtype=DTYPE_PTEPM)
    arr['pT'] = c_pT; arr['eta'] = c_eta; arr['phi'] = c_phi; arr['mass'] = 0.0
    try:
        subjets = cluster(arr, R=R, p=1).exclusive_jets(N)
    except Exception:
        return np.nan
    if len(subjets) < N:
        return np.nan
    s_eta = np.array([s.eta for s in subjets])
    s_phi = np.array([s.phi for s in subjets])
    deta  = c_eta[:, None] - s_eta[None, :]
    dphi_ = (c_phi[:, None] - s_phi[None, :] + np.pi) % (2 * np.pi) - np.pi
    dR    = np.sqrt(deta**2 + dphi_**2)
    d0    = c_pT.sum() * R**beta
    return (c_pT * dR.min(axis=1)**beta).sum() / (d0 + 1e-8)


def _tau32(jet, R=R_JET):
    t2 = _tau_N(jet, 2, R=R)
    t3 = _tau_N(jet, 3, R=R)
    if np.isnan(t2) or t2 < 1e-8:
        return np.nan
    return t3 / t2


def process_events(parts, mask_2d, verbose_every=1000):
    """
    parts:   (N, P, 6)  — [eta, sin_phi, cos_phi, log_pT, pid, charge]
    mask_2d: (N, P)     — bool or float mask
    Returns dict of per-event observable arrays (NaN where not defined).
    """
    N   = len(parts)
    mk2 = mask_2d.astype(bool)

    # ── vectorised event-level 4-momentum sums ───────────────────────────────
    pT_mat = np.exp(np.clip(parts[:, :, 3], -10, 10)) * mk2
    eta_m  = parts[:, :, 0]
    sphi_m = parts[:, :, 1]
    cphi_m = parts[:, :, 2]

    sum_E  = (pT_mat * np.cosh(eta_m)).sum(1)
    sum_px = (pT_mat * cphi_m).sum(1)
    sum_py = (pT_mat * sphi_m).sum(1)
    sum_pz = (pT_mat * np.sinh(eta_m)).sum(1)

    # ── per-event jet clustering ──────────────────────────────────────────────
    nans   = np.full(N, np.nan)
    j1_pT  = nans.copy(); j1_eta  = nans.copy(); j1_mass = nans.copy()
    j2_pT  = nans.copy(); j2_eta  = nans.copy(); j2_mass = nans.copy()
    dphi12 = nans.copy(); m_jj    = nans.copy()
    t32_j1 = nans.copy(); t32_j2  = nans.copy()

    for i in range(N):
        if verbose_every and (i % verbose_every == 0):
            print(f'    event {i}/{N} ...', flush=True)
        p  = parts[i][mk2[i]]
        if len(p) < 2:
            continue
        pT_  = np.exp(np.clip(p[:, 3], -10, 10))
        ok   = np.isfinite(pT_) & (pT_ > 0.01)
        p, pT_ = p[ok], pT_[ok]
        if len(p) < 2:
            continue

        arr = np.zeros(len(p), dtype=DTYPE_PTEPM)
        arr['pT']   = pT_
        arr['eta']  = p[:, 0]
        arr['phi']  = np.arctan2(p[:, 1], p[:, 2])
        arr['mass'] = 0.0

        jets = cluster(arr, R=R_JET, p=-1).inclusive_jets(ptmin=PT_MIN)
        if not jets:
            continue

        j1 = jets[0]
        j1_pT[i]  = j1.pt;  j1_eta[i]  = j1.eta;  j1_mass[i] = j1.mass
        t32_j1[i] = _tau32(j1)

        if len(jets) >= 2:
            j2 = jets[1]
            j2_pT[i]  = j2.pt;  j2_eta[i]  = j2.eta;  j2_mass[i] = j2.mass
            t32_j2[i] = _tau32(j2)
            dphi12[i] = abs(_dphi(j1.phi, j2.phi))
            E_  = j1.e  + j2.e
            px_ = j1.px + j2.px
            py_ = j1.py + j2.py
            pz_ = j1.pz + j2.pz
            m_jj[i] = np.sqrt(max(E_**2 - px_**2 - py_**2 - pz_**2, 0.0))

    return dict(
        j1_pT=j1_pT, j1_eta=j1_eta, j1_mass=j1_mass,
        j2_pT=j2_pT, j2_eta=j2_eta, j2_mass=j2_mass,
        dphi_j1j2=dphi12, m_jj=m_jj,
        sum_E=sum_E, sum_px=sum_px, sum_py=sum_py, sum_pz=sum_pz,
        tau32_j1=t32_j1, tau32_j2=t32_j2,
    )


# ── observable metadata ───────────────────────────────────────────────────────
# (key, x-axis label, units string, log_x, bin_function(t_arr, g_arr)->bins)

def _pct_bins(lo_p, hi_p, n=60):
    def _fn(t, g):
        lo = min(np.nanpercentile(t, lo_p), np.nanpercentile(g, lo_p))
        hi = max(np.nanpercentile(t, hi_p), np.nanpercentile(g, hi_p))
        return np.linspace(lo * 0.98, hi * 1.02, n)
    return _fn


OBS_META = [
    ('j1_pT',    r'Leading jet $p_T$',          'GeV', False, _pct_bins(0.5, 99.5)),
    ('j1_eta',   r'Leading jet $\eta$',          '',    False, lambda t, g: np.linspace(-5, 5, 60)),
    ('j1_mass',  r'Leading jet mass',            'GeV', False, _pct_bins(0,   99.5)),
    ('j2_pT',    r'Subleading jet $p_T$',        'GeV', False, _pct_bins(0.5, 99.5)),
    ('j2_eta',   r'Subleading jet $\eta$',       '',    False, lambda t, g: np.linspace(-5, 5, 60)),
    ('j2_mass',  r'Subleading jet mass',         'GeV', False, _pct_bins(0,   99.5)),
    ('dphi_j1j2',r'$|\Delta\phi(J_1,J_2)|$',    'rad', False, lambda t, g: np.linspace(0, np.pi, 60)),
    ('m_jj',     r'$m(J_1+J_2)$',               'GeV', False, _pct_bins(0.5, 99.5)),
    ('sum_E',    r'$\Sigma E$',                  'GeV', False, _pct_bins(0,   99.5)),
    ('sum_px',   r'$\Sigma p_x$',               'GeV', False, _pct_bins(0.5, 99.5)),
    ('sum_py',   r'$\Sigma p_y$',               'GeV', False, _pct_bins(0.5, 99.5)),
    ('sum_pz',   r'$\Sigma p_z$',               'GeV', False, _pct_bins(0.5, 99.5)),
    ('tau32_j1', r'$\tau_{32}$ leading jet',    '',    False, _pct_bins(0,   99)),
    ('tau32_j2', r'$\tau_{32}$ subleading jet', '',    False, _pct_bins(0,   99)),
]

COLORS = {'truth': '#1f77b4', 'gen': '#ff7f0e'}

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description='Large-R fat-jet + event-sum locality diagnostics.')
    pa.add_argument('--infer_dir', required=True,
                    help='Directory containing bsm_mX*_mY*.npz inference files.')
    pa.add_argument('--out_base',  default='figures/E022_locality_diagnostics',
                    help='Base output directory (grid-point subdirs created here).')
    pa.add_argument('--run_label', default='E020c',
                    help='Label for plot titles (e.g. E020c, E022).')
    pa.add_argument('--n_events',  type=int, default=5000,
                    help='Max events per grid point to process.')
    args = pa.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.infer_dir, 'bsm_mX*.npz')))
    if not npz_files:
        sys.exit(f'No bsm_mX*.npz files found in {args.infer_dir}')
    print(f'Found {len(npz_files)} grid point(s): {[os.path.basename(f) for f in npz_files]}')

    for path in npz_files:
        d   = np.load(path)
        mX  = int(float(d['mass_x']))
        mY  = int(float(d['mass_y']))
        key = f'{mX}_{mY}'
        N   = min(args.n_events, len(d['parts_truth']))
        label = rf'$m_X$={mX} GeV, $m_Y$={mY} GeV'

        print(f'\n{"="*60}')
        print(f'Grid point {key}  (N={N} events)')
        print(f'{"="*60}')

        mask_t  = d['mask'][:N].astype(bool)
        mask_g  = d['mask_gen'][:N].astype(bool)
        parts_t = d['parts_truth'][:N]
        parts_g = d['parts_gen'][:N]

        print('  Truth clustering ...', flush=True)
        obs_t = process_events(parts_t, mask_t)
        print('  Generated clustering ...', flush=True)
        obs_g = process_events(parts_g, mask_g)

        out_dir = os.path.join(args.out_base, key)
        os.makedirs(out_dir, exist_ok=True)

        summary_rows = []

        for obs_key, xlabel, units, log_x, bin_fn in OBS_META:
            t_raw = obs_t[obs_key]
            g_raw = obs_g[obs_key]
            t_fin = t_raw[np.isfinite(t_raw)]
            g_fin = g_raw[np.isfinite(g_raw)]
            if len(t_fin) < 10 or len(g_fin) < 10:
                print(f'  {obs_key}: insufficient finite data ({len(t_fin)} t, {len(g_fin)} g), skip')
                continue

            bins      = bin_fn(t_fin, g_fin)
            w1        = float(wasserstein_distance(t_fin, g_fin))
            obs_range = float(max(t_fin.max() - t_fin.min(), 1e-8))
            rel_w1    = w1 / obs_range

            summary_rows.append(dict(
                observable=obs_key,
                truth_mean=float(np.nanmean(t_raw)),
                gen_mean=float(np.nanmean(g_raw)),
                wasserstein=w1,
                obs_range=obs_range,
                rel_wasserstein=rel_w1,
            ))

            # ── plot ──────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
            ax.hist(t_fin, bins=bins, density=True, histtype='step',
                    lw=1.8, color=COLORS['truth'], label='Truth')
            ax.hist(g_fin, bins=bins, density=True, histtype='step',
                    lw=1.8, color=COLORS['gen'],   label='Generated', ls='--')
            u = f' {units}' if units else ''
            ax.set_xlabel(f'{xlabel}{u}', fontsize=10)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title(
                f'{args.run_label}  —  {label}\n'
                f'anti-$k_T$  R=1.0,  $p_T>${PT_MIN:.0f} GeV  |  '
                f'W₁={w1:.3g}{u},  rel-W₁={rel_w1:.2%}',
                fontsize=8)
            if log_x:
                ax.set_xscale('log')
            ax.legend(fontsize=9, loc='upper left')
            ax.text(0.97, 0.97,
                    f'truth: mean={np.nanmean(t_raw):.3g}{u}\n'
                    f'gen:   mean={np.nanmean(g_raw):.3g}{u}',
                    transform=ax.transAxes, fontsize=7,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='none'))

            out_path = os.path.join(out_dir, f'{obs_key}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  {obs_key:<20} W₁={w1:>9.3g}{u:4s}  rel={rel_w1:>6.2%}  -> {out_path}')

        # ── summary table ────────────────────────────────────────────────────
        print(f'\n  Summary — {key}')
        hdr = f'  {"Observable":<22} {"Truth mean":>12} {"Gen mean":>12} {"W₁":>10} {"Range":>10} {"Rel W₁":>8}'
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for row in summary_rows:
            u = ''
            for (k, xl, un, *_) in OBS_META:
                if k == row['observable']:
                    u = f' {un}' if un else ''
                    break
            print(f'  {row["observable"]:<22} '
                  f'{row["truth_mean"]:>12.4g} '
                  f'{row["gen_mean"]:>12.4g} '
                  f'{row["wasserstein"]:>10.4g} '
                  f'{row["obs_range"]:>10.4g} '
                  f'{row["rel_wasserstein"]:>8.2%}')

        # ── CSV ──────────────────────────────────────────────────────────────
        csv_path = os.path.join(out_dir, 'summary.csv')
        with open(csv_path, 'w') as fh:
            fh.write('observable,truth_mean,gen_mean,wasserstein,obs_range,rel_wasserstein\n')
            for row in summary_rows:
                fh.write(f'{row["observable"]},'
                         f'{row["truth_mean"]:.6g},'
                         f'{row["gen_mean"]:.6g},'
                         f'{row["wasserstein"]:.6g},'
                         f'{row["obs_range"]:.6g},'
                         f'{row["rel_wasserstein"]:.6g}\n')
        print(f'\n  CSV: {csv_path}')

    print('\nAll grid points done.')


if __name__ == '__main__':
    main()
