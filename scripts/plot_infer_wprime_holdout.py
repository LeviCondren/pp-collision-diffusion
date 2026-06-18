#!/usr/bin/env python3
"""
Inference comparison plots for the wprimeGrid holdout results.
Produces the same 5 plot types as plot_infer_5proc.py, one set per mass point.

Usage:
    python plot_infer_wprime_holdout.py \
        --infer_dir /pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints/wprimeGrid/infer_holdout \
        --out_dir   /pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints/wprimeGrid/plots
"""

import os, sys, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import wasserstein_distance as _wass

p = argparse.ArgumentParser()
p.add_argument('--infer_dir', type=str,
               default='/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints/wprimeGrid/infer_holdout')
p.add_argument('--out_dir', type=str, default=None)
p.add_argument('--n_events', type=int, default=20000)
args = p.parse_args()

OUT_DIR = args.out_dir or os.path.join(os.path.dirname(args.infer_dir), 'plots')
os.makedirs(OUT_DIR, exist_ok=True)
N = args.n_events

# ── Load all mass points ──────────────────────────────────────────────────────
npz_files = sorted(glob.glob(os.path.join(args.infer_dir, 'mX*.npz')))
assert npz_files, f'No mX*.npz files found in {args.infer_dir}'

mass_points = {}
for path in npz_files:
    d = np.load(path)
    mX = float(d['mass_x'][0])
    mY = float(d['mass_y'][0])
    key = f'mX{int(mX):04d}_mY{int(mY):04d}'
    label = rf"W' $m_{{X}}$={int(mX)} GeV, $m_{{Y}}$={int(mY)} GeV"
    mass_points[key] = {
        'parts_truth': d['parts_truth'][:N],
        'parts_gen':   d['parts_gen'][:N],
        'mask':        d['mask'][:N, :, None].astype(np.float32),
        'mask_gen':    d['mask_gen'][:N, :, None].astype(np.float32),
        'parton_feat': d['parton_feat'][:N],
        'mX': mX, 'mY': mY, 'label': label,
    }
    npt = d['mask'][:N].sum(axis=1).mean()
    npg = d['mask_gen'][:N].sum(axis=1).mean()
    print(f'  {key}: {N} events  truth npart={npt:.1f}  gen npart={npg:.1f}')

COLORS = {'truth': '#1f77b4', 'gen': '#ff7f0e'}

# ── Metrics ───────────────────────────────────────────────────────────────────
def _hist_jsd(arr_t, arr_g, bins):
    eps = 1e-10
    ht, _ = np.histogram(arr_t, bins=bins, density=True)
    hg, _ = np.histogram(arr_g, bins=bins, density=True)
    ht = ht.astype(float) + eps;  hg = hg.astype(float) + eps
    ht /= ht.sum();                hg /= hg.sum()
    m = 0.5 * (ht + hg)
    return float(np.clip(0.5*(ht*np.log(ht/m)).sum() + 0.5*(hg*np.log(hg/m)).sum(), 0, None))

def _binned_jsd(ht, hg):
    eps = 1e-10
    ht = np.array(ht, dtype=float) + eps;  hg = np.array(hg, dtype=float) + eps
    ht /= ht.sum();                          hg /= hg.sum()
    m = 0.5 * (ht + hg)
    return float(np.clip(0.5*(ht*np.log(ht/m)).sum() + 0.5*(hg*np.log(hg/m)).sum(), 0, None))

def _score_text(ax, jsd, w1, units='', pos=(0.97, 0.03)):
    u = f' {units}' if units else ''
    ax.text(*pos, f'JSD={jsd:.4f}\nW₁={w1:.3g}{u}',
            transform=ax.transAxes, fontsize=6.5, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'))

def _img_pearson(H_t, H_g):
    a, b = H_t.ravel(), H_g.ravel()
    if a.std() < 1e-10 or b.std() < 1e-10: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

# ── Plot 1: Particle distributions ───────────────────────────────────────────
print('\nPlot 1: particle distributions ...')
n_pts = len(mass_points)
fig, axes = plt.subplots(n_pts, 4, figsize=(20, 4*n_pts))
if n_pts == 1: axes = axes[np.newaxis, :]
fig.suptitle('Particle distributions — W\' holdout  (Truth vs Generated)', fontsize=11)

for row, (key, r) in enumerate(mass_points.items()):
    m_t = r['mask'][:, :, 0].astype(bool);  m_g = r['mask_gen'][:, :, 0].astype(bool)
    eta_t = r['parts_truth'][m_t, 0];        eta_g = r['parts_gen'][m_g, 0]
    phi_t = np.arctan2(r['parts_truth'][m_t, 1], r['parts_truth'][m_t, 2])
    phi_g = np.arctan2(r['parts_gen'][m_g, 1],   r['parts_gen'][m_g, 2])
    pT_t  = np.exp(r['parts_truth'][m_t, 3])
    pT_g  = np.exp(np.clip(r['parts_gen'][m_g, 3], -10, 10))
    pT_g  = pT_g[np.isfinite(pT_g) & (pT_g > 0)]
    chg_t = r['parts_truth'][m_t, 5];  chg_g = r['parts_gen'][m_g, 5]

    bins_phi = np.linspace(-np.pi, np.pi, 60)
    axes[row, 0].hist(phi_t, bins=bins_phi, density=True, histtype='step', lw=1.5, label='Truth')
    axes[row, 0].hist(phi_g, bins=bins_phi, density=True, histtype='step', lw=1.5, label='Generated')
    axes[row, 0].set_xlabel(r'Particle $\phi$ [rad]')
    axes[row, 0].set_ylabel(r['label'], fontsize=8)
    axes[row, 0].set_title(r'$\phi$');  axes[row, 0].legend(fontsize=8)
    _score_text(axes[row, 0], _hist_jsd(phi_t, phi_g, bins_phi), _wass(phi_t, phi_g), units='rad')

    _pT_max = max(np.percentile(pT_t, 99), np.percentile(pT_g, 99)) if len(pT_g) else np.percentile(pT_t, 99)
    bins_pT = np.logspace(np.log10(0.3), np.log10(_pT_max + 1), 60)
    axes[row, 1].hist(pT_t, bins=bins_pT, density=True, histtype='step', lw=1.5)
    axes[row, 1].hist(pT_g, bins=bins_pT, density=True, histtype='step', lw=1.5)
    axes[row, 1].set_xscale('log');  axes[row, 1].set_xlabel(r'Particle $p_T$ [GeV]')
    axes[row, 1].set_title(r'$p_T$ spectrum')
    _score_text(axes[row, 1], _hist_jsd(pT_t, pT_g, bins_pT), _wass(pT_t, pT_g), units='GeV')

    bins_eta = np.linspace(-5, 5, 60)
    axes[row, 2].hist(eta_t, bins=bins_eta, density=True, histtype='step', lw=1.5)
    axes[row, 2].hist(eta_g, bins=bins_eta, density=True, histtype='step', lw=1.5)
    axes[row, 2].set_xlabel(r'Particle $\eta$');  axes[row, 2].set_title(r'$\eta$ distribution')
    _score_text(axes[row, 2], _hist_jsd(eta_t, eta_g, bins_eta), _wass(eta_t, eta_g))

    bins_chg = np.linspace(-2, 2, 41)
    axes[row, 3].hist(chg_t, bins=bins_chg, density=True, histtype='step', lw=1.5)
    axes[row, 3].hist(chg_g, bins=bins_chg, density=True, histtype='step', lw=1.5)
    axes[row, 3].set_xlabel('Particle charge');  axes[row, 3].set_title('Charge distribution')
    _score_text(axes[row, 3], _hist_jsd(chg_t, chg_g, bins_chg), _wass(chg_t, chg_g))

fig.tight_layout()
out = f'{OUT_DIR}/particle_dists.png'
fig.savefig(out, dpi=150, bbox_inches='tight');  plt.close(fig)
print(f'  -> {out}')

# ── Plot 2: Global event observables ─────────────────────────────────────────
print('Plot 2: global event observables ...')

def _pT_masked(parts, mask):   return np.exp(np.clip(parts[:, :, 3], -10, 10)) * mask[:, :, 0]
def _obs_mult(mask):            return mask[:, :, 0].sum(axis=1)
def _obs_HT(parts, mask):       return _pT_masked(parts, mask).sum(axis=1)
def _obs_MET(parts, mask):
    pT = _pT_masked(parts, mask)
    return np.sqrt((pT*parts[:, :, 2]).sum(1)**2 + (pT*parts[:, :, 1]).sum(1)**2)
def _obs_sph(parts, mask):
    pT = _pT_masked(parts, mask); px = pT*parts[:, :, 2]; py = pT*parts[:, :, 1]
    Sxx = (px**2).sum(1); Syy = (py**2).sum(1); Sxy = (px*py).sum(1)
    d = np.clip(Sxx + Syy, 1e-8, None); det = (Sxx*Syy - Sxy**2) / d**2
    lam = (1. - np.sqrt(np.clip(1. - 4.*det, 0, None))) / 2.
    sph = np.clip(2.*lam, 0, 1); sph[d < 1e-6] = 0.0; return sph
def _obs_ef(parts, mask, bins):
    m = mask[:, :, 0].ravel() > 0
    ef, _ = np.histogram(parts[:, :, 0].ravel()[m], bins=bins,
                         weights=np.exp(np.clip(parts[:, :, 3], -10, 10)).ravel()[m])
    return ef / len(parts)

ETA_BINS_EF = np.linspace(-5, 5, 26)
eta_centers  = 0.5 * (ETA_BINS_EF[:-1] + ETA_BINS_EF[1:])
COL_TITLES = ['Multiplicity', 'Scalar HT [GeV]', 'MET [GeV]',
              'Sphericity ST', 'pT spectrum [GeV]', 'η distribution', 'Energy flow vs η']

fig, axes = plt.subplots(n_pts, 7, figsize=(28, 4.5*n_pts))
if n_pts == 1: axes = axes[np.newaxis, :]
fig.suptitle("Global event observables — W' holdout  (Truth vs Generated)", fontsize=11)

for row, (key, r) in enumerate(mass_points.items()):
    pt = r['parts_truth']; msk = r['mask']; pg = r['parts_gen']; msk_g = r['mask_gen']
    mult_t = _obs_mult(msk);    mult_g = _obs_mult(msk_g)
    HT_t   = _obs_HT(pt, msk); HT_g   = _obs_HT(pg, msk_g)
    MET_t  = _obs_MET(pt, msk);MET_g  = _obs_MET(pg, msk_g)
    sph_t  = _obs_sph(pt, msk);sph_g  = _obs_sph(pg, msk_g)
    pT_t   = np.exp(np.clip(pt[:, :, 3], -10, 10))[msk[:, :, 0] > 0]
    pT_g   = np.exp(np.clip(pg[:, :, 3], -10, 10))[msk_g[:, :, 0] > 0]
    eta_t2 = pt[:, :, 0][msk[:, :, 0] > 0]; eta_g2 = pg[:, :, 0][msk_g[:, :, 0] > 0]
    ef_t   = _obs_ef(pt, msk, ETA_BINS_EF); ef_g = _obs_ef(pg, msk_g, ETA_BINS_EF)

    def _hist(ax, t, g, bins, xlabel, log_x=False, units=''):
        ax.hist(t, bins=bins, density=True, histtype='step', lw=1.8, color=COLORS['truth'], label='Truth')
        ax.hist(g, bins=bins, density=True, histtype='step', lw=1.8, color=COLORS['gen'], label='Generated', ls='--')
        ax.set_xlabel(xlabel, fontsize=8)
        if row == 0: ax.legend(fontsize=7)
        if log_x: ax.set_xscale('log')
        tf = t[np.isfinite(t)]; gf = g[np.isfinite(g)]
        if len(tf) > 1 and len(gf) > 1:
            _score_text(ax, _hist_jsd(tf, gf, bins), _wass(tf, gf), units=units)

    nmax = int(max(mult_t.max(), mult_g.max())) + 1; bstep = max(1, nmax // 60)
    _hist(axes[row, 0], mult_t, mult_g, np.arange(0, nmax+bstep, bstep), 'Particle count')
    axes[row, 0].set_ylabel(r['label'], fontsize=8)
    _ht_max = max(np.percentile(HT_t, 99), np.percentile(HT_g, 99))
    _hist(axes[row, 1], HT_t, HT_g, np.linspace(0, _ht_max*1.05, 60), 'GeV', units='GeV')
    _met_max = max(np.percentile(MET_t, 99), np.percentile(MET_g, 99))
    _hist(axes[row, 2], MET_t, MET_g, np.linspace(0, _met_max*1.05, 60), 'GeV', units='GeV')
    _hist(axes[row, 3], sph_t, sph_g, np.linspace(0, 1, 50), r'$S_T$')
    _pT_hi = max(np.percentile(pT_t, 99.5), np.percentile(pT_g, 99.5) if len(pT_g) else 1)
    _hist(axes[row, 4], pT_t, pT_g,
          np.logspace(np.log10(0.3), np.log10(_pT_hi+1), 55), 'GeV', log_x=True, units='GeV')
    _hist(axes[row, 5], eta_t2, eta_g2, np.linspace(-5, 5, 55), r'$\eta$')

    ax = axes[row, 6]
    ax.step(eta_centers, ef_t, where='mid', lw=1.8, color=COLORS['truth'], label='Truth')
    ax.step(eta_centers, ef_g, where='mid', lw=1.8, color=COLORS['gen'], label='Generated', ls='--')
    ax.set_xlabel(r'$\eta$', fontsize=8); ax.set_ylabel(r'$\langle\Sigma p_T\rangle$ [GeV]', fontsize=7)
    if row == 0: ax.legend(fontsize=7)
    _score_text(ax, _binned_jsd(ef_t, ef_g),
                _wass(eta_centers, eta_centers,
                      u_weights=ef_t+1e-10, v_weights=ef_g+1e-10))

    if row == 0:
        for c, title in enumerate(COL_TITLES): axes[0, c].set_title(title, fontsize=9)

fig.tight_layout()
out = f'{OUT_DIR}/global_obs.png'
fig.savefig(out, dpi=150, bbox_inches='tight');  plt.close(fig)
print(f'  -> {out}')

# ── Plot 3: Jet observables ───────────────────────────────────────────────────
print('Plot 3: jet observables ...')
from pyjet import cluster, DTYPE_PTEPM

R_JET = 0.4; PT_MIN = 20.0; BETA_N = 1.0
# PDG norm ÷10 encoding used in wprime parton_feat (raw 4-slot format)
_PDG_TO_FL = [0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4]
FL_G, FL_L, FL_C, FL_B, FL_T = 0, 1, 2, 3, 4
FLAVOR_NAMES = ['Gluon', 'Light (u/d/s)', 'Charm', 'Bottom', 'Top']
TRUTH_COL = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
GEN_COL   = ['#aec7e8', '#98df8a', '#ffbb78', '#c5b0d5', '#f5a9a9']
N_FL = 5

def _dphi(a, b):        return (a - b + np.pi) % (2*np.pi) - np.pi
def _dR(e1, p1, e2, p2): return np.sqrt((e1-e2)**2 + _dphi(p1, p2)**2)

def _to_psj(parts_ev, mask_bool):
    p = parts_ev[mask_bool]; pT = np.exp(np.clip(p[:, 3], -10, 10))
    ok = np.isfinite(pT) & (pT > 0.01); p, pT = p[ok], pT[ok]
    arr = np.zeros(len(p), dtype=DTYPE_PTEPM)
    arr['pT'] = pT; arr['eta'] = p[:, 0]; arr['mass'] = 0.0
    arr['phi'] = np.arctan2(p[:, 1], p[:, 2]); return arr

def _jet_fl(jet_eta, jet_phi, pf_ev):
    n_slots = pf_ev.shape[0]
    best_dr, best_pdg = np.inf, None
    for slot in range(2, n_slots):
        if float(pf_ev[slot, 5]) < 0.5: continue
        pz_e = float(np.clip(pf_ev[slot, 3], -1+1e-7, 1-1e-7))
        eta_p = 0.5 * np.log((1+pz_e) / (1-pz_e))
        phi_p = np.arctan2(float(pf_ev[slot, 1]), float(pf_ev[slot, 2]))
        dr = _dR(jet_eta, jet_phi, eta_p, phi_p)
        if dr < best_dr:
            best_dr = dr
            best_pdg = int(round(float(pf_ev[slot, 4]) * 10))
    if best_dr > R_JET or best_pdg is None: return FL_L
    return _PDG_TO_FL[min(best_pdg, 12)]

def _measure_jet(jet, pf_ev):
    cs = jet.constituents()
    cpt = np.array([c.pt for c in cs]); ceta = np.array([c.eta for c in cs])
    cphi = np.array([c.phi for c in cs]); dR_c = _dR(ceta, cphi, jet.eta, jet.phi)
    width = (cpt*dR_c).sum() / (cpt.sum()*R_JET + 1e-8)
    d0 = cpt.sum()*R_JET**BETA_N + 1e-8; tau1 = (cpt*dR_c**BETA_N).sum()/d0; tau21 = np.nan
    if len(cs) >= 2:
        sub = np.zeros(len(cs), dtype=DTYPE_PTEPM)
        sub['pT'], sub['eta'], sub['phi'], sub['mass'] = cpt, ceta, cphi, 0.0
        axes2 = cluster(sub, R=R_JET, p=1).exclusive_jets(2)
        min_dR = np.full(len(cs), np.inf)
        for ax2 in axes2: min_dR = np.minimum(min_dR, _dR(ceta, cphi, ax2.eta, ax2.phi))
        tau2 = (cpt*min_dR**BETA_N).sum()/d0; tau21 = tau2 / (tau1 + 1e-8)
    fl = _jet_fl(jet.eta, jet.phi, pf_ev)
    return width, tau21, fl

def _cluster_and_measure(parts_arr, mask_arr, parton_feat_arr):
    N_ = len(parts_arr)
    fl_n   = {fl: np.zeros(N_, dtype=int) for fl in range(N_FL)}
    fl_pT  = {fl: [] for fl in range(N_FL)};  fl_lpT = {fl: [] for fl in range(N_FL)}
    fl_mass= {fl: [] for fl in range(N_FL)};  fl_w   = {fl: [] for fl in range(N_FL)}
    fl_t21 = {fl: [] for fl in range(N_FL)};  fl_dR  = {fl: [] for fl in range(N_FL)}
    fl_nc  = {fl: [] for fl in range(N_FL)};  mult   = np.zeros(N_, dtype=int)
    for i in range(N_):
        if i % 5000 == 0: print(f'    clustering {i}/{N_} ...', flush=True)
        arr = _to_psj(parts_arr[i], mask_arr[i, :, 0].astype(bool))
        if len(arr) < 2: continue
        jets = cluster(arr, R=R_JET, p=-1).inclusive_jets(ptmin=PT_MIN)
        jets.sort(key=lambda j: j.pt, reverse=True); mult[i] = len(jets)
        by_fl = {fl: [] for fl in range(N_FL)}
        for jet in jets:
            w, t21, fl = _measure_jet(jet, parton_feat_arr[i])
            fl_pT[fl].append(jet.pt);  fl_mass[fl].append(jet.mass)
            fl_w[fl].append(w)
            if not np.isnan(t21): fl_t21[fl].append(t21)
            fl_nc[fl].append(len(jet.constituents()))
            by_fl[fl].append((jet.pt, jet.eta, jet.phi))
        for fl in range(N_FL):
            fl_n[fl][i] = len(by_fl[fl])
            if by_fl[fl]: fl_lpT[fl].append(by_fl[fl][0][0])
            if len(by_fl[fl]) >= 2:
                (_, e1, p1), (_, e2, p2) = by_fl[fl][0], by_fl[fl][1]
                fl_dR[fl].append(_dR(e1, p1, e2, p2))
    for fl in range(N_FL):
        for k in [fl_pT, fl_lpT, fl_mass, fl_w, fl_t21, fl_dR, fl_nc]:
            k[fl] = np.array(k[fl])
    return mult, fl_n, fl_pT, fl_lpT, fl_mass, fl_w, fl_t21, fl_dR, fl_nc

_jres = {}
for key, r in mass_points.items():
    print(f'  {key} truth ...', flush=True)
    _jres[key] = {'truth': _cluster_and_measure(r['parts_truth'], r['mask'], r['parton_feat'])}
    print(f'  {key} gen ...', flush=True)
    _jres[key]['gen'] = _cluster_and_measure(r['parts_gen'], r['mask_gen'], r['parton_feat'])

def _safe_bins(arr, lo, hi, n=50):
    lo = float(np.clip(lo, -np.inf, hi - 1e-6))
    return np.linspace(lo, hi if hi > lo else lo+1, n)

_OBS = [
    (1, 'Jet multiplicity',           lambda a: np.arange(0, max(int(a.max())+2, 6))),
    (3, r'Leading jet $p_T$ [GeV]',   lambda a: _safe_bins(a, PT_MIN, min(float(a.max())*1.05, 800))),
    (4, 'Jet mass [GeV]',             lambda a: _safe_bins(a, 0, min(float(a.max())*1.05, 250))),
    (5, 'Jet width / R',              lambda a: _safe_bins(a, 0, min(float(a.max())*1.05, 1.0))),
    (6, r'$\tau_{21}$',               lambda a: _safe_bins(a, 0, min(float(a.max())*1.05, 1.2))),
    (7, r'$\Delta R(j_1,j_2)$',       lambda a: _safe_bins(a, 0, min(float(a.max())*1.05, 6.0))),
    (8, 'Constituents per jet',       lambda a: _safe_bins(a, 0, min(float(a.max())*1.05, 150))),
]

for key, r in mass_points.items():
    _t = _jres[key]['truth']; _g = _jres[key]['gen']
    _plot_fl = [fl for fl in range(N_FL) if _t[1][fl].sum() > 0 or _g[1][fl].sum() > 0]
    _n_cols = len(_plot_fl)
    print(f'  {key}: active flavors={[FLAVOR_NAMES[f] for f in _plot_fl]}')
    fig, axes_j = plt.subplots(7, _n_cols, figsize=(3.5*_n_cols, 26),
                               constrained_layout=True, squeeze=False)
    fig.suptitle(f'Jet observables — {r["label"]}\nanti-kT R={R_JET}, pT>{PT_MIN:.0f} GeV', fontsize=10)
    for row, (tidx, ylab, bins_fn) in enumerate(_OBS):
        at_all = _t[tidx]; ag_all = _g[tidx]
        for col, fl in enumerate(_plot_fl):
            ax = axes_j[row, col]
            at_f = np.array(at_all[fl], dtype=float); ag_f = np.array(ag_all[fl], dtype=float)
            at_f = at_f[np.isfinite(at_f)];           ag_f = ag_f[np.isfinite(ag_f)]
            if len(at_f) == 0 and len(ag_f) == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
            else:
                ref = at_f if len(at_f) else ag_f; bins = bins_fn(ref)
                if len(at_f) > 0: ax.hist(at_f, bins=bins, density=True, histtype='step',
                                           lw=1.8, color=TRUTH_COL[fl], label='Truth')
                if len(ag_f) > 0: ax.hist(ag_f, bins=bins, density=True, histtype='step',
                                           lw=1.8, color=GEN_COL[fl], label='Gen', ls='--')
                if len(at_f) > 1 and len(ag_f) > 1:
                    _score_text(ax, _hist_jsd(at_f, ag_f, bins), _wass(at_f, ag_f))
            ax.set_xlabel(ylab, fontsize=8)
            if col == 0: ax.set_ylabel('Density', fontsize=7)
            if row == 0: ax.set_title(FLAVOR_NAMES[fl], fontsize=10,
                                       fontweight='bold', color=TRUTH_COL[fl])
            if row == 0 and col == 0: ax.legend(fontsize=7)
    out = f'{OUT_DIR}/jet_obs_{key}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight');  plt.close(fig)
    print(f'  -> {out}')

# ── Plot 4: Average jet images ────────────────────────────────────────────────
print('Plot 4: jet images ...')
_cone_th = np.linspace(0, 2*np.pi, 200)
N_BINS = 40; DR_MAX = 0.6; _JBINS = np.linspace(-DR_MAX, DR_MAX, N_BINS+1)

def _collect_jet_images(parts_arr, mask_arr, parton_feat_arr):
    imgs   = {fl: np.zeros((N_BINS, N_BINS)) for fl in range(N_FL)}
    counts = {fl: 0 for fl in range(N_FL)}
    for i in range(len(parts_arr)):
        arr = _to_psj(parts_arr[i], mask_arr[i, :, 0].astype(bool))
        if len(arr) < 2: continue
        jets = cluster(arr, R=R_JET, p=-1).inclusive_jets(ptmin=PT_MIN)
        for jet in jets:
            fl = _jet_fl(jet.eta, jet.phi, parton_feat_arr[i])
            cs = jet.constituents()
            if not cs: continue
            ceta = np.array([c.eta for c in cs]) - jet.eta
            cphi = np.array([(c.phi-jet.phi+np.pi) % (2*np.pi)-np.pi for c in cs])
            cpt  = np.array([c.pt for c in cs])
            H, _, _ = np.histogram2d(ceta, cphi, bins=[_JBINS, _JBINS], weights=cpt)
            imgs[fl] += H; counts[fl] += 1
    for fl in range(N_FL):
        if counts[fl] > 0: imgs[fl] /= counts[fl]
    return imgs, counts

for key, r in mass_points.items():
    print(f'  {key} ...', flush=True)
    _imgs_t, _cnts_t = _collect_jet_images(r['parts_truth'], r['mask'],     r['parton_feat'])
    _imgs_g, _cnts_g = _collect_jet_images(r['parts_gen'],   r['mask_gen'], r['parton_feat'])
    _active_fl = [fl for fl in range(N_FL) if _cnts_t[fl] > 0 or _cnts_g[fl] > 0]
    if not _active_fl: continue
    fig, axes_i = plt.subplots(len(_active_fl), 2, figsize=(9, 4.2*len(_active_fl)),
                               constrained_layout=True, squeeze=False)
    fig.suptitle(f'Average jet images (Δη–Δφ) — {r["label"]}', fontsize=10)
    for _row, fl in enumerate(_active_fl):
        _corr = _img_pearson(_imgs_t[fl], _imgs_g[fl])
        for _col, (_img, _n, _side) in enumerate(
                [(_imgs_t[fl], _cnts_t[fl], 'Truth'), (_imgs_g[fl], _cnts_g[fl], 'Generated')]):
            ax = axes_i[_row, _col]
            _pos = _img[_img > 0]
            if len(_pos) == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
            else:
                _norm = mcolors.LogNorm(vmin=max(_pos.min(), _img.max()*1e-4), vmax=_img.max())
                im = ax.pcolormesh(_JBINS, _JBINS, _img.T, norm=_norm, cmap='hot', rasterized=True)
                cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
                cb.set_label(r'$\langle p_T\rangle$ [GeV/bin]', fontsize=7)
                cb.ax.tick_params(labelsize=7)
            ax.plot(R_JET*np.cos(_cone_th), R_JET*np.sin(_cone_th), 'w--', lw=0.8, alpha=0.6)
            ax.axhline(0, color='w', lw=0.4, alpha=0.4); ax.axvline(0, color='w', lw=0.4, alpha=0.4)
            ax.set_xlabel(r'$\Delta\eta$', fontsize=9); ax.set_ylabel(r'$\Delta\phi$', fontsize=9)
            ax.set_aspect('equal')
            ax.set_title(f'{FLAVOR_NAMES[fl]} — {_side} (N={_n:,} jets)'
                         + (f'\nr={_corr:.3f}' if _col == 1 else ''), fontsize=9,
                         color=TRUTH_COL[fl] if _col == 0 else 'gray')
    out = f'{OUT_DIR}/jet_images_{key}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight');  plt.close(fig)
    print(f'  -> {out}')

# ── Plot 5: Parton-cone comparison ────────────────────────────────────────────
print('Plot 5: parton-cone observables ...')

def _parton_cone_measure(parts_arr, mask_arr, parton_feat_arr):
    props = {fl: {'pT': [], 'mass': [], 'width': [], 'nconst': []} for fl in range(N_FL)}
    n_slots = parton_feat_arr.shape[1]
    for i in range(len(parts_arr)):
        m = mask_arr[i, :, 0].astype(bool); p = parts_arr[i, m]
        pT = np.exp(np.clip(p[:, 3], -10, 10)); ok = np.isfinite(pT) & (pT > 0.01)
        p, pT = p[ok], pT[ok]
        eta_all = p[:, 0]               if len(p) else np.array([])
        phi_all = np.arctan2(p[:, 1], p[:, 2]) if len(p) else np.array([])
        for slot in range(2, n_slots):
            if float(parton_feat_arr[i, slot, 5]) < 0.5: continue
            pz_e  = float(np.clip(parton_feat_arr[i, slot, 3], -1+1e-7, 1-1e-7))
            eta_p = 0.5 * np.log((1+pz_e) / (1-pz_e))
            phi_p = np.arctan2(float(parton_feat_arr[i, slot, 1]), float(parton_feat_arr[i, slot, 2]))
            pdg_c = int(round(float(parton_feat_arr[i, slot, 4]) * 10))
            fl    = _PDG_TO_FL[min(pdg_c, 12)]
            if len(eta_all) > 0:
                in_cone = _dR(eta_all, phi_all, eta_p, phi_p) < R_JET
                cpt = pT[in_cone]; ceta = eta_all[in_cone]; cphi = phi_all[in_cone]
            else: cpt = np.array([])
            if cpt.sum() > 0:
                pT_c = cpt.sum(); E_ = (cpt*np.cosh(ceta)).sum()
                px_ = (cpt*np.cos(cphi)).sum(); py_ = (cpt*np.sin(cphi)).sum()
                pz_ = (cpt*np.sinh(ceta)).sum()
                mass  = float(np.sqrt(max(0.0, E_**2 - px_**2 - py_**2 - pz_**2)))
                dR_c  = _dR(ceta, cphi, eta_p, phi_p)
                width = float((cpt*dR_c).sum() / (pT_c*R_JET + 1e-8))
                props[fl]['pT'].append(pT_c);    props[fl]['mass'].append(mass)
                props[fl]['width'].append(width); props[fl]['nconst'].append(int(in_cone.sum()))
            else:
                for k in ['pT', 'mass', 'width', 'nconst']:
                    props[fl][k].append(0.0 if k != 'nconst' else 0)
    for fl in range(N_FL):
        for k in props[fl]: props[fl][k] = np.array(props[fl][k])
    return props

_pcres = {}
for key, r in mass_points.items():
    print(f'  {key} truth ...', flush=True)
    _tp = _parton_cone_measure(r['parts_truth'], r['mask'],     r['parton_feat'])
    print(f'  {key} gen ...',   flush=True)
    _gp = _parton_cone_measure(r['parts_gen'],   r['mask_gen'], r['parton_feat'])
    _pcres[key] = {'truth': _tp, 'gen': _gp, 'label': r['label']}

_PCOBS = [('pT', r'Cone $p_T$ [GeV]', 'GeV'), ('mass', 'Cone mass [GeV]', 'GeV'),
          ('width', 'Cone width / R', ''),     ('nconst', 'Particles in cone', '')]

for key, res in _pcres.items():
    _tp = res['truth']; _gp = res['gen']
    _plot_fl = [fl for fl in range(N_FL) if len(_tp[fl]['pT']) > 0]
    _n_cols  = len(_plot_fl)
    fig, axes_p = plt.subplots(len(_PCOBS), _n_cols, figsize=(3.5*_n_cols, 4*len(_PCOBS)),
                               constrained_layout=True, squeeze=False)
    fig.suptitle(f'Parton-cone comparison — {res["label"]}\n'
                 f'Particles within ΔR<{R_JET} of hard parton axes', fontsize=10)
    for row, (k, ylab, units) in enumerate(_PCOBS):
        for col, fl in enumerate(_plot_fl):
            ax = axes_p[row, col]
            at_f = _tp[fl][k][np.isfinite(_tp[fl][k])]
            ag_f = _gp[fl][k][np.isfinite(_gp[fl][k])]
            if len(at_f) == 0 and len(ag_f) == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
            else:
                ref  = at_f if len(at_f) else ag_f
                _lo  = float(np.percentile(ref, 0.5)); _hi = float(np.percentile(ref, 99.5))
                bins = np.linspace(_lo, max(_hi, _lo+1e-6), 50)
                if len(at_f) > 0: ax.hist(at_f, bins=bins, density=True, histtype='step',
                                           lw=1.8, color=TRUTH_COL[fl], label='Truth')
                if len(ag_f) > 0: ax.hist(ag_f, bins=bins, density=True, histtype='step',
                                           lw=1.8, color=GEN_COL[fl], label='Generated', ls='--')
                if len(at_f) > 1 and len(ag_f) > 1:
                    _score_text(ax, _hist_jsd(at_f, ag_f, bins), _wass(at_f, ag_f), units=units)
            ax.set_xlabel(ylab, fontsize=8)
            if col == 0: ax.set_ylabel('Density', fontsize=7)
            if row == 0: ax.set_title(FLAVOR_NAMES[fl], fontsize=10,
                                       fontweight='bold', color=TRUTH_COL[fl])
            if row == 0 and col == 0: ax.legend(fontsize=7)

    out = f'{OUT_DIR}/parton_cone_{key}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight');  plt.close(fig)
    print(f'  -> {out}')

print('\nAll plots saved to', OUT_DIR)
