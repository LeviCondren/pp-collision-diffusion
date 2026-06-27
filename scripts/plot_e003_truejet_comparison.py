#!/usr/bin/env python3
"""
E003 — sampled-log_npart vs true-log_npart comparison plots.

Overlays three curves per observable per process:
  truth        — validation data ground truth
  gen_sampled  — E002a: stage-1 samples log_npart freely
  gen_truejet  — E002b: ground-truth log_npart supplied to stage-2

Usage:
    python plot_e003_truejet_comparison.py \
        --sampled_dir .../infer_20k_sampled \
        --truejet_dir .../infer_20k_truejet \
        --out_dir     .../figures/E003_truejet_comparison
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance as _wass

p = argparse.ArgumentParser()
p.add_argument('--sampled_dir', type=str, required=True)
p.add_argument('--truejet_dir', type=str, required=True)
p.add_argument('--out_dir',     type=str, required=True)
p.add_argument('--n_events',    type=int, default=20000)
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
N = args.n_events

PROCS = ['dijet', 'zjets', 'ttbar', 'wjets', 'wprime']
PROC_LABEL = {
    'dijet':  'Dijet',
    'zjets':  'Z+jets',
    'ttbar':  r'$t\bar{t}$',
    'wjets':  'W+jets',
    'wprime': r"W' (500,100)",
}

C_TRUTH   = '#1f77b4'
C_SAMPLED = '#ff7f0e'
C_TRUE    = '#2ca02c'

# ── helpers ───────────────────────────────────────────────────────────────────
def jsd(a, b, bins):
    eps = 1e-10
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    ha = ha.astype(float) + eps;  hb = hb.astype(float) + eps
    ha /= ha.sum();                hb /= hb.sum()
    m = 0.5 * (ha + hb)
    return float(np.clip(0.5*(ha*np.log(ha/m)).sum() + 0.5*(hb*np.log(hb/m)).sum(), 0, None))

def score(ax, j_s, j_t, pos=(0.97, 0.03)):
    ax.text(*pos,
            f'JSD sampled={j_s:.4f}\nJSD truejet={j_t:.4f}',
            transform=ax.transAxes, fontsize=6.5, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='none'))

def _npz(directory, proc):
    for suffix in (f'{proc}_rank00_of01.npz', f'{proc}_20k.npz', f'{proc}.npz'):
        path = os.path.join(directory, suffix)
        if os.path.exists(path):
            return np.load(path)
    raise FileNotFoundError(f'{proc} not found in {directory}')

# ── load ──────────────────────────────────────────────────────────────────────
print('Loading inference outputs...')
data = {}
for proc in PROCS:
    try:
        ds = _npz(args.sampled_dir, proc)
        dt = _npz(args.truejet_dir, proc)
    except FileNotFoundError as e:
        print(f'  skip {proc}: {e}')
        continue
    data[proc] = dict(
        pt  = ds['parts_truth'][:N],
        ps  = ds['parts_gen'][:N],
        pg  = dt['parts_gen'][:N],
        mt  = ds['mask'][:N].astype(bool),
        ms  = ds['mask_gen'][:N].astype(bool),
        mg  = dt['mask_gen'][:N].astype(bool),
    )
    ns = data[proc]['ms'].sum(1).mean()
    ng = data[proc]['mg'].sum(1).mean()
    nt = data[proc]['mt'].sum(1).mean()
    print(f'  {proc:8s}  truth npart={nt:.1f}  sampled={ns:.1f}  truejet={ng:.1f}')

active = [p for p in PROCS if p in data]

# ── Plot 1: particle-level (η, φ, pT, multiplicity) ──────────────────────────
print('\nPlot 1: particle distributions...')
fig, axes = plt.subplots(len(active), 4, figsize=(20, 4*len(active)))
if len(active) == 1:
    axes = axes[np.newaxis, :]
fig.suptitle('E003 — particle distributions: sampled vs true log_npart', fontsize=11)

for row, proc in enumerate(active):
    d = data[proc]
    mt, ms, mg = d['mt'], d['ms'], d['mg']

    eta_t = d['pt'][mt, 0];   eta_s = d['ps'][ms, 0];   eta_g = d['pg'][mg, 0]
    phi_t = np.arctan2(d['pt'][mt,1], d['pt'][mt,2])
    phi_s = np.arctan2(d['ps'][ms,1], d['ps'][ms,2])
    phi_g = np.arctan2(d['pg'][mg,1], d['pg'][mg,2])
    pT_t  = np.exp(d['pt'][mt, 3])
    pT_s  = np.exp(np.clip(d['ps'][ms, 3], -10, 10))
    pT_g  = np.exp(np.clip(d['pg'][mg, 3], -10, 10))
    npart_t = mt.sum(1).astype(float)
    npart_s = ms.sum(1).astype(float)
    npart_g = mg.sum(1).astype(float)
    lbl = PROC_LABEL[proc]

    def _h3(ax, arr_t, arr_s, arr_g, bins, xlabel, xscale='linear'):
        kw = dict(density=True, histtype='step', lw=1.5)
        ax.hist(arr_t, bins=bins, color=C_TRUTH,   label='Truth',   **kw)
        ax.hist(arr_s, bins=bins, color=C_SAMPLED, label='Sampled', **kw)
        ax.hist(arr_g, bins=bins, color=C_TRUE,    label='TrueJet', **kw)
        ax.set_xlabel(xlabel); ax.set_ylabel(lbl if ax.get_subplotspec().colspan.start==0 else '')
        if xscale == 'log': ax.set_xscale('log')
        j_s = jsd(arr_s[np.isfinite(arr_s)], arr_t, bins)
        j_t = jsd(arr_g[np.isfinite(arr_g)], arr_t, bins)
        score(ax, j_s, j_t)

    bins_eta  = np.linspace(-5, 5, 60)
    bins_phi  = np.linspace(-np.pi, np.pi, 60)
    pT_hi     = max(np.percentile(pT_t, 99.5), 1)
    bins_pT   = np.logspace(np.log10(0.3), np.log10(pT_hi + 1), 60)
    nhi       = max(npart_t.max(), npart_s.max(), npart_g.max())
    bins_n    = np.linspace(0, nhi + 5, 60)

    _h3(axes[row, 0], eta_t, eta_s, eta_g, bins_eta,  r'Particle $\eta$')
    _h3(axes[row, 1], phi_t, phi_s, phi_g, bins_phi,  r'Particle $\phi$ [rad]')
    _h3(axes[row, 2], pT_t,  pT_s,  pT_g,  bins_pT,  r'Particle $p_T$ [GeV]', xscale='log')
    _h3(axes[row, 3], npart_t, npart_s, npart_g, bins_n, r'$N_\mathrm{part}$')

    axes[row, 0].legend(fontsize=7)

fig.tight_layout()
out = f'{args.out_dir}/particle_dists_comparison.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'  -> {out}')

# ── Plot 2: global event observables (HT, MET, sphericity) ───────────────────
print('Plot 2: global event observables...')

def _pT_arr(parts, mask):
    return np.exp(np.clip(parts[:, :, 3], -10, 10)) * mask

def _HT(parts, mask):   return _pT_arr(parts, mask).sum(1)
def _MET(parts, mask):
    pT = _pT_arr(parts, mask)
    return np.sqrt((pT * parts[:,:,2]).sum(1)**2 + (pT * parts[:,:,1]).sum(1)**2)
def _sph(parts, mask):
    pT = _pT_arr(parts, mask)
    px = pT * parts[:,:,2]; py = pT * parts[:,:,1]
    Sxx=(px**2).sum(1); Syy=(py**2).sum(1); Sxy=(px*py).sum(1)
    d = np.clip(Sxx+Syy, 1e-8, None)
    det = (Sxx*Syy - Sxy**2) / d**2
    lam = (1. - np.sqrt(np.clip(1. - 4.*det, 0, None))) / 2.
    return np.clip(2.*lam, 0, 1)

fig, axes = plt.subplots(len(active), 3, figsize=(15, 4*len(active)))
if len(active) == 1:
    axes = axes[np.newaxis, :]
fig.suptitle('E003 — global observables: sampled vs true log_npart', fontsize=11)

for row, proc in enumerate(active):
    d = data[proc]
    mt = d['mt'].astype(float)[:, :, np.newaxis]  # keep dims for broadcasting
    ms = d['ms'].astype(float)[:, :, np.newaxis]
    mg = d['mg'].astype(float)[:, :, np.newaxis]
    # squeeze back for _pT_arr which expects (N, P)
    mt2 = mt[:,:,0]; ms2 = ms[:,:,0]; mg2 = mg[:,:,0]

    HT_t  = _HT(d['pt'], mt2);  HT_s  = _HT(d['ps'], ms2);  HT_g  = _HT(d['pg'], mg2)
    MET_t = _MET(d['pt'], mt2); MET_s = _MET(d['ps'], ms2); MET_g = _MET(d['pg'], mg2)
    sph_t = _sph(d['pt'], mt2); sph_s = _sph(d['ps'], ms2); sph_g = _sph(d['pg'], mg2)
    lbl   = PROC_LABEL[proc]

    def _h3g(ax, t, s, g, bins, xlabel, xscale='linear'):
        kw = dict(density=True, histtype='step', lw=1.5)
        ax.hist(t, bins=bins, color=C_TRUTH,   label='Truth',   **kw)
        ax.hist(s, bins=bins, color=C_SAMPLED, label='Sampled', **kw)
        ax.hist(g, bins=bins, color=C_TRUE,    label='TrueJet', **kw)
        ax.set_xlabel(xlabel)
        if ax.get_subplotspec().colspan.start == 0: ax.set_ylabel(lbl)
        if xscale == 'log': ax.set_xscale('log')
        j_s = jsd(s[np.isfinite(s)], t, bins)
        j_t = jsd(g[np.isfinite(g)], t, bins)
        score(ax, j_s, j_t)

    ht_hi  = max(np.percentile(HT_t,99.5), 1)
    met_hi = max(np.percentile(MET_t,99.5), 1)
    bins_HT  = np.logspace(np.log10(max(HT_t.min(),1)),  np.log10(ht_hi+1),  60)
    bins_MET = np.logspace(np.log10(max(MET_t.min(),1)), np.log10(met_hi+1), 60)
    bins_sph = np.linspace(0, 1, 50)

    _h3g(axes[row,0], HT_t,  HT_s,  HT_g,  bins_HT,  r'$H_T$ [GeV]',  xscale='log')
    _h3g(axes[row,1], MET_t, MET_s, MET_g, bins_MET, r'$p_T^\mathrm{miss}$ [GeV]', xscale='log')
    _h3g(axes[row,2], sph_t, sph_s, sph_g, bins_sph, 'Sphericity')
    axes[row, 0].legend(fontsize=7)

fig.tight_layout()
out = f'{args.out_dir}/global_obs_comparison.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'  -> {out}')

# ── Summary table: JSD scores ─────────────────────────────────────────────────
print('\n=== JSD summary (lower = better) ===')
print(f'{"proc":8s}  {"obs":12s}  {"sampled":>10s}  {"truejet":>10s}  {"delta":>10s}')
print('-' * 60)

obs_list = [
    ('eta',    lambda d: (d['pt'][d['mt'],0], d['ps'][d['ms'],0], d['pg'][d['mg'],0]),
               np.linspace(-5,5,60)),
    ('pT',     lambda d: (np.exp(d['pt'][d['mt'],3]),
                          np.exp(np.clip(d['ps'][d['ms'],3],-10,10)),
                          np.exp(np.clip(d['pg'][d['mg'],3],-10,10))),
               None),
    ('npart',  lambda d: (d['mt'].sum(1).astype(float),
                          d['ms'].sum(1).astype(float),
                          d['mg'].sum(1).astype(float)),
               None),
    ('HT',     lambda d: (_HT(d['pt'],d['mt'].astype(float)),
                          _HT(d['ps'],d['ms'].astype(float)),
                          _HT(d['pg'],d['mg'].astype(float))),
               None),
]

for proc in active:
    d = data[proc]
    for obs_name, extractor, fixed_bins in obs_list:
        t, s, g = extractor(d)
        s = s[np.isfinite(s)]; g = g[np.isfinite(g)]
        if fixed_bins is None:
            hi = max(np.percentile(t, 99.5), 1)
            lo = max(t.min(), 0.01) if obs_name in ('pT','HT') else t.min()
            fixed_bins = (np.logspace(np.log10(lo), np.log10(hi+1), 60)
                          if obs_name in ('pT','HT')
                          else np.linspace(t.min(), hi, 60))
        j_s = jsd(s, t, fixed_bins)
        j_t = jsd(g, t, fixed_bins)
        delta = j_t - j_s
        flag = '<<' if delta < -0.001 else ('>' if delta > 0.001 else '~')
        print(f'{proc:8s}  {obs_name:12s}  {j_s:10.5f}  {j_t:10.5f}  {delta:+10.5f}  {flag}')
    print()

print(f'\nAll figures saved to {args.out_dir}')
