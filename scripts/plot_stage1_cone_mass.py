#!/usr/bin/env python3
"""
Plot stage-1 predicted cone masses vs truth particle cone masses.

Shows three distributions per panel for cone-X and cone-Y:
  - Truth particle cone mass (from parts_truth, computed in A025)
  - Stage-1 prediction (decoded from jets_gen)
  - Generated particle cone mass (from parts_gen, after stage-2)

Reveals where the mass-dependent separation is lost: stage-1 predicts
correctly-separated targets, but stage-2 output collapses them.
"""

import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--infer_dir', default=(
    '/pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid/'
    'bsm_grid_event_c_stage1_mpi_snap_e127/infer_holdout_e2e_fixmet'))
p.add_argument('--out_dir', default=(
    '/pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid/'
    'bsm_grid_event_c_stage1_mpi_snap_e127/diag_stage1_cone'))
p.add_argument('--n_events', type=int, default=2000)
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# Normalisation stats (jet_mean/std indices 5 and 7 = log1p(cone_mass_X/Y))
JET_MEAN = np.array([5.606, 2.1416, 0, 0, 7.247, 5.595, 7.252, 5.611])
JET_STD  = np.array([0.302, 1.1405, 1, 1, 0.402, 0.686, 0.365, 0.667])

R_CONE = 0.4
MASS_POINTS = [
    ('mX0250_mY0250', 250, 250),
    ('mX0250_mY0300', 250, 300),
    ('mX0300_mY0250', 300, 250),
    ('mX0300_mY0300', 300, 300),
]
LABELS = ['(250,250)', '(250,300)', '(300,250)', '(300,300)']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


# ── helpers ──────────────────────────────────────────────────────────────────

def decode_stage1_cone(jets_gen, slot):
    """Decode stage-1 predicted cone mass from normalised jets_gen. slot: 5=X, 7=Y."""
    log_mass = jets_gen[:, slot] * JET_STD[slot] + JET_MEAN[slot]
    return np.expm1(log_mass)


def decode_4vec(parts, mask):
    eta = np.clip(parts[:, :, 0], -8, 8)
    sp  = parts[:, :, 1]
    cp  = parts[:, :, 2]
    pT  = np.exp(np.clip(parts[:, :, 3], -10, 10)) * mask
    phi = np.arctan2(sp, cp)
    E   = pT * np.cosh(eta)
    px  = pT * cp
    py  = pT * sp
    pz  = pT * np.sinh(eta)
    return E, px, py, pz, phi, eta


def compute_cone_mass(parts, mask, parton_feat, slot):
    pze   = np.clip(parton_feat[:, slot, 3], -1 + 1e-7, 1 - 1e-7)
    eta_p = 0.5 * np.log((1 + pze) / (1 - pze))
    phi_p = np.arctan2(parton_feat[:, slot, 1], parton_feat[:, slot, 2])
    E, px, py, pz, phi, eta = decode_4vec(parts, mask)
    deta = eta - eta_p[:, None]
    dphi = phi - phi_p[:, None]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    dR   = np.sqrt(deta**2 + dphi**2)
    in_c = (dR < R_CONE) & mask.astype(bool)
    E_c  = (E  * in_c).sum(1)
    px_c = (px * in_c).sum(1)
    py_c = (py * in_c).sum(1)
    pz_c = (pz * in_c).sum(1)
    m2   = np.maximum(E_c**2 - px_c**2 - py_c**2 - pz_c**2, 0.)
    return np.sqrt(m2)


# ── load ──────────────────────────────────────────────────────────────────────

N = args.n_events
print(f'Loading {N} events per mass point from {args.infer_dir}\n')
datasets = []
for tag, mx, my in MASS_POINTS:
    path = os.path.join(args.infer_dir, f'bsm_{tag}_rank00_of01.npz')
    d = np.load(path)
    n = min(N, d['jets_gen'].shape[0])
    ds = {
        'mx': mx, 'my': my,
        'jets_gen':    d['jets_gen'][:n],
        'parts_truth': d['parts_truth'][:n],
        'parts_gen':   d['parts_gen'][:n],
        'mask':        d['mask'][:n],
        'mask_gen':    d['mask_gen'][:n],
        'parton_feat': d['parton_feat'][:n],
    }
    # stage-1 predictions
    ds['s1_cx'] = decode_stage1_cone(ds['jets_gen'], 5)
    ds['s1_cy'] = decode_stage1_cone(ds['jets_gen'], 7)
    # truth particle cone masses
    ds['cx_t'] = compute_cone_mass(ds['parts_truth'], ds['mask'],     ds['parton_feat'], 2)
    ds['cy_t'] = compute_cone_mass(ds['parts_truth'], ds['mask'],     ds['parton_feat'], 3)
    # generated particle cone masses
    ds['cx_g'] = compute_cone_mass(ds['parts_gen'],   ds['mask_gen'], ds['parton_feat'], 2)
    ds['cy_g'] = compute_cone_mass(ds['parts_gen'],   ds['mask_gen'], ds['parton_feat'], 3)
    datasets.append(ds)
    print(f"  ({mx},{my})  s1_cX={ds['s1_cx'].mean():.1f}  truth_cX={ds['cx_t'].mean():.1f}  gen_cX={ds['cx_g'].mean():.1f} |"
          f"  s1_cY={ds['s1_cy'].mean():.1f}  truth_cY={ds['cy_t'].mean():.1f}  gen_cY={ds['cy_g'].mean():.1f}")


# ── separations ───────────────────────────────────────────────────────────────

def sep(datasets, key, group_fn):
    hi = np.mean([ds[key].mean() for ds in datasets if group_fn(ds) == 'hi'])
    lo = np.mean([ds[key].mean() for ds in datasets if group_fn(ds) == 'lo'])
    return hi - lo

cx_grp = lambda ds: 'hi' if ds['mx'] == 300 else 'lo'
cy_grp = lambda ds: 'hi' if ds['my'] == 300 else 'lo'

seps = {
    'truth_cx': sep(datasets, 'cx_t', cx_grp),
    'truth_cy': sep(datasets, 'cy_t', cy_grp),
    's1_cx':    sep(datasets, 's1_cx', cx_grp),
    's1_cy':    sep(datasets, 's1_cy', cy_grp),
    'gen_cx':   sep(datasets, 'cx_g', cx_grp),
    'gen_cy':   sep(datasets, 'cy_g', cy_grp),
}
print(f"\nSeparations (mX or mY 300 vs 250):")
print(f"  cone-X:  truth {seps['truth_cx']:.1f} GeV | stage-1 {seps['s1_cx']:.1f} GeV | gen {seps['gen_cx']:.1f} GeV")
print(f"  cone-Y:  truth {seps['truth_cy']:.1f} GeV | stage-1 {seps['s1_cy']:.1f} GeV | gen {seps['gen_cy']:.1f} GeV")


# ── plot ──────────────────────────────────────────────────────────────────────

KW_TRUTH = dict(histtype='step', density=True, lw=2.0)
KW_S1    = dict(histtype='stepfilled', density=True, lw=1.5, alpha=0.25)
KW_GEN   = dict(histtype='step', density=True, lw=1.5, ls='--')

BINS = np.linspace(0, 700, 71)

BG = '#f8f9fc'
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)
fig.suptitle(
    'Stage-1 predicted vs truth vs generated cone masses  |  E032 epoch 127\n'
    'Solid = truth particles · Filled = stage-1 prediction · Dashed = stage-2 generated',
    fontsize=11, fontweight='bold', color='#111827', y=1.01
)

for ax, cone_key, s1_key, true_mass_key, title, label_prefix in [
    (axes[0], 'cx_t', 's1_cx', 'cx_g', 'Cone X  (slot 2, $m_X$)', 'mX'),
    (axes[1], 'cy_t', 's1_cy', 'cy_g', 'Cone Y  (slot 3, $m_Y$)', 'mY'),
]:
    ax.set_facecolor(BG)
    for ds, c, lab in zip(datasets, COLORS, LABELS):
        truth_vals = ds[cone_key]; truth_vals = truth_vals[truth_vals > 1.]
        s1_vals    = ds[s1_key]
        gen_vals   = ds[true_mass_key]; gen_vals = gen_vals[gen_vals > 1.]
        ax.hist(truth_vals, bins=BINS, color=c, label=lab, **KW_TRUTH)
        ax.hist(s1_vals,    bins=BINS, color=c, **KW_S1)
        ax.hist(gen_vals,   bins=BINS, color=c, **KW_GEN)
    # vlines at true masses
    for ds, c in zip(datasets, COLORS):
        ax.axvline(ds['mx'] if 'X' in title else ds['my'],
                   color=c, lw=0.9, ls=':', alpha=0.6)
    ax.set_xlabel('Cone invariant mass [GeV]', fontsize=10)
    ax.set_ylabel('Norm. density', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='semibold', color='#111827')
    ax.tick_params(labelsize=9)
    ax.grid(True, lw=0.3, alpha=0.4, color='#b0b8c8')
    ax.set_xlim(0, 700)

axes[0].legend(title='$(m_X,\\,m_Y)$', fontsize=9, title_fontsize=9, framealpha=0.9)

# separation annotation table
def sep_str(val, truth_val):
    pct = 100 * val / truth_val if truth_val else 0
    return f'{val:+.1f} GeV ({pct:.0f}% of truth)'

ann = (
    f"Separation (300 vs 250 GeV group mean):\n"
    f"  Cone-X  truth {seps['truth_cx']:.1f} | stage-1 {sep_str(seps['s1_cx'], seps['truth_cx'])} | gen {sep_str(seps['gen_cx'], seps['truth_cx'])}\n"
    f"  Cone-Y  truth {seps['truth_cy']:.1f} | stage-1 {sep_str(seps['s1_cy'], seps['truth_cy'])} | gen {sep_str(seps['gen_cy'], seps['truth_cy'])}"
)
fig.text(0.5, -0.04, ann, ha='center', va='top', fontsize=9,
         color='#374151', family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', fc='#fff', ec='#d0d8e4', lw=1.0))

fig.tight_layout()
out = os.path.join(args.out_dir, 'stage1_cone_mass_separation.png')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f'\nSaved: {out}')


# ── second figure: mean vs true mass scatter ──────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.patch.set_facecolor(BG)
fig2.suptitle(
    'Mean cone mass vs true mass  |  E032 epoch 127\n'
    'Filled circle = stage-1 prediction · Open circle = truth particle · Square = stage-2 generated',
    fontsize=10, fontweight='bold', color='#111827', y=1.01
)

for ax, s1_key, truth_key, gen_key, mx_key, title in [
    (axes2[0], 's1_cx', 'cx_t', 'cx_g', 'mx', 'Cone X  ($m_X$ axis)'),
    (axes2[1], 's1_cy', 'cy_t', 'cy_g', 'my', 'Cone Y  ($m_Y$ axis)'),
]:
    ax.set_facecolor(BG)
    true_masses = [ds[mx_key] for ds in datasets]
    s1_means    = [ds[s1_key].mean()   for ds in datasets]
    truth_means = [ds[truth_key][ds[truth_key] > 1.].mean() for ds in datasets]
    gen_means   = [ds[gen_key][ds[gen_key] > 1.].mean()     for ds in datasets]

    ax.plot(true_masses, s1_means,    'o-', color='#7c3aed', lw=1.8, ms=9,
            label='Stage-1 pred', zorder=3)
    ax.plot(true_masses, truth_means, 'o--', color='#059669', lw=1.8, ms=9,
            mfc='white', label='Truth particles', zorder=3)
    ax.plot(true_masses, gen_means,   's--', color='#dc2626', lw=1.8, ms=8,
            mfc='white', label='Stage-2 generated', zorder=3)
    # reference line: pred = true mass
    mrange = np.array([240, 310])
    ax.plot(mrange, mrange, ':', color='#9ca3af', lw=1.2, label='pred = true mass')

    ax.set_xlabel('True mass [GeV]', fontsize=10)
    ax.set_ylabel('Mean cone mass [GeV]', fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='semibold', color='#111827')
    ax.tick_params(labelsize=9)
    ax.grid(True, lw=0.3, alpha=0.4, color='#b0b8c8')
    ax.legend(fontsize=9, framealpha=0.9)

fig2.tight_layout()
out2 = os.path.join(args.out_dir, 'stage1_cone_mass_scatter.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print(f'Saved: {out2}')
print('\nAll done.')
