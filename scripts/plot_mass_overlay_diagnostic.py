#!/usr/bin/env python3
"""
Diagnostic: overlay jet mass and cone mass across mass points (truth vs generated).

Checks whether the model encodes mass-dependent kinematic structure by comparing
truth and generated distributions across the 4 holdout mass points. If the
generated distributions don't separate the same way truth does, the model hasn't
learned to encode m_X / m_Y in the particle-level kinematics.

Outputs:
  diag_cone_mass_overlay.png  — parton-cone invariant mass, slots 2 and 3
  diag_jet_mass_overlay.png   — anti-kT R=0.4 leading and subleading jet mass
"""

import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyjet import cluster

p = argparse.ArgumentParser()
p.add_argument('--infer_dir', default=(
    '/pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid/'
    'bsm_grid_event_c_stage1_mpi_snap_e127/infer_holdout_e2e_fixmet'))
p.add_argument('--out_dir', default=(
    '/pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid/'
    'bsm_grid_event_c_stage1_mpi_snap_e127/diag_mass_overlay'))
p.add_argument('--n_events', type=int, default=2000)
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

MASS_POINTS = [
    ('mX0250_mY0250', 250, 250),
    ('mX0250_mY0300', 250, 300),
    ('mX0300_mY0250', 300, 250),
    ('mX0300_mY0300', 300, 300),
]
LABELS = ['(250,250)', '(250,300)', '(300,250)', '(300,300)']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
R_CONE = 0.4
R_JET  = 0.4
PT_MIN = 20.


# ── helpers ──────────────────────────────────────────────────────────────────

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
    """Invariant mass of all valid particles within R=0.4 of parton slot."""
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


def compute_jet_masses(parts, mask, n):
    """Anti-kT R=0.4, return leading and subleading jet masses for first n events."""
    lead, sublead = np.zeros(n), np.zeros(n)
    for i in range(n):
        if i % 500 == 0:
            print(f'    clustering {i}/{n} ...', flush=True)
        valid = mask[i].astype(bool)
        if not valid.any():
            continue
        pT  = np.exp(np.clip(parts[i, valid, 3], -10, 10))
        eta = parts[i, valid, 0]
        sp  = parts[i, valid, 1]
        cp  = parts[i, valid, 2]
        phi = np.arctan2(sp, cp)

        arr = np.zeros(valid.sum(),
                       dtype=[('pT','f8'), ('eta','f8'), ('phi','f8'), ('mass','f8')])
        arr['pT'] = pT; arr['eta'] = eta; arr['phi'] = phi
        jets = cluster(arr, R=R_JET, p=-1).inclusive_jets(ptmin=PT_MIN)
        masses = sorted([j.mass for j in jets], reverse=True)
        lead[i]    = masses[0] if len(masses) > 0 else 0.
        sublead[i] = masses[1] if len(masses) > 1 else 0.
    return lead, sublead


# ── load ──────────────────────────────────────────────────────────────────────

N = args.n_events
print(f'Loading {N} events per mass point from {args.infer_dir}\n')
datasets = []
for tag, mx, my in MASS_POINTS:
    path = os.path.join(args.infer_dir, f'bsm_{tag}_rank00_of01.npz')
    d = np.load(path)
    datasets.append({
        'mx': mx, 'my': my,
        'parts_truth': d['parts_truth'][:N],
        'parts_gen':   d['parts_gen'][:N],
        'mask':        d['mask'][:N],
        'mask_gen':    d['mask_gen'][:N],
        'parton_feat': d['parton_feat'][:N],
    })
    print(f'  ({mx},{my}): loaded')


# ── cone masses ───────────────────────────────────────────────────────────────

print('\nCone masses ...')
for ds in datasets:
    ds['cx_t'] = compute_cone_mass(ds['parts_truth'], ds['mask'],     ds['parton_feat'], 2)
    ds['cy_t'] = compute_cone_mass(ds['parts_truth'], ds['mask'],     ds['parton_feat'], 3)
    ds['cx_g'] = compute_cone_mass(ds['parts_gen'],   ds['mask_gen'], ds['parton_feat'], 2)
    ds['cy_g'] = compute_cone_mass(ds['parts_gen'],   ds['mask_gen'], ds['parton_feat'], 3)
    print(f"  ({ds['mx']},{ds['my']})  cone_X: truth {ds['cx_t'].mean():.1f}  gen {ds['cx_g'].mean():.1f}  |  "
          f"cone_Y: truth {ds['cy_t'].mean():.1f}  gen {ds['cy_g'].mean():.1f}")


# ── jet masses ────────────────────────────────────────────────────────────────

print('\nJet masses (anti-kT R=0.4, pT>20) ...')
for ds in datasets:
    print(f"  ({ds['mx']},{ds['my']}) truth ...")
    ds['jl_t'], ds['js_t'] = compute_jet_masses(ds['parts_truth'], ds['mask'],     N)
    print(f"  ({ds['mx']},{ds['my']}) gen ...")
    ds['jl_g'], ds['js_g'] = compute_jet_masses(ds['parts_gen'],   ds['mask_gen'], N)


# ── plot helpers ──────────────────────────────────────────────────────────────

KW = dict(histtype='step', density=True, lw=2.0)

def hist_panel(ax, datasets, key, bins, xlabel, title, vline_key=None):
    for ds, c, lab in zip(datasets, COLORS, LABELS):
        vals = ds[key]
        vals = vals[vals > 1.]   # drop empty-cone / zero-mass entries
        ax.hist(vals, bins=bins, color=c, label=lab, **KW)
    if vline_key:
        for ds, c in zip(datasets, COLORS):
            ax.axvline(ds[vline_key], color=c, lw=1.1, ls='--', alpha=0.55)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('Norm. density', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='semibold', color='#111827')
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.3, alpha=0.4, color='#b0b8c8')
    ax.set_facecolor('#f8f9fc')


# ── Figure 1: Cone masses ─────────────────────────────────────────────────────

BINS_C = np.linspace(0, 600, 51)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.patch.set_facecolor('#f8f9fc')
fig.suptitle(
    'Parton-cone invariant mass  |  E032 epoch 127 + fix_met  |  R=0.4\n'
    'Do generated distributions separate across mass points the way truth does?',
    fontsize=11, fontweight='bold', color='#111827', y=0.995
)

hist_panel(axes[0, 0], datasets, 'cx_t', BINS_C, 'Cone mass [GeV]',
           'Truth — cone X  (slot 2, $m_X$)', vline_key='mx')
hist_panel(axes[0, 1], datasets, 'cx_g', BINS_C, 'Cone mass [GeV]',
           'Generated — cone X  (slot 2, $m_X$)', vline_key='mx')
hist_panel(axes[1, 0], datasets, 'cy_t', BINS_C, 'Cone mass [GeV]',
           'Truth — cone Y  (slot 3, $m_Y$)', vline_key='my')
hist_panel(axes[1, 1], datasets, 'cy_g', BINS_C, 'Cone mass [GeV]',
           'Generated — cone Y  (slot 3, $m_Y$)', vline_key='my')

axes[0, 0].legend(title='$(m_X,\\,m_Y)$', fontsize=8, title_fontsize=8, framealpha=0.9)
axes[0, 1].text(0.97, 0.97, 'Dashed = true mass', transform=axes[0, 1].transAxes,
                fontsize=7.5, va='top', ha='right', color='#555')

fig.tight_layout(rect=[0, 0, 1, 0.96])
out1 = os.path.join(args.out_dir, 'diag_cone_mass_overlay.png')
fig.savefig(out1, dpi=150, bbox_inches='tight', facecolor='#f8f9fc')
plt.close(fig)
print(f'\nSaved: {out1}')


# ── Figure 2: Jet masses ──────────────────────────────────────────────────────

BINS_J = np.linspace(0, 400, 41)

fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))
fig2.patch.set_facecolor('#f8f9fc')
fig2.suptitle(
    'Anti-kT R=0.4 jet mass  |  E032 epoch 127 + fix_met  |  $p_T > 20$ GeV\n'
    'Leading and subleading jets — truth vs generated across mass points',
    fontsize=11, fontweight='bold', color='#111827', y=0.995
)

hist_panel(axes2[0, 0], datasets, 'jl_t', BINS_J, 'Jet mass [GeV]',
           'Truth — leading jet mass')
hist_panel(axes2[0, 1], datasets, 'jl_g', BINS_J, 'Jet mass [GeV]',
           'Generated — leading jet mass')
hist_panel(axes2[1, 0], datasets, 'js_t', BINS_J, 'Jet mass [GeV]',
           'Truth — subleading jet mass')
hist_panel(axes2[1, 1], datasets, 'js_g', BINS_J, 'Jet mass [GeV]',
           'Generated — subleading jet mass')

axes2[0, 0].legend(title='$(m_X,\\,m_Y)$', fontsize=8, title_fontsize=8, framealpha=0.9)

fig2.tight_layout(rect=[0, 0, 1, 0.96])
out2 = os.path.join(args.out_dir, 'diag_jet_mass_overlay.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#f8f9fc')
plt.close(fig2)
print(f'Saved: {out2}')
print('\nAll done.')
