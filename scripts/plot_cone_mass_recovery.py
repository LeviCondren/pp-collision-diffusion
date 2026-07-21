#!/usr/bin/env python3
"""
Cone-mass recovery diagnostic for E023 truth-conditioned inference (A011-t).

For each holdout mass point, computes:
  - cone_mass_X / cone_mass_Y  from generated particles  (parts_gen  + parton_feat)
  - cone_mass_X / cone_mass_Y  from truth particles       (parts_truth + parton_feat)

Using R=1.0 parton-cone matching (same approach as compare_event_conditioning.py).
This directly tests whether model_part reproduces the cone masses when conditioned
on the true cone_mass event features.

Usage:
  python3 plot_cone_mass_recovery.py \
      --infer_dir /pscratch/.../bsm_grid_event_c_stage1/infer_holdout_truth \
      --out_dir   /pscratch/.../figures/A011t_E023_cone_mass_recovery
"""

import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

R_CONE = 1.0


def cone_mass(parts, mask, parton_feat, boson_slot):
    """Parton-cone invariant mass: collect particles within R_CONE of boson_slot parton.

    parts       : (N, npart, 6)  [eta, sin_phi, cos_phi, log_pT, pid, charge]
    mask        : (N, npart)
    parton_feat : (N, 4, 7)      slot 1=sin_phi, 2=cos_phi, 3=pz/p
    boson_slot  : int            2 = X boson, 3 = Y boson
    """
    float_mask = mask.astype(np.float32)
    valid = mask.astype(bool)

    pT  = np.exp(np.clip(parts[:, :, 3], -10, 10)) * float_mask
    sp  = parts[:, :, 1]
    cp  = parts[:, :, 2]
    eta = np.clip(parts[:, :, 0], -8, 8)
    phi = np.arctan2(sp, cp)

    pze   = np.clip(parton_feat[:, boson_slot, 3], -1 + 1e-7, 1 - 1e-7)
    eta_p = 0.5 * np.log((1 + pze) / (1 - pze))
    phi_p = np.arctan2(parton_feat[:, boson_slot, 1], parton_feat[:, boson_slot, 2])

    deta = eta - eta_p[:, None]
    dphi = (phi - phi_p[:, None] + np.pi) % (2 * np.pi) - np.pi
    dR   = np.sqrt(deta**2 + dphi**2)
    in_c = (dR < R_CONE) & valid

    wt   = pT * in_c
    E_c  = (wt * np.cosh(eta)).sum(1)
    px_c = (wt * cp).sum(1)
    py_c = (wt * sp).sum(1)
    pz_c = (wt * np.sinh(eta)).sum(1)
    m2   = np.maximum(E_c**2 - px_c**2 - py_c**2 - pz_c**2, 0.0)
    return np.sqrt(m2)


def rel_w1(gen, truth):
    r = np.ptp(truth)
    return wasserstein_distance(gen, truth) / r if r > 0 else np.nan


def plot_mass_point(npz_path, out_dir):
    d = np.load(npz_path)
    mX = int(d['mass_x'])
    mY = int(d['mass_y'])
    label = f'mX={mX} mY={mY}'
    os.makedirs(out_dir, exist_ok=True)

    parts_gen   = d['parts_gen'].astype(np.float32)
    parts_truth = d['parts_truth'].astype(np.float32)
    mask_gen    = d['mask_gen'].astype(np.float32)
    mask_truth  = d['mask'].astype(np.float32)
    parton_feat = d['parton_feat'].astype(np.float32)

    results = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, boson_slot, suffix in [(axes[0], 2, 'X'), (axes[1], 3, 'Y')]:
        m_gen   = cone_mass(parts_gen,   mask_gen,   parton_feat, boson_slot)
        m_truth = cone_mass(parts_truth, mask_truth, parton_feat, boson_slot)

        w1 = rel_w1(m_gen, m_truth)
        results.append((suffix, m_truth.mean(), m_gen.mean(), w1))

        hi = max(np.percentile(m_truth, 99.5), np.percentile(m_gen, 99.5))
        bins = np.linspace(0, hi * 1.05, 60)

        ax.hist(m_truth, bins=bins, histtype='step', density=True,
                color='steelblue', linewidth=1.8, label='Truth')
        ax.hist(m_gen,   bins=bins, histtype='step', density=True,
                color='tomato', linewidth=1.8, label='Generated')
        ax.set_xlabel(f'Cone mass {suffix} [GeV]', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'cone_mass_{suffix}  |  rel-W₁={w1:.3f}', fontsize=10)
        ax.legend(frameon=False, fontsize=9)

        print(f'  cone_mass_{suffix}  truth_mean={m_truth.mean():.1f}  '
              f'gen_mean={m_gen.mean():.1f}  rel-W₁={w1:.4f}')

    fig.suptitle(f'Cone-mass recovery (truth-conditioned) — {label}', fontsize=12)
    fig.tight_layout()
    panel_path = os.path.join(out_dir, 'cone_mass_recovery.png')
    fig.savefig(panel_path, dpi=150)
    plt.close(fig)
    print(f'  saved {panel_path}')

    with open(os.path.join(out_dir, 'cone_mass_recovery.csv'), 'w') as f:
        f.write('suffix,truth_mean,gen_mean,rel_W1\n')
        for suffix, tm, gm, w1 in results:
            f.write(f'{suffix},{tm:.4f},{gm:.4f},{w1:.4f}\n')


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--infer_dir', required=True,
                    help='Directory containing bsm_mX*_rank00_of01.npz files')
    pa.add_argument('--out_dir', required=True,
                    help='Output base directory; mass-point subdirs created inside')
    args = pa.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.infer_dir, 'bsm_mX*_rank00_of01.npz')))
    print(f'Found {len(npz_files)} mass point(s) in {args.infer_dir}')

    for npz_path in npz_files:
        fname  = os.path.basename(npz_path)
        parts  = fname.replace('_rank00_of01.npz', '').split('_')
        mx     = parts[1].replace('mX', '').lstrip('0') or '0'
        my     = parts[2].replace('mY', '').lstrip('0') or '0'
        out_dir = os.path.join(args.out_dir, f'{mx}_{my}')
        print(f'\n{"="*50}  mX={mx} mY={my}')
        plot_mass_point(npz_path, out_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
