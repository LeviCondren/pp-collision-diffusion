#!/usr/bin/env python3
"""
E023 event-feature quality diagnostic.

Compares stage-1 generated event features (from jets_gen[:,1:] in infer_holdout_pred NPZs)
against truth event features (event_feat_truth) for each holdout mass point.

Also compares log_npart: jets_gen[:,0] (generated) vs log(sum(mask_truth)+1) (truth).

Features (all in normalised space, then denormalised to physical units):
  [0] log1p(MET)         [GeV-ish, log scale]
  [1] sin(MET_phi)       [dimensionless]
  [2] cos(MET_phi)       [dimensionless]
  [3] log1p(cone_pT_X)   [GeV-ish, log scale]
  [4] log1p(cone_mass_X) [GeV-ish, log scale]
  [5] log1p(cone_pT_Y)   [GeV-ish, log scale]
  [6] log1p(cone_mass_Y) [GeV-ish, log scale]

Usage:
  python3 plot_e023_event_features.py \
      --pred_dir  /pscratch/.../bsm_grid_event_c_stage1/infer_holdout_pred \
      --truth_dir /pscratch/.../bsm_grid_event_c_stage1/infer_holdout_truth \
      --stats     /pscratch/.../checkpoints_bsm_grid/normalisation_stats_event_c_stage1.json \
      --out_base  /pscratch/.../figures/A011_event_features
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

FEAT_NAMES_NORM = [
    'log1p(MET) [norm]', 'sin(MET_phi)', 'cos(MET_phi)',
    'log1p(cone_pT_X) [norm]', 'log1p(cone_mass_X) [norm]',
    'log1p(cone_pT_Y) [norm]', 'log1p(cone_mass_Y) [norm]',
]
FEAT_NAMES_PHYS = [
    'log1p(MET / GeV)', 'sin(MET_phi)', 'cos(MET_phi)',
    'log1p(cone_pT_X / GeV)', 'log1p(cone_mass_X / GeV)',
    'log1p(cone_pT_Y / GeV)', 'log1p(cone_mass_Y / GeV)',
]

def denorm(x_norm, mean, std):
    return x_norm * std + mean

def wasserstein_rel(gen, truth):
    r = np.ptp(truth)
    return wasserstein_distance(gen, truth) / r if r > 0 else np.nan

def plot_mass_point(pred_path, truth_path, stats, out_dir, mass_label):
    dp = np.load(pred_path)
    dt = np.load(truth_path)

    jet_mean = np.array(stats['jet_mean'], dtype=np.float32)
    jet_std  = np.array(stats['jet_std'],  dtype=np.float32)

    # log_npart: generated vs truth-mask-derived
    lognpart_gen   = dp['jets_gen'][:, 0]
    lognpart_truth = np.log1p(dt['mask'].sum(axis=1).astype(np.float32))

    # Event features: generated and truth, both in normalised space
    ef_gen_norm   = dp['jets_gen'][:, 1:]       # (N, 7) normalised
    ef_truth_norm = dp['event_feat_truth']        # (N, 7) normalised

    # Denormalise to physical space
    ef_gen_phys   = denorm(ef_gen_norm,   jet_mean[1:], jet_std[1:])
    ef_truth_phys = denorm(ef_truth_norm, jet_mean[1:], jet_std[1:])

    os.makedirs(out_dir, exist_ok=True)

    # ── log_npart ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    bins = np.linspace(
        min(lognpart_truth.min(), lognpart_gen.min()),
        max(lognpart_truth.max(), lognpart_gen.max()), 50)
    ax.hist(lognpart_truth, bins=bins, histtype='step', density=True,
            color='steelblue', linewidth=1.5, label='Truth')
    ax.hist(lognpart_gen, bins=bins, histtype='step', density=True,
            color='tomato', linewidth=1.5, label='Stage-1 gen')
    w1 = wasserstein_rel(lognpart_gen, lognpart_truth)
    ax.set_xlabel('log(n_part + 1)')
    ax.set_ylabel('Density')
    ax.set_title(f'{mass_label}  |  log_npart  |  rel-W₁={w1:.3f}')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'log_npart.png'), dpi=150)
    plt.close(fig)
    print(f'  log_npart  rel-W₁={w1:.4f}')

    # ── 7 event features ──────────────────────────────────────────────────────
    rows = []
    for i, (name_norm, name_phys) in enumerate(zip(FEAT_NAMES_NORM, FEAT_NAMES_PHYS)):
        gen   = ef_gen_phys[:, i]
        truth = ef_truth_phys[:, i]
        w1    = wasserstein_rel(gen, truth)
        rows.append((name_norm, truth.mean(), gen.mean(), w1))

        lo = min(np.percentile(truth, 0.5), np.percentile(gen, 0.5))
        hi = max(np.percentile(truth, 99.5), np.percentile(gen, 99.5))
        bins = np.linspace(lo, hi, 50)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(truth, bins=bins, histtype='step', density=True,
                color='steelblue', linewidth=1.5, label='Truth')
        ax.hist(gen, bins=bins, histtype='step', density=True,
                color='tomato', linewidth=1.5, label='Stage-1 gen')
        ax.set_xlabel(name_phys)
        ax.set_ylabel('Density')
        ax.set_title(f'{mass_label}  |  {name_norm}  |  rel-W₁={w1:.3f}')
        ax.legend(frameon=False)
        fig.tight_layout()
        fname = name_norm.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        fig.savefig(os.path.join(out_dir, f'{fname}.png'), dpi=150)
        plt.close(fig)
        print(f'  {name_norm:<30s}  rel-W₁={w1:.4f}')

    # ── summary CSV ───────────────────────────────────────────────────────────
    with open(os.path.join(out_dir, 'summary.csv'), 'w') as f:
        f.write('feature,truth_mean,gen_mean,rel_W1\n')
        for name, tm, gm, w1 in rows:
            f.write(f'{name},{tm:.4f},{gm:.4f},{w1:.4f}\n')

    # ── 2×4 summary panel ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # log_npart in slot 0
    bins = np.linspace(
        min(lognpart_truth.min(), lognpart_gen.min()),
        max(lognpart_truth.max(), lognpart_gen.max()), 40)
    axes[0].hist(lognpart_truth, bins=bins, histtype='step', density=True,
                 color='steelblue', linewidth=1.5, label='Truth')
    axes[0].hist(lognpart_gen, bins=bins, histtype='step', density=True,
                 color='tomato', linewidth=1.5, label='Stage-1 gen')
    w1_npart = wasserstein_rel(lognpart_gen, lognpart_truth)
    axes[0].set_title(f'log_npart  W₁={w1_npart:.3f}', fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)

    for i, (name_norm, name_phys) in enumerate(zip(FEAT_NAMES_NORM, FEAT_NAMES_PHYS)):
        ax  = axes[i + 1]
        gen   = ef_gen_phys[:, i]
        truth = ef_truth_phys[:, i]
        w1    = wasserstein_rel(gen, truth)
        lo = min(np.percentile(truth, 0.5), np.percentile(gen, 0.5))
        hi = max(np.percentile(truth, 99.5), np.percentile(gen, 99.5))
        bins = np.linspace(lo, hi, 40)
        ax.hist(truth, bins=bins, histtype='step', density=True,
                color='steelblue', linewidth=1.5)
        ax.hist(gen,   bins=bins, histtype='step', density=True,
                color='tomato', linewidth=1.5)
        ax.set_title(f'{name_norm}  W₁={w1:.3f}', fontsize=8)

    fig.suptitle(f'E023 stage-1 event features — {mass_label}', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'panel.png'), dpi=150)
    plt.close(fig)
    print(f'  panel -> {out_dir}/panel.png')


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--pred_dir',  required=True)
    pa.add_argument('--truth_dir', required=True,
                    help='truth NPZs (provides event_feat_truth for comparison)')
    pa.add_argument('--stats',     required=True,
                    help='normalisation_stats_event_c_stage1.json')
    pa.add_argument('--out_base',  default='figures/E023_event_features')
    args = pa.parse_args()

    with open(args.stats) as f:
        stats = json.load(f)

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, 'bsm_mX*.npz')))
    print(f'Found {len(pred_files)} mass point(s)')

    for pred_path in pred_files:
        fname = os.path.basename(pred_path)
        truth_path = os.path.join(args.truth_dir, fname)
        if not os.path.exists(truth_path):
            print(f'  WARNING: no matching truth file for {fname}, skipping')
            continue

        # extract mass label from filename
        parts = fname.replace('.npz','').split('_')
        mx = parts[1].replace('mX','')
        my = parts[2].replace('mY','')
        mass_label = f'mX={mx} mY={my}'
        out_dir = os.path.join(args.out_base, f'{mx}_{my}')

        print(f'\n{"="*60}')
        print(f'Mass point {mx}_{my}')
        print(f'{"="*60}')
        plot_mass_point(pred_path, truth_path, stats, out_dir, mass_label)

    print('\nAll mass points done.')


if __name__ == '__main__':
    main()
