#!/usr/bin/env python3
"""
PCA process-separation diagnostic for the proc-label 5-process model.

Computes per-event summary features from generated particle clouds, then
projects to 2-D with PCA to visualise whether the model has learned to
produce physically distinct events for each process.

Features (one scalar per event):
  multiplicity, HT, log_HT, MET, leading pT, subleading pT,
  mean |eta|, rms |eta|, pT-weighted <|eta|>,
  q25/q50/q75 of pT distribution,
  charged fraction, neutral fraction.

Usage:
    python pca_process_separation.py [--run_name proc_label_5proc_p3] \
                                     [--data_dir ...] \
                                     [--source truth|gen|both]
"""

import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROCESSES = ['dijet', 'zjets', 'ttbar', 'wjets', 'wprime']
PROC_LABEL = {
    'dijet':  'Dijet',
    'zjets':  r'Z+jets',
    'ttbar':  r'$t\bar{t}$',
    'wjets':  'W+jets',
    'wprime': "W'",
}
COLORS = {
    'dijet':  '#1f77b4',
    'zjets':  '#ff7f0e',
    'ttbar':  '#2ca02c',
    'wjets':  '#d62728',
    'wprime': '#9467bd',
}


# ── Feature extraction ────────────────────────────────────────────────────────

def event_features(parts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute (N, F) summary feature matrix from particle clouds.

    Parameters
    ----------
    parts : (N, P, 6)  float  —  [eta, sin_phi, cos_phi, log_pT, pid, charge]
    mask  : (N, P)     float  —  1 = valid particle

    Returns
    -------
    feats : (N, 14) float32
    """
    N = len(parts)
    m  = mask.astype(bool)                          # (N, P)
    pT = np.exp(np.clip(parts[:, :, 3], -6, 6))    # (N, P)  raw pT
    pT_masked = np.where(m, pT, 0.0)

    eta   = parts[:, :, 0]
    sinp  = parts[:, :, 1]
    cosp  = parts[:, :, 2]
    pid   = parts[:, :, 4]
    chg   = parts[:, :, 5]

    # Multiplicity
    mult  = m.sum(axis=1).astype(np.float32)

    # HT and log HT
    HT    = pT_masked.sum(axis=1)
    logHT = np.log(np.clip(HT, 1e-3, None))

    # MET (missing transverse momentum magnitude)
    METx  = (pT_masked * sinp).sum(axis=1)
    METy  = (pT_masked * cosp).sum(axis=1)
    MET   = np.sqrt(METx**2 + METy**2)

    # Leading and subleading pT
    pT_sorted = np.sort(pT_masked, axis=1)[:, ::-1]
    lead_pT   = pT_sorted[:, 0]
    sublead_pT = pT_sorted[:, 1] if pT_sorted.shape[1] > 1 else np.zeros(N)

    # pT quantiles (computed over valid particles only)
    q25 = np.zeros(N); q50 = np.zeros(N); q75 = np.zeros(N)
    for i in range(N):
        vals = pT[i][m[i]]
        if len(vals) == 0:
            continue
        q25[i], q50[i], q75[i] = np.percentile(vals, [25, 50, 75])

    # Eta statistics
    abs_eta    = np.abs(eta)
    mean_aeta  = (np.where(m, abs_eta, 0.0).sum(axis=1) /
                  np.where(m, 1.0, 0.0).sum(axis=1).clip(1))
    rms_aeta   = np.sqrt((np.where(m, abs_eta**2, 0.0).sum(axis=1) /
                          np.where(m, 1.0, 0.0).sum(axis=1).clip(1)))
    pT_wt_eta  = (np.where(m, pT * abs_eta, 0.0).sum(axis=1) /
                  pT_masked.sum(axis=1).clip(1e-6))

    # Charge fractions
    chg_sum  = np.where(m, np.abs(chg), 0.0).sum(axis=1)
    chg_frac = chg_sum / mult.clip(1)
    neut_frac = 1.0 - chg_frac

    feats = np.column_stack([
        mult, HT, logHT, MET,
        lead_pT, sublead_pT,
        q25, q50, q75,
        mean_aeta, rms_aeta, pT_wt_eta,
        chg_frac, neut_frac,
    ]).astype(np.float32)

    return feats


FEAT_NAMES = [
    'multiplicity', 'HT', 'log HT', 'MET',
    'lead pT', 'sublead pT',
    'pT q25', 'pT q50', 'pT q75',
    r'mean |η|', r'rms |η|', r'pT-wt |η|',
    'charged frac', 'neutral frac',
]


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', type=str, default='proc_label_5proc_p3')
    p.add_argument('--data_dir', type=str,
                   default='/pscratch/sd/l/lcondren/MCsim/full_event_mixed')
    p.add_argument('--infer_dir', type=str, default=None,
                   help='Directory of {proc}_20k.npz (default: checkpoints/{run_name}/infer_20k)')
    p.add_argument('--out_dir',  type=str, default=None,
                   help='Output directory for plots (default: checkpoints/{run_name}/pca)')
    p.add_argument('--source',   choices=['truth', 'gen', 'both'], default='both',
                   help='Use truth particles, generated particles, or overlay both')
    p.add_argument('--n_events', type=int, default=5000,
                   help='Events per process (subsample for speed)')
    p.add_argument('--n_comp',   type=int, default=6,
                   help='Number of PCA components to compute')
    return p.parse_args()


def main():
    args = _parse()

    ckpt_base = f'{args.data_dir}/checkpoints'
    infer_dir = args.infer_dir or f'{ckpt_base}/{args.run_name}/infer_20k'
    out_dir   = args.out_dir   or f'{ckpt_base}/{args.run_name}/pca'
    os.makedirs(out_dir, exist_ok=True)

    sources = ['truth', 'gen'] if args.source == 'both' else [args.source]

    for source in sources:
        parts_key = 'parts_truth' if source == 'truth' else 'parts_gen'
        mask_key  = 'mask'        if source == 'truth' else 'mask_gen'

        print(f'\n=== source={source} ===')
        all_feats  = []
        all_labels = []

        for proc in PROCESSES:
            npz = f'{infer_dir}/{proc}_20k.npz'
            if not os.path.exists(npz):
                print(f'  {proc}: missing {npz}, skipping')
                continue
            d     = np.load(npz)
            parts = d[parts_key]
            mask  = d[mask_key]
            N     = min(args.n_events, len(parts))
            parts = parts[:N]; mask = mask[:N]
            feats = event_features(parts, mask)
            all_feats.append(feats)
            all_labels.extend([proc] * N)
            print(f'  {proc}: {N} events  feat_shape={feats.shape}')

        if not all_feats:
            print(f'  No data found in {infer_dir}')
            continue

        X      = np.concatenate(all_feats, axis=0)
        labels = np.array(all_labels)

        # Replace any NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        pca    = PCA(n_components=args.n_comp)
        X_pca  = pca.fit_transform(X_sc)

        ev     = pca.explained_variance_ratio_
        print(f'  Explained variance: {[f"{v*100:.1f}%" for v in ev]}')

        # ── Plot 1: PC1 vs PC2, PC3 vs PC4 ───────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{args.run_name}  —  PCA process separation  ({source})',
                     fontsize=12)

        pairs = [(0, 1), (2, 3)] if args.n_comp >= 4 else [(0, 1)]
        for ax, (i, j) in zip(axes, pairs):
            for proc in PROCESSES:
                sel = labels == proc
                if not sel.any():
                    continue
                ax.scatter(X_pca[sel, i], X_pca[sel, j],
                           c=COLORS[proc], label=PROC_LABEL[proc],
                           s=2, alpha=0.3, rasterized=True)
            ax.set_xlabel(f'PC{i+1}  ({ev[i]*100:.1f}%)', fontsize=10)
            ax.set_ylabel(f'PC{j+1}  ({ev[j]*100:.1f}%)', fontsize=10)
            ax.legend(markerscale=4, fontsize=8)

        plt.tight_layout()
        path1 = f'{out_dir}/pca_scatter_{source}.pdf'
        fig.savefig(path1, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f'  Saved: {path1}')

        # ── Plot 2: explained variance bar ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(range(1, len(ev)+1), ev * 100, color='steelblue')
        ax.set_xlabel('PC index'); ax.set_ylabel('Explained variance (%)')
        ax.set_title(f'{args.run_name}  PCA explained variance  ({source})')
        plt.tight_layout()
        path2 = f'{out_dir}/pca_variance_{source}.pdf'
        fig.savefig(path2, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path2}')

        # ── Plot 3: feature loadings heatmap ─────────────────────────────────
        n_show = min(4, args.n_comp)
        fig, ax = plt.subplots(figsize=(10, 5))
        loading = pca.components_[:n_show]   # (n_show, n_feat)
        im = ax.imshow(loading, aspect='auto', cmap='RdBu_r',
                       vmin=-1, vmax=1)
        ax.set_xticks(range(len(FEAT_NAMES)))
        ax.set_xticklabels(FEAT_NAMES, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels([f'PC{i+1}' for i in range(n_show)], fontsize=8)
        ax.set_title(f'{args.run_name}  PCA feature loadings  ({source})')
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        plt.tight_layout()
        path3 = f'{out_dir}/pca_loadings_{source}.pdf'
        fig.savefig(path3, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path3}')

    print('\nDone.')


if __name__ == '__main__':
    main()
