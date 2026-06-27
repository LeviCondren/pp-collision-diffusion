#!/usr/bin/env python3
"""
A002 — diffusion-step sweep analysis.

Compares eta/logpT/phi distributions across num_steps in {50, 100, 200, 500}
using true-log_npart inference (stage-2 only).  Metrics per process:
  - Normalized-space std (target = 1.0)
  - Wasserstein distance to truth in physical space
  - Fraction of particles with |eta| > 5

Usage:
    python analyze_a002_diffstep_sweep.py \
        --base_dir .../checkpoints/proc_label_5proc_p3 \
        --out_dir  .../figures/A002_diffstep_sweep \
        --stats    .../full_event_mixed/normalisation_stats.json
"""
import os, argparse, json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

p = argparse.ArgumentParser()
p.add_argument('--base_dir', required=True)
p.add_argument('--out_dir',  required=True)
p.add_argument('--stats',    required=True)
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

with open(args.stats) as fh:
    stats = json.load(fh)
part_mean = np.array(stats['part_mean'], dtype=np.float32)
part_std  = np.array(stats['part_std'],  dtype=np.float32)

PROCS = ['dijet', 'zjets', 'ttbar', 'wjets', 'wprime']
PROC_LABEL = {'dijet': 'Dijet', 'zjets': 'Z+jets', 'ttbar': r'$t\bar{t}$',
              'wjets': 'W+jets', 'wprime': r"W' (500,100)"}

STEP_CONFIGS = [
    (50,  'infer_20k_truejet',      '--'),
    (100, 'infer_truejet_steps100', '-'),
    (200, 'infer_truejet_steps200', '-'),
    (500, 'infer_truejet_steps500', '-'),
]
COLORS = {50: '#1f77b4', 100: '#ff7f0e', 200: '#2ca02c', 500: '#d62728'}

def _npz(directory, proc):
    for suffix in (f'{proc}_rank00_of01.npz', f'{proc}_20k.npz'):
        path = os.path.join(directory, suffix)
        if os.path.exists(path):
            return np.load(path)
    return None

# ── Load all outputs ──────────────────────────────────────────────────────────
print('Loading inference outputs...')
data = {}   # data[(steps, proc)] = {'gen': occ_gen, 'truth': occ_truth}
for steps, subdir, _ in STEP_CONFIGS:
    dirpath = os.path.join(args.base_dir, subdir)
    if not os.path.isdir(dirpath):
        print(f'  {steps} steps: directory not found ({dirpath}), skipping')
        continue
    for proc in PROCS:
        d = _npz(dirpath, proc)
        if d is None:
            print(f'  {steps} steps / {proc}: not found, skipping')
            continue
        gen_mask   = d['mask_gen'].astype(bool)
        truth_mask = d['mask'].astype(bool)
        gen_phys   = d['parts_gen'][gen_mask]
        truth_phys = d['parts_truth'][truth_mask]
        # Reverse de-norm to get model's normalized output
        gen_norm   = (gen_phys - part_mean) / part_std
        truth_norm = (truth_phys - part_mean) / part_std
        data[(steps, proc)] = {
            'gen_phys':   gen_phys,
            'truth_phys': truth_phys,
            'gen_norm':   gen_norm,
            'truth_norm': truth_norm,
            'n_events':   d['parts_gen'].shape[0],
        }
        print(f'  {steps:3d} steps / {proc:8s}: {d["parts_gen"].shape[0]} events, '
              f'{gen_mask.sum()} gen particles')

available_steps = sorted({s for s, _ in data.keys()})

# ── Metrics ───────────────────────────────────────────────────────────────────
print('\nComputing metrics...')
rows = []
for steps in available_steps:
    for proc in PROCS:
        key = (steps, proc)
        if key not in data:
            continue
        d = data[key]
        g = d['gen_phys']
        t = d['truth_phys']
        gn = d['gen_norm']

        eta_g, eta_t = g[:, 0], t[:, 0]
        pT_g,  pT_t  = g[:, 3], t[:, 3]
        phi_g  = np.arctan2(g[:, 1], g[:, 2])
        phi_t  = np.arctan2(t[:, 1], t[:, 2])

        row = {
            'steps':    steps,
            'proc':     proc,
            'n_events': d['n_events'],
            # normalized-space std
            'eta_norm_std':   float(gn[:, 0].std()),
            'logpT_norm_std': float(gn[:, 3].std()),
            'sinphi_norm_std':float(gn[:, 1].std()),
            # Wasserstein in physical space
            'eta_W1':   float(wasserstein_distance(eta_g, eta_t)),
            'logpT_W1': float(wasserstein_distance(pT_g,  pT_t)),
            'phi_W1':   float(wasserstein_distance(phi_g,  phi_t)),
            # out-of-acceptance fraction
            'eta_oor':  float((np.abs(eta_g) > 5.0).mean()),
        }
        rows.append(row)
        print(f'  {steps:3d} steps / {proc:8s}  '
              f'eta_norm_std={row["eta_norm_std"]:.4f}  '
              f'eta_W1={row["eta_W1"]:.4f}  '
              f'eta_oor={row["eta_oor"]*100:.2f}%')

# Write CSV
csv_path = os.path.join(args.out_dir, 'summary_table.csv')
if rows:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nWrote {csv_path}')

# ── Plots: eta overlay per process ───────────────────────────────────────────
print('\nGenerating eta overlay plots...')
bins_eta = np.linspace(-7, 7, 80)

for proc in PROCS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{PROC_LABEL[proc]} — η distribution vs diffusion steps (true log_npart)',
                 fontsize=11)

    # Left: physical space
    ax = axes[0]
    # Truth (same across steps — use 50-step or first available)
    ref_key = next(((s, proc) for s in available_steps if (s, proc) in data), None)
    if ref_key:
        truth_eta = data[ref_key]['truth_phys'][:, 0]
        ax.hist(truth_eta, bins=bins_eta, density=True, histtype='step',
                lw=2, color='black', linestyle='--', label='Truth', zorder=10)

    for steps, _, ls in STEP_CONFIGS:
        if (steps, proc) not in data:
            continue
        eta_g = data[(steps, proc)]['gen_phys'][:, 0]
        ax.hist(eta_g, bins=bins_eta, density=True, histtype='step',
                lw=1.5, color=COLORS[steps], linestyle=ls,
                label=f'{steps} steps')

    ax.axvline(-5, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax.axvline(+5, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax.set_xlabel(r'Particle $\eta$ (physical)')
    ax.set_ylabel('Density')
    ax.set_title('Physical space')
    ax.legend(fontsize=8)

    # Right: normalized space
    ax = axes[1]
    bins_norm = np.linspace(-4, 4, 80)
    if ref_key:
        truth_norm_eta = data[ref_key]['truth_norm'][:, 0]
        ax.hist(truth_norm_eta, bins=bins_norm, density=True, histtype='step',
                lw=2, color='black', linestyle='--', label='Truth', zorder=10)

    for steps, _, ls in STEP_CONFIGS:
        if (steps, proc) not in data:
            continue
        eta_n = data[(steps, proc)]['gen_norm'][:, 0]
        ax.hist(eta_n, bins=bins_norm, density=True, histtype='step',
                lw=1.5, color=COLORS[steps], linestyle=ls,
                label=f'{steps} steps  (std={eta_n.std():.3f})')

    ax.set_xlabel(r'Normalized $\eta$')
    ax.set_title('Normalized space (target std = 1.0)')
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = os.path.join(args.out_dir, f'{proc}_eta_steps_overlay.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')

# ── Summary plot: eta_norm_std vs steps per process ───────────────────────────
print('\nGenerating summary metric plot...')
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('A002 — generation quality vs diffusion steps (true log_npart, E000 ckpt)',
             fontsize=11)

metric_keys  = ['eta_norm_std',  'eta_W1',    'eta_oor']
metric_labels = ['η norm-space std\n(target = 1.0)',
                 'η Wasserstein dist\n(truth)',
                 'Fraction |η| > 5\n(out-of-acceptance)']
proc_colors = {'dijet': '#1f77b4', 'zjets': '#ff7f0e', 'ttbar': '#2ca02c',
               'wjets': '#d62728', 'wprime': '#9467bd'}

for col, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
    ax = axes[col]
    for proc in PROCS:
        xs = [r['steps'] for r in rows if r['proc'] == proc]
        ys = [r[mk]      for r in rows if r['proc'] == proc]
        if xs:
            ax.plot(xs, ys, 'o-', color=proc_colors[proc],
                    label=PROC_LABEL[proc], lw=1.5, ms=5)
    if col == 0:
        ax.axhline(1.0, color='gray', ls='--', lw=1, label='Target (1.0)')
    ax.set_xlabel('Diffusion steps')
    ax.set_ylabel(ml)
    ax.set_xscale('log')
    ax.set_xticks(available_steps)
    ax.set_xticklabels(available_steps)
    ax.legend(fontsize=7)

fig.tight_layout()
out = os.path.join(args.out_dir, 'summary_metrics_vs_steps.pdf')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'  -> {out}')

print(f'\nAll outputs in {args.out_dir}')
