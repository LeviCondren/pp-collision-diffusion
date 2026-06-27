#!/usr/bin/env python3
"""
Inference on the wprimeGrid holdout region: mX ∈ {300,350}, mY ∈ {300,350}.

Loads the wprimeGrid PET_pp_parton checkpoint, generates particle clouds for
each of the 4 held-out mass points, saves NPZ files, and produces comparison
plots (particle distributions, jet observables, parton-cone observables).

Architecture: PET_pp_parton, NUM_PARTONS=4, NUM_COND=24, NUM_FEAT=6.
"""

import os, sys, json, argparse, time
import numpy as np

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--signal_dir',  type=str,
                   default='/pscratch/sd/l/lcondren/MCsim/wprime_signal')
    p.add_argument('--run_name',    type=str, default='wprimeGrid')
    p.add_argument('--n_events',    type=int, default=20000)
    p.add_argument('--num_steps',   type=int, default=50)
    p.add_argument('--chunk_size',  type=int, default=200)
    p.add_argument('--npart',       type=int, default=500)
    p.add_argument('--proj_dim',    type=int, default=128)
    p.add_argument('--num_layers',  type=int, default=8)
    p.add_argument('--num_gen_layers', type=int, default=2)
    p.add_argument('--gpu_id',      type=int, default=0)
    return p.parse_args()

args = _parse()

os.environ['CUDA_VISIBLE_DEVICES']  = str(args.gpu_id)
os.environ['TF_GPU_ALLOCATOR']      = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2')

import ctypes as _ctypes
for _lib in [
    '/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64/libcudart.so.12',
    '/global/common/software/nersc9/cudnn/8.9.3-cuda12/lib/libcudnn.so.8',
]:
    try: _ctypes.CDLL(_lib, mode=_ctypes.RTLD_GLOBAL)
    except OSError: pass

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.random.set_seed(42)
print(f'Visible TF GPUs: {len(gpus)}')

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts)
from PET_pp_parton import PET_pp_parton

NUM_PARTONS = 4
PARTON_FEAT = 6
NUM_COND    = NUM_PARTONS * PARTON_FEAT   # 24
NUM_FEAT    = 6

HOLDOUT_POINTS = [(300, 300), (300, 350), (350, 300), (350, 350)]

CKPT_DIR  = os.path.join(args.signal_dir, 'checkpoints', args.run_name)
CKPT_PATH = os.path.join(CKPT_DIR, 'pet_pp.weights.h5')
STATS_PATH = os.path.join(CKPT_DIR, 'normalisation_stats.json')
OUT_DIR   = os.path.join(CKPT_DIR, 'infer_holdout')
PLOT_DIR  = os.path.join(CKPT_DIR, 'plots_holdout')
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Normalisation stats ───────────────────────────────────────────────────────
with open(STATS_PATH) as fh:
    stats = json.load(fh)

jet_mean  = float(stats['jet_mean'][0])
jet_std   = float(stats['jet_std'][0])
part_mean = np.array(stats['part_mean'], dtype=np.float32)
part_std  = np.array(stats['part_std'],  dtype=np.float32)
cond_mean = np.array(stats['cond_mean'], dtype=np.float32)   # (24,)
cond_std  = np.array(stats['cond_std'],  dtype=np.float32)   # (24,)

# ── Load model ────────────────────────────────────────────────────────────────
model = PET_pp_parton(
    num_feat=NUM_FEAT, num_jet=1,
    num_cond=NUM_COND,
    num_partons=NUM_PARTONS,
    parton_feat=PARTON_FEAT,
    num_part=args.npart,
    projection_dim=args.proj_dim,
    local=True, K=5,
    num_layers=args.num_layers,
    num_gen_layers=args.num_gen_layers,
    drop_probability=0.0,
    simple=False, layer_scale=True, talking_head=False,
    mode='generator',
)
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f'Checkpoint not found: {CKPT_PATH}')
model.load_weights(CKPT_PATH)
print(f'Loaded {CKPT_PATH}')

# ── Per-mass-point inference ──────────────────────────────────────────────────
results = {}

for (mX, mY) in HOLDOUT_POINTS:
    label    = f'mX{mX:04d}_mY{mY:04d}'
    out_file = os.path.join(OUT_DIR, f'{label}.npz')

    if os.path.exists(out_file):
        print(f'{label}: already done, loading {out_file}')
        results[label] = dict(np.load(out_file))
        continue

    hdf5_path = os.path.join(args.signal_dir, f'signal_{label}.hdf5')
    print(f'{label}: loading {hdf5_path}')
    with h5py.File(hdf5_path, 'r') as f:
        n_avail = f['particle_features'].shape[0]
        n       = min(args.n_events, n_avail)
        pf      = f['particle_features'][:n].astype(np.float32)   # (N,500,7)
        pt      = f['parton_features'][:n, :NUM_PARTONS, :PARTON_FEAT].astype(np.float32)

    mask    = pf[:, :, 6]           # (N, 500)
    X_raw   = pf[:, :, :6]         # (N, 500, 6)

    cond_raw  = pt.reshape(n, NUM_COND)
    cond_norm = (cond_raw - cond_mean) / cond_std   # (N, 24)

    print(f'  {n} events, mean npart={mask.sum(axis=1).mean():.1f}')

    nsplit       = max(1, n // args.chunk_size)
    actual_chunk = n // nsplit
    print(f'  nsplit={nsplit} ({actual_chunk} events/chunk), num_steps={args.num_steps}')

    t1 = time.perf_counter()
    parts_gen, jets_gen = model.generate(
        cond=cond_norm,
        jet_mean=jet_mean,
        jet_std=jet_std,
        nsplit=nsplit,
        num_steps=args.num_steps,
        use_tqdm=True,
    )
    dt = time.perf_counter() - t1
    print(f'  {dt/60:.2f} min  ({dt/n*1000:.0f} ms/event)')

    log_npart_gen = jets_gen[:, 0] * jet_std + jet_mean
    npart_gen     = np.clip(np.round(np.exp(log_npart_gen)).astype(int), 1, args.npart)
    mask_gen      = (np.arange(args.npart)[None, :] < npart_gen[:, None]).astype(np.float32)
    parts_phys    = (parts_gen * part_std + part_mean) * mask_gen[:, :, None]
    parts_phys[:, :, 5] = np.round(parts_phys[:, :, 5])

    np.savez_compressed(out_file,
        parts_truth  = X_raw,
        parts_gen    = parts_phys,
        mask         = mask,
        mask_gen     = mask_gen,
        parton_feat  = pt,
        mass_x       = np.array([mX]),
        mass_y       = np.array([mY]),
    )
    print(f'  saved -> {out_file}')
    results[label] = {'parts_truth': X_raw, 'parts_gen': parts_phys,
                      'mask': mask, 'mask_gen': mask_gen, 'parton_feat': pt}

# ── Plotting ──────────────────────────────────────────────────────────────────
from scipy.stats import wasserstein_distance as _wass

COLORS = {'truth': '#1f77b4', 'gen': '#ff7f0e'}

FEAT_NAMES  = [r'$\eta$', r'$\sin\phi$', r'$\cos\phi$', r'$\log p_T$',
               'pid', 'charge']
PARTON_NAMES = ['incm+', 'incm-', 'W(X)', 'Z(Y)']
PARTON_FEAT_NAMES = [r'$\log E$', r'$\sin\phi$', r'$\cos\phi$', r'$p_z/E$',
                     'pdg_norm', 'occ']

def _masked(arr, mask):
    """Flatten particles where mask=1."""
    return arr[mask.astype(bool)]

def _jet_obs(parts, mask):
    """Compute per-event jet observables: [npart, pT_sum, eta_lead, phi_lead]."""
    npart   = mask.sum(axis=1)
    log_pt  = parts[:, :, 3]
    pt      = np.exp(np.clip(log_pt, -10, 10)) * mask
    pT_sum  = np.log(pt.sum(axis=1) + 1)
    eta     = parts[:, :, 0]
    sin_phi = parts[:, :, 1]
    cos_phi = parts[:, :, 2]
    # leading particle (highest pT)
    lead_idx  = np.argmax(pt, axis=1)
    N         = len(parts)
    eta_lead  = eta[np.arange(N), lead_idx]
    phi_lead  = np.arctan2(sin_phi[np.arange(N), lead_idx],
                           cos_phi[np.arange(N), lead_idx])
    return npart, pT_sum, eta_lead, phi_lead

def plot_point(label, d, mX, mY):
    truth_m = d['mask'].astype(bool)
    gen_m   = d['mask_gen'].astype(bool)
    pt_arr  = d['parts_truth']
    pg_arr  = d['parts_gen']

    # ── Particle distributions ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'wprimeGrid holdout  mX={mX} mY={mY} GeV — particle features', fontsize=13)
    for fi, ax in enumerate(axes.flat):
        vt = _masked(pt_arr[:, :, fi], truth_m)
        vg = _masked(pg_arr[:, :, fi], gen_m)
        lo = np.percentile(np.concatenate([vt, vg]), 1)
        hi = np.percentile(np.concatenate([vt, vg]), 99)
        bins = np.linspace(lo, hi, 50)
        ax.hist(vt, bins=bins, density=True, histtype='step', lw=1.5,
                color=COLORS['truth'], label='Truth')
        ax.hist(vg, bins=bins, density=True, histtype='step', lw=1.5,
                color=COLORS['gen'],   label='Generated')
        w = _wass(np.clip(vt, lo, hi), np.clip(vg, lo, hi))
        ax.set_xlabel(FEAT_NAMES[fi])
        ax.set_title(f'W={w:.4f}', fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f'particle_dists_{label}.png'), dpi=120)
    plt.close(fig)

    # ── Jet observables ───────────────────────────────────────────────────────
    nt, pT_t, eta_t, phi_t = _jet_obs(pt_arr, d['mask'])
    ng, pT_g, eta_g, phi_g = _jet_obs(pg_arr, d['mask_gen'])
    obs = [(nt,   ng,   'N particles', np.linspace(0, 500, 50)),
           (pT_t, pT_g, r'$\log\Sigma p_T$', np.linspace(pT_t.min(), pT_t.max(), 50)),
           (eta_t, eta_g, r'Lead $\eta$', np.linspace(-5, 5, 50)),
           (phi_t, phi_g, r'Lead $\phi$', np.linspace(-np.pi, np.pi, 50))]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f'wprimeGrid holdout  mX={mX} mY={mY} GeV — jet observables', fontsize=13)
    for ax, (vt, vg, name, bins) in zip(axes, obs):
        ax.hist(vt, bins=bins, density=True, histtype='step', lw=1.5,
                color=COLORS['truth'], label='Truth')
        ax.hist(vg, bins=bins, density=True, histtype='step', lw=1.5,
                color=COLORS['gen'],   label='Generated')
        w = _wass(vt, vg)
        ax.set_xlabel(name)
        ax.set_title(f'W={w:.4f}', fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f'jet_obs_{label}.png'), dpi=120)
    plt.close(fig)

    # ── Parton-cone observables (Σ pT inside ΔR<0.8 of each hard parton) ─────
    pt_hard = d['parton_feat'][:, 2:4, :]   # W(X) and Z(Y) slots, shape (N,2,6)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'wprimeGrid holdout  mX={mX} mY={mY} GeV — parton-cone $p_T$', fontsize=13)
    for si, (ax, slabel) in enumerate(zip(axes, [f'W cone (mX={mX})', f'Z cone (mY={mY})'])):
        sin_p = pt_hard[:, si, 1]
        cos_p = pt_hard[:, si, 2]
        eta_p = pt_hard[:, si, 3]   # pz/E ≈ tanh(η) for massless
        phi_p = np.arctan2(sin_p, cos_p)
        for tag, p_arr, m in [('truth', pt_arr, d['mask']),
                               ('gen',   pg_arr, d['mask_gen'])]:
            deta = p_arr[:, :, 0] - eta_p[:, None]
            dphi = p_arr[:, :, 1] * cos_p[:, None] + p_arr[:, :, 2] * sin_p[:, None]
            dphi = np.arctan2(p_arr[:, :, 1] - sin_p[:, None]*dphi,
                              p_arr[:, :, 2] - cos_p[:, None]*dphi + 1e-9)
            dR   = np.sqrt(deta**2 + dphi**2)
            cone = ((dR < 0.8) * m).astype(bool)
            pt_in = np.exp(np.clip(p_arr[:, :, 3], -10, 10)) * cone
            pT_sum_cone = np.log(pt_in.sum(axis=1) + 1)
            bins = np.linspace(0, pT_sum_cone.max() * 1.05, 50)
            ax.hist(pT_sum_cone, bins=bins, density=True, histtype='step', lw=1.5,
                    color=COLORS[tag], label=tag.capitalize())
        ax.set_xlabel(r'$\log\Sigma p_T^{\rm cone}$')
        ax.set_title(slabel, fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f'parton_cone_{label}.png'), dpi=120)
    plt.close(fig)

print('\n=== Generating plots ===')
for (mX, mY) in HOLDOUT_POINTS:
    label = f'mX{mX:04d}_mY{mY:04d}'
    if label not in results:
        print(f'  {label}: no data, skipping plots')
        continue
    print(f'  {label} ...')
    plot_point(label, results[label], mX, mY)

print(f'\nAll plots written to {PLOT_DIR}')
print('NPZ files written to  ', OUT_DIR)
