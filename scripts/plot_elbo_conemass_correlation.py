#!/usr/bin/env python3
"""Per-event ELBO log-likelihood vs predicted cone mass correlation analysis.

Scores events from an inference NPZ under the truth mass hypothesis, then
builds a Pearson correlation matrix across:
  - truth cone masses (from event_feat_truth, indices 4 and 6)
  - generated cone masses (from jets_gen, indices 5 and 7)
  - per-event ELBO log-likelihoods (particle, jet, and total terms)

Per-event scores are cached to disk so the correlation/plot step can be
re-run cheaply without re-scoring.

Typical usage (E032 holdout, mX=300 mY=300):
  python scripts/plot_elbo_conemass_correlation.py \
    --npz_path /pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid/\
bsm_grid_event_c_stage1_mpi_snap_e127/infer_holdout_e2e_hpc3/bsm_mX0300_mY0300_rank00_of01.npz \
    --ckpt_dir /pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid \
    --run_name bsm_grid_event_c_stage1_mpi_snap_e127 \
    --out_dir /pub/lcondren/wprime_signal_mpi/elbo_correlation \
    --n_events 2000 --n_t 50
"""

import os, sys, json, argparse, time
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_PARTONS    = 4
PARTON_FEAT    = 7
NUM_FEAT       = 6
NUM_JET        = 8
NUM_EVENT_FEAT = 7


def _parse():
    p = argparse.ArgumentParser(
        description='ELBO LL vs cone mass correlation analysis')
    p.add_argument('--npz_path',       required=True,
                   help='Inference NPZ from infer_bsm_grid_event_c_stage1.py')
    p.add_argument('--ckpt_dir',       default=None,
                   help='Checkpoint root (default: derived from npz_path)')
    p.add_argument('--run_name',       default='bsm_grid_event_c_stage1_mpi_snap_e127')
    p.add_argument('--stats_path',     default=None,
                   help='Combined 8-dim stats JSON '
                        '(default: ckpt_dir/normalisation_stats_event_c_stage1.json)')
    p.add_argument('--out_dir',        default='./elbo_correlation')
    p.add_argument('--n_events',       type=int, default=2000,
                   help='Events to score (subset of NPZ)')
    p.add_argument('--n_t',            type=int, default=50,
                   help='Monte Carlo timestep samples per event')
    p.add_argument('--chunk',          type=int, default=200,
                   help='Events per GPU chunk during scoring')
    p.add_argument('--npart',          type=int, default=500)
    p.add_argument('--proj_dim',       type=int, default=128)
    p.add_argument('--num_layers',     type=int, default=8)
    p.add_argument('--num_gen_layers', type=int, default=2)
    p.add_argument('--num_jet_mlp',    type=int, default=512)
    p.add_argument('--gpu_id',         type=int, default=0)
    p.add_argument('--force_rescore',  action='store_true',
                   help='Recompute scores even if cached file exists')
    return p.parse_args()


args = _parse()

os.environ['CUDA_VISIBLE_DEVICES']  = str(args.gpu_id)
os.environ['TF_GPU_ALLOCATOR']      = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
tf.random.set_seed(42)
print(f'Visible GPUs: {len(gpus)}')

os.makedirs(args.out_dir, exist_ok=True)

# ── Load inference NPZ ────────────────────────────────────────────────────────

npz     = np.load(args.npz_path)
N_total = npz['parts_truth'].shape[0]
N       = min(args.n_events, N_total)
print(f'NPZ: {os.path.basename(args.npz_path)}  (using {N}/{N_total} events)')

mass_x = float(npz['mass_x'])
mass_y = float(npz['mass_y'])
print(f'Truth mass: m_X={mass_x:.0f} GeV  m_Y={mass_y:.0f} GeV')

parts_truth_raw = npz['parts_truth'][:N, :args.npart, :NUM_FEAT].astype(np.float32)
mask_2d         = npz['mask'][:N, :args.npart].astype(np.float32)
event_feat_norm = npz['event_feat_truth'][:N].astype(np.float32)   # (N, 7) normalized
jets_gen        = npz['jets_gen'][:N].astype(np.float32)           # (N, 8) normalized
parton_feat     = npz['parton_feat'][:N].astype(np.float32)        # (N, 4, 7) = part7 raw

# ── Load stats ────────────────────────────────────────────────────────────────

ckpt_dir = args.ckpt_dir
if ckpt_dir is None:
    # Derive: NPZ lives at ckpt_dir/run_name/infer_*/
    ckpt_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.npz_path)))

stats_path = (args.stats_path
              or os.path.join(ckpt_dir, 'normalisation_stats_event_c_stage1.json'))

if not os.path.exists(stats_path):
    alt = os.path.join(ckpt_dir, args.run_name,
                       'normalisation_stats_event_c_stage1.json')
    if os.path.exists(alt):
        stats_path = alt
    else:
        raise FileNotFoundError(f'Stats not found:\n  {stats_path}\n  {alt}')

ckpt_path = os.path.join(ckpt_dir, args.run_name, 'pet_pp.weights.h5')
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

with open(stats_path) as fh:
    stats = json.load(fh)

part_mean  = np.array(stats['part_mean'],  dtype=np.float32)
part_std   = np.array(stats['part_std'],   dtype=np.float32)
cond_mean  = np.array(stats['cond_mean'],  dtype=np.float32)
cond_std   = np.array(stats['cond_std'],   dtype=np.float32)
jet_mean_a = np.array(stats['jet_mean'],   dtype=np.float32)  # (8,)
jet_std_a  = np.array(stats['jet_std'],    dtype=np.float32)  # (8,)
ev_mean    = jet_mean_a[1:]   # (7,)
ev_std     = jet_std_a[1:]    # (7,)

print(f'Stats loaded | part_dims={len(part_mean)}  jet_dims=8')
print(f'  jet_mean[0]={jet_mean_a[0]:.3f}  ev_mean[4]={ev_mean[4]:.3f} (cone_mass_X mean)')

# ── Build ELBO inputs ─────────────────────────────────────────────────────────

mask_3d = mask_2d[:, :, np.newaxis]

# Normalize particles (parts_truth is raw in the NPZ)
x_norm = ((parts_truth_raw - part_mean) / part_std) * mask_3d

# 8-dim jet_norm: col 0 = log_npart_norm, cols 1-7 = event_feat_norm
n_valid       = mask_2d.sum(axis=1, keepdims=True)
log_npart     = np.log(np.maximum(n_valid, 1.0))
log_npart_n   = (log_npart - jet_mean_a[0]) / jet_std_a[0]
jet_norm      = np.concatenate([log_npart_n, event_feat_norm], axis=1)  # (N, 8)

# Conditioning vector (parton_feat = part7, already has truth mass in slots 2,3)
cond_raw  = parton_feat.reshape(N, MAX_PARTONS * PARTON_FEAT)
cond_norm = (cond_raw - cond_mean) / cond_std
pmask     = np.ones((N, MAX_PARTONS), dtype=np.float32)
cond      = np.concatenate([cond_norm, pmask], axis=1)   # (N, 32)

print(f'x_norm={x_norm.shape}  jet_norm={jet_norm.shape}  cond={cond.shape}')

# ── Load model ────────────────────────────────────────────────────────────────

sys.path.insert(0, _SCRIPT_DIR)
from PET_pp_parton_vpar_bsm_event_c_stage1 import PET_pp_parton_vpar_bsm_event_c_stage1

model = PET_pp_parton_vpar_bsm_event_c_stage1(
    num_feat=NUM_FEAT, num_jet=NUM_JET,
    max_partons=MAX_PARTONS, parton_feat=PARTON_FEAT,
    num_event_feat=NUM_EVENT_FEAT,
    num_part=args.npart,
    projection_dim=args.proj_dim,
    local=True, K=5,
    num_layers=args.num_layers,
    num_gen_layers=args.num_gen_layers,
    num_jet_mlp=args.num_jet_mlp,
    drop_probability=0.0,
    simple=False, layer_scale=True, talking_head=False,
    mode='generator',
)
model.load_weights(ckpt_path)
print(f'Loaded: {ckpt_path}')

# ── Per-event ELBO scoring ────────────────────────────────────────────────────

npz_tag      = os.path.splitext(os.path.basename(args.npz_path))[0]
scores_cache = os.path.join(args.out_dir, f'elbo_scores_{npz_tag}.npz')

if os.path.exists(scores_cache) and not args.force_rescore:
    print(f'Loading cached scores: {scores_cache}')
    sc = np.load(scores_cache)
    if int(sc['n_events']) == N and int(sc['n_t']) >= args.n_t:
        part_scores = sc['part_scores']
        jet_scores  = sc['jet_scores']
        print(f'  cached: N={N}  n_t={sc["n_t"]}')
    else:
        print(f'  cache mismatch (N={sc["n_events"]}/{N}, n_t={sc["n_t"]}/{args.n_t}) — rescoring')
        args.force_rescore = True

if not os.path.exists(scores_cache) or args.force_rescore:
    part_acc = np.zeros(N, dtype=np.float64)
    jet_acc  = np.zeros(N, dtype=np.float64)

    t0 = time.perf_counter()
    for s in range(0, N, args.chunk):
        e  = min(s + args.chunk, N)
        nc = e - s

        x_tf   = tf.constant(x_norm[s:e],          dtype=tf.float32)
        jet_tf = tf.constant(jet_norm[s:e],         dtype=tf.float32)
        m2_tf  = tf.constant(mask_2d[s:e],          dtype=tf.float32)
        m3_tf  = tf.constant(mask_3d[s:e].astype(np.float32), dtype=tf.float32)
        c_tf   = tf.constant(cond[s:e],             dtype=tf.float32)
        e_tf   = tf.constant(event_feat_norm[s:e],  dtype=tf.float32)

        p_acc_chunk = np.zeros(nc, dtype=np.float64)
        j_acc_chunk = np.zeros(nc, dtype=np.float64)

        for _ in range(args.n_t):
            t_samp = tf.random.uniform((nc, 1), dtype=tf.float32)
            _, alpha, sigma = model.get_logsnr_alpha_sigma(t_samp)

            # particle term
            eps_p  = tf.random.normal((nc, args.npart, NUM_FEAT), dtype=tf.float32) * m3_tf
            x_t    = alpha[:, None] * x_tf + eps_p * sigma[:, None]
            x_t_m  = x_t * m3_tf
            v_body = model.ema_body([x_t_m, x_t_m[:, :, :2], m3_tf, t_samp], training=False)
            v_pred = m3_tf * model.ema_head(
                [v_body, jet_tf[:, :1], m3_tf, t_samp, c_tf, e_tf], training=False)
            v_true = (alpha[:, None] * eps_p - sigma[:, None] * x_tf) * m3_tf
            sq     = tf.reduce_sum(tf.square(v_true - v_pred), axis=[1, 2])
            n_p    = tf.reduce_sum(m2_tf, axis=1) + 1e-8
            p_acc_chunk += (sq / n_p).numpy()

            # jet term
            eps_j      = tf.random.normal((nc, NUM_JET), dtype=tf.float32)
            jet_t      = alpha * jet_tf + eps_j * sigma
            v_pred_jet = model.ema_jet([jet_t, t_samp, c_tf], training=False)
            v_true_jet = alpha * eps_j - sigma * jet_tf
            j_acc_chunk += tf.reduce_mean(tf.square(v_true_jet - v_pred_jet), axis=1).numpy()

        part_acc[s:e] = p_acc_chunk / args.n_t
        jet_acc[s:e]  = j_acc_chunk / args.n_t

        elapsed = time.perf_counter() - t0
        frac    = e / N
        eta     = elapsed / frac * (1 - frac)
        print(f'  chunk [{s}:{e}]  elapsed {elapsed:.0f}s  ETA {eta:.0f}s')

    part_scores = part_acc.astype(np.float32)
    jet_scores  = jet_acc.astype(np.float32)

    np.savez_compressed(scores_cache,
        part_scores=part_scores,
        jet_scores=jet_scores,
        n_events=np.int32(N),
        n_t=np.int32(args.n_t),
        mass_x=np.float32(mass_x),
        mass_y=np.float32(mass_y),
    )
    print(f'Scores cached → {scores_cache}')

# ELBO LL = negative MSE (higher = better fit)
ll_part  = -part_scores
ll_jet   = -jet_scores
ll_total = ll_part + ll_jet

print(f'\nScore summary (mean ± std):')
print(f'  ELBO_part:  {ll_part.mean():.4f} ± {ll_part.std():.4f}')
print(f'  ELBO_jet:   {ll_jet.mean():.4f} ± {ll_jet.std():.4f}')
print(f'  ELBO_total: {ll_total.mean():.4f} ± {ll_total.std():.4f}')

# ── Unnormalize cone masses ───────────────────────────────────────────────────
# event_feat layout: [MET, sin_phi, cos_phi, pT_X, coneM_X, pT_Y, coneM_Y]
# jets_gen layout:   [log_npart, MET, sin_phi, cos_phi, pT_X, coneM_X, pT_Y, coneM_Y]

def _unnorm(x_n, mean, std):
    """Inverse of log1p + standardisation: returns values in GeV."""
    return np.expm1(np.clip(x_n * std + mean, -5, 20))

cm_X_truth = _unnorm(event_feat_norm[:, 4], ev_mean[4], ev_std[4])
cm_Y_truth = _unnorm(event_feat_norm[:, 6], ev_mean[6], ev_std[6])
cm_X_gen   = _unnorm(jets_gen[:, 5],        ev_mean[4], ev_std[4])
cm_Y_gen   = _unnorm(jets_gen[:, 7],        ev_mean[6], ev_std[6])

print(f'\nCone mass summary (GeV):')
print(f'  coneM_X_truth: {cm_X_truth.mean():.1f} ± {cm_X_truth.std():.1f}')
print(f'  coneM_Y_truth: {cm_Y_truth.mean():.1f} ± {cm_Y_truth.std():.1f}')
print(f'  coneM_X_gen:   {cm_X_gen.mean():.1f} ± {cm_X_gen.std():.1f}')
print(f'  coneM_Y_gen:   {cm_Y_gen.mean():.1f} ± {cm_Y_gen.std():.1f}')

# ── Build correlation matrix ──────────────────────────────────────────────────

LABELS = [
    'coneM_X_truth', 'coneM_Y_truth',
    'coneM_X_gen',   'coneM_Y_gen',
    'LL_part',       'LL_jet',        'LL_total',
]
mat = np.stack([cm_X_truth, cm_Y_truth, cm_X_gen, cm_Y_gen,
                ll_part, ll_jet, ll_total], axis=1)   # (N, 7)

# Remove extreme outliers (5σ clip per variable)
mu_v  = mat.mean(axis=0)
sig_v = mat.std(axis=0)
ok    = np.all(np.abs(mat - mu_v) < 5 * sig_v, axis=1)
mat_c = mat[ok]
N_c   = mat_c.shape[0]
print(f'\nAfter 5σ clip: {N_c}/{N} events retained')

corr = np.corrcoef(mat_c.T)   # (7, 7)

print('\n── Pearson r (LL_total vs others) ──')
for j, lbl in enumerate(LABELS[:-1]):
    print(f'  LL_total vs {lbl:20s}: r = {corr[6, j]:+.3f}')

# ── Plot ──────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

tag   = f'mX{mass_x:.0f}_mY{mass_y:.0f}'
title = (f'E032 (snap_e127) | m_X={mass_x:.0f} GeV  m_Y={mass_y:.0f} GeV'
         f' | N={N_c}  n_t={args.n_t}')

fig = plt.figure(figsize=(20, 9))
gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

# ─── Heatmap ───────────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[:, 0:2])
im  = ax0.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
ax0.set_xticks(range(7))
ax0.set_xticklabels(LABELS, rotation=40, ha='right', fontsize=8.5)
ax0.set_yticks(range(7))
ax0.set_yticklabels(LABELS, fontsize=8.5)
for i in range(7):
    for j in range(7):
        col = 'white' if abs(corr[i, j]) > 0.5 else 'black'
        ax0.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                 fontsize=7.5, color=col, fontweight='bold')
plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04, label='Pearson r')
ax0.set_title(title, fontsize=9, pad=8)

# ─── Scatter plots ────────────────────────────────────────────────────────
SCATTER_PAIRS = [
    # (x_col, y_col, x_label, y_label)
    (0, 2, 'coneM_X_truth (GeV)', 'coneM_X_gen (GeV)'),
    (1, 3, 'coneM_Y_truth (GeV)', 'coneM_Y_gen (GeV)'),
    (0, 6, 'coneM_X_truth (GeV)', 'LL_total'),
    (1, 6, 'coneM_Y_truth (GeV)', 'LL_total'),
    (2, 6, 'coneM_X_gen (GeV)',   'LL_total'),
    (3, 6, 'coneM_Y_gen (GeV)',   'LL_total'),
]
scatter_axes = [
    fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
    fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3]),
]
# Use only 4 of the 6 pairs (the truth-vs-gen and LL-vs-gen pairs are most diagnostic)
for idx, (xi, yi, xl, yl) in enumerate(SCATTER_PAIRS[:4]):
    ax = scatter_axes[idx]
    ax.scatter(mat_c[:, xi], mat_c[:, yi], s=2, alpha=0.15, rasterized=True)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    ax.set_title(f'r = {corr[yi, xi]:+.3f}', fontsize=9, pad=3)
    ax.tick_params(labelsize=7)

fig.suptitle('ELBO log-likelihood × cone mass correlations', fontsize=11, y=1.01)

out_pdf = os.path.join(args.out_dir, f'elbo_corr_{tag}_{args.run_name}.pdf')
fig.savefig(out_pdf, bbox_inches='tight', dpi=150)
print(f'\nSaved → {out_pdf}')

# ─── Second figure: LL vs gen cone masses with color = truth cone mass ────────
fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle(title, fontsize=9)

for ax, (xi, yi, xl, lbl_corr) in [
    (ax_a, (2, 6, 'coneM_X_gen (GeV)', f'r(LL,coneM_X_gen)={corr[6,2]:+.3f}')),
    (ax_b, (3, 6, 'coneM_Y_gen (GeV)', f'r(LL,coneM_Y_gen)={corr[6,3]:+.3f}')),
]:
    c_col = mat_c[:, xi - 2]   # truth cone mass (X→col 0, Y→col 1)
    sc = ax.scatter(mat_c[:, xi], mat_c[:, 6], c=c_col, s=3, alpha=0.25,
                    cmap='plasma', rasterized=True)
    plt.colorbar(sc, ax=ax, label='truth cone mass (GeV)')
    ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel('LL_total', fontsize=9)
    ax.set_title(lbl_corr, fontsize=9)

fig2.tight_layout()
out_pdf2 = os.path.join(args.out_dir, f'elbo_corr_scatter_{tag}_{args.run_name}.pdf')
fig2.savefig(out_pdf2, bbox_inches='tight', dpi=150)
print(f'Saved → {out_pdf2}')
