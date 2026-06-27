#!/usr/bin/env python3
"""
Ablation eval: are both process-conditioning paths in proc_label_5proc_p3 active?

Ablation 0 (unmodified): load checkpoint, evaluate val loss as-is.
Ablation 1 (zero proc_token): zero the new cross-attention token before it enters
    cond_set.  Disables the Phase-3 path without touching any weights.
Ablation 2 (zero proc_emb): zero the additive proc_emb contribution to cond_token.
    Disables the pre-existing additive path without touching any weights.

All three ablations load the SAME checkpoint.  Layer names within the head model
are identical across builds (clear_session() resets Keras name counters between
builds).  No weights are modified; the checkpoint format is unchanged.

Usage (interactive, single GPU):
    module load tensorflow/2.15.0
    python3 ablation_proc_paths.py --run_name proc_label_5proc_p3 \
        --data_dir /pscratch/sd/l/lcondren/MCsim/full_event_mixed
"""

import argparse, json, os, sys
import numpy as np

# ── GPU / TF setup ────────────────────────────────────────────────────────────
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['TF_GPU_ALLOCATOR']     = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2')

import ctypes
for _lib in [
    '/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64/libcudart.so.12',
    '/global/common/software/nersc9/cudnn/8.9.3-cuda12/lib/libcudnn.so.8',
]:
    try: ctypes.CDLL(_lib, ctypes.RTLD_GLOBAL)
    except OSError: pass

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

import keras
from keras import layers, Input
import h5py

# ── Args ──────────────────────────────────────────────────────────────────────
def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='/pscratch/sd/l/lcondren/MCsim/full_event_mixed')
    p.add_argument('--run_name',   default='proc_label_5proc_p3')
    p.add_argument('--val_start',  type=int, default=400000)
    p.add_argument('--n_val',      type=int, default=10000,
                   help='Events per process for validation')
    p.add_argument('--batch_size', type=int, default=500)
    p.add_argument('--n_batches',  type=int, default=40,
                   help='Batches to average loss over (each draws new random t, eps)')
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()

args = _parse()

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

# ── Import base classes (no horovod involved) ─────────────────────────────────
# Use ProcLabelPET as the base — it already overrides _resnet_vpar to slice the
# parton mask correctly as labels[:, P*PF : P*PF+P] rather than labels[:, P*PF:]
# which would incorrectly include the process-label bits.
from proc_label_train import ProcLabelPET
from PET import FourierProjection
from layers import LayerScale, StochasticDepth

# ── Constants matching training run ──────────────────────────────────────────
PROCESSES    = ['dijet', 'zjets', 'ttbar', 'wjets', 'wprime']
N_PROC       = len(PROCESSES)
MAX_PARTONS  = 6
PARTON_FEAT  = 6
NUM_FEAT     = 6
NUM_JET      = 1
NUM_PART     = 500

CKPT_PATH = (f'{args.data_dir}/checkpoints/{args.run_name}/pet_pp.weights.h5')
STATS_PATH = f'{args.data_dir}/normalisation_stats.json'


# ── Load validation data (numpy, done once before any TF model build) ─────────
def load_val_data():
    with open(STATS_PATH) as fh:
        stats = json.load(fh)

    part_mean = np.array(stats['part_mean'], dtype=np.float32)
    part_std  = np.array(stats['part_std'],  dtype=np.float32)
    jet_mean  = float(stats['jet_mean'][0])
    jet_std   = float(stats['jet_std'][0])

    cond_mean_raw = np.array(stats['cond_mean'], dtype=np.float32)
    cond_std_raw  = np.array(stats['cond_std'],  dtype=np.float32)
    n_cond_feat   = MAX_PARTONS * PARTON_FEAT
    cond_mean     = np.zeros(n_cond_feat, dtype=np.float32)
    cond_std      = np.ones(n_cond_feat,  dtype=np.float32)
    n_fill        = min(len(cond_mean_raw), n_cond_feat)
    cond_mean[:n_fill] = cond_mean_raw[:n_fill]
    cond_std[:n_fill]  = np.where(cond_std_raw[:n_fill] > 0,
                                  cond_std_raw[:n_fill], 1.0)

    all_pf, all_mask, all_cond, all_jet = [], [], [], []

    for proc_idx, proc in enumerate(PROCESSES):
        path = f'{args.data_dir}/{proc}.hdf5'
        with h5py.File(path, 'r') as f:
            n_total  = f['particle_features'].shape[0]
            r0       = min(args.val_start, n_total)
            r1       = min(r0 + args.n_val, n_total)
            pf_raw   = f['particle_features'][r0:r1].astype(np.float32)
            part_raw = f['parton_features'][r0:r1].astype(np.float32)

            if 'n_partons' in f:
                n_par       = f['n_partons'][r0:r1].astype(np.int32)
                parton_mask = (np.arange(MAX_PARTONS)[None, :] <
                               n_par[:, None]).astype(np.float32)
            else:
                p_norms = np.linalg.norm(part_raw, axis=2)
                valid   = (p_norms > 1e-6).astype(np.float32)
                if valid.shape[1] < MAX_PARTONS:
                    pad   = np.zeros((len(valid), MAX_PARTONS - valid.shape[1]),
                                     dtype=np.float32)
                    valid = np.concatenate([valid, pad], axis=1)
                parton_mask = valid[:, :MAX_PARTONS]

        if part_raw.shape[1] < MAX_PARTONS:
            pad      = np.zeros((len(part_raw), MAX_PARTONS - part_raw.shape[1],
                                 PARTON_FEAT), dtype=np.float32)
            part_raw = np.concatenate([part_raw, pad], axis=1)
        part_raw = part_raw[:, :MAX_PARTONS, :]

        mask      = pf_raw[:, :NUM_PART, 6].astype(np.float32)
        pf6       = pf_raw[:, :NUM_PART, :6]
        npart     = mask.sum(axis=1, keepdims=True)
        jet       = (np.log(np.maximum(npart, 1.0)) - jet_mean) / jet_std

        cond_raw  = part_raw.reshape(len(part_raw), n_cond_feat)
        cond_norm = (cond_raw - cond_mean) / cond_std
        proc_oh   = np.zeros((len(cond_norm), N_PROC), dtype=np.float32)
        proc_oh[:, proc_idx] = 1.0
        cond      = np.concatenate([cond_norm, parton_mask, proc_oh], axis=1)

        pf6_norm  = (pf6 - part_mean) / part_std * mask[:, :, None]

        all_pf.append(pf6_norm)
        all_mask.append(mask)
        all_cond.append(cond)
        all_jet.append(jet)
        print(f'  loaded val {proc}: {len(pf6_norm)} events', flush=True)

    return (np.concatenate(all_pf),   np.concatenate(all_mask),
            np.concatenate(all_cond), np.concatenate(all_jet))


# ── Model variants ────────────────────────────────────────────────────────────
# Ablation1 and Ablation2 inherit from ProcLabelPET (proc_label_train.py).
# ProcLabelPET already overrides _resnet_vpar with the correct parton-mask slice.
# The ablations only override _build_vpar_generator_head to zero specific paths.
# The unmodified case just uses ProcLabelPET directly.

def _ablation_head_impl(self, zero_proc_token: bool, zero_proc_emb: bool):
    D  = self.projection_dim
    nh = self.num_heads
    kd = D // nh
    P  = self.max_partons
    PF = self.parton_feat

    inp_encoded = Input(shape=(None, D),         name='vph_encoded')
    inp_jet     = Input(shape=(1,),              name='vph_jet')
    inp_mask    = Input(shape=(None, 1),         name='vph_mask')
    inp_time    = Input(shape=(1,),              name='vph_time')
    inp_cond    = Input(shape=(self.num_cond,),  name='vph_cond')

    parton_feat_flat = inp_cond[:, :P * PF]
    parton_mask_in   = inp_cond[:, P * PF : P * PF + P]
    proc_label_in    = inp_cond[:, P * PF + P :]

    parton_tokens = layers.Reshape((P, PF))(parton_feat_flat)
    parton_emb    = layers.Dense(D)(parton_tokens)
    parton_emb    = StochasticDepth(self.feature_drop)(parton_emb)

    # Phase-3 path: proc_token as cross-attention key/value
    proc_token = layers.Dense(D, activation='gelu')(proc_label_in)
    proc_token = layers.Dense(D)(proc_token)
    proc_token = tf.expand_dims(proc_token, axis=1)
    if zero_proc_token:
        proc_token = tf.zeros_like(proc_token)   # Ablation 1
    cond_set = tf.concat([parton_emb, proc_token], axis=1)

    ones_col  = tf.ones_like(parton_mask_in[:, :1])
    attn_mask = tf.cast(
        tf.concat([parton_mask_in, ones_col], axis=1)[:, None, :], tf.bool)

    time_emb   = FourierProjection(inp_time, D)
    jet_emb    = layers.Dense(D)(inp_jet)
    cond_token = layers.Dense(2 * D, activation='gelu')(time_emb + jet_emb)
    cond_token = layers.Dense(D, activation='gelu')(cond_token)

    # Additive path: proc_emb added to cond_token
    proc_emb = layers.Dense(D, activation='gelu')(proc_label_in)
    proc_emb = layers.Dense(D)(proc_emb)
    if not zero_proc_emb:
        cond_token = cond_token + proc_emb          # original path
    # else: proc_emb Dense layers still built (same names), output ignored

    cond_token = tf.tile(cond_token[:, None, :],
                         [1, tf.shape(inp_encoded)[1], 1]) * inp_mask
    encoded = inp_encoded

    for i in range(self.num_gen_layers):
        x   = layers.Add()([cond_token, encoded])
        x1  = layers.GroupNormalization(groups=1)(x)
        upd = layers.MultiHeadAttention(num_heads=nh, key_dim=kd)(
            query=x1, key=x1, value=x1)
        if self.layer_scale:
            upd = LayerScale(self.layer_scale_init, D)(upd, inp_mask)
        x2 = layers.Add()([upd, cond_token])

        x2n   = layers.GroupNormalization(groups=1)(x2)
        cross = layers.MultiHeadAttention(num_heads=nh, key_dim=kd,
                                          name=f'parton_xattn_{i}')(
            query=x2n, key=cond_set, value=cond_set,
            attention_mask=attn_mask)
        cross = cross * inp_mask
        if self.layer_scale:
            cross = LayerScale(self.layer_scale_init, D)(cross, inp_mask)
        x2 = layers.Add()([cross, x2])

        x3 = layers.GroupNormalization(groups=1)(x2)
        x3 = layers.Dense(2 * D, activation='gelu')(x3)
        x3 = layers.Dense(D)(x3)
        if self.layer_scale:
            x3 = LayerScale(self.layer_scale_init, D)(x3, inp_mask)
        cond_token = layers.Add()([x3, x2])

    out = layers.GroupNormalization(groups=1)(cond_token + encoded)
    out = layers.Dense(self.num_feat)(out) * inp_mask

    return keras.Model(
        inputs=[inp_encoded, inp_jet, inp_mask, inp_time, inp_cond],
        outputs=out,
        name='vpar_generator_head')


class Ablation1(ProcLabelPET):
    """Ablation 1: zero proc_token before cond_set (disable Phase-3 path)."""
    def _build_vpar_generator_head(self):
        self.num_cond = (self.max_partons * self.parton_feat +
                         self.max_partons + self._num_proc_labels)
        return _ablation_head_impl(self, zero_proc_token=True, zero_proc_emb=False)


class Ablation2(ProcLabelPET):
    """Ablation 2: zero proc_emb additive contribution (disable old path)."""
    def _build_vpar_generator_head(self):
        self.num_cond = (self.max_partons * self.parton_feat +
                         self.max_partons + self._num_proc_labels)
        return _ablation_head_impl(self, zero_proc_token=False, zero_proc_emb=True)


# ── Loss evaluation ───────────────────────────────────────────────────────────
def build_model(model_cls):
    return model_cls(
        num_proc_labels=N_PROC,
        num_feat=NUM_FEAT, num_jet=NUM_JET,
        max_partons=MAX_PARTONS, parton_feat=PARTON_FEAT,
        num_part=NUM_PART,
        projection_dim=128,
        local=True, K=5,
        num_layers=8,
        num_gen_layers=2,
        drop_probability=0.0,
        simple=False, layer_scale=True, talking_head=False,
        mode='generator',
    )


def loss_on_batch(model, x_feat, x_mask, x_jet, y):
    """Compute diffusion loss directly via body + head, bypassing model_part.

    Calling model.model_part() requires matching Input((None,1)) mask exactly;
    calling body and head separately is cleaner and avoids shape ambiguity.
    """
    batch_size = tf.shape(x_feat)[0]
    mask = x_mask[:, :, None]           # (B, N, 1) — body and head both expect this

    t = tf.random.uniform((batch_size, 1))
    _, alpha, sigma = model.get_logsnr_alpha_sigma(t)

    # Part loss ─────────────────────────────────────────────────────────────────
    eps         = tf.random.normal(tf.shape(x_feat), dtype=tf.float32) * mask
    perturbed_x = alpha[:, None] * x_feat + eps * sigma[:, None]

    # Run body then head directly (same computation as model_part)
    body_out = model.body([perturbed_x * mask,
                           perturbed_x[:, :, :2] * mask,
                           mask, t])
    v_pred   = model.head([body_out, x_jet, mask, t, y]) * mask
    v_pred   = tf.reshape(v_pred, (batch_size, -1))
    v_tgt    = alpha[:, None] * eps - sigma[:, None] * x_feat
    v_tgt    = tf.reshape(v_tgt, (batch_size, -1))
    loss_part = (tf.reduce_sum(tf.square(v_tgt - v_pred))
                 / (tf.reduce_sum(x_mask) + 1e-10))

    # Jet loss ──────────────────────────────────────────────────────────────────
    eps2        = tf.random.normal((batch_size, NUM_JET), dtype=tf.float32)
    perturbed_j = alpha * x_jet + eps2 * sigma
    v_pred_j    = model.model_jet([perturbed_j, t, y])
    v_jet       = alpha * eps2 - sigma * x_jet
    loss_jet    = tf.reduce_mean(tf.square(v_pred_j - v_jet))

    return float(loss_part + loss_jet)


def evaluate(model_cls, label, pf, mask, cond, jet, batch_size, n_batches, seed):
    print(f'\n--- {label} ---', flush=True)

    keras.backend.clear_session()
    tf.random.set_seed(seed)

    model = build_model(model_cls)
    model.load_weights(CKPT_PATH)
    print(f'  Loaded {CKPT_PATH}', flush=True)

    N = len(pf)
    rng = np.random.default_rng(seed)
    losses = []

    for b in range(n_batches):
        idx  = rng.choice(N, size=batch_size, replace=False)
        xf   = tf.constant(pf[idx],   dtype=tf.float32)
        xm   = tf.constant(mask[idx], dtype=tf.float32)
        xj   = tf.constant(jet[idx],  dtype=tf.float32)
        y    = tf.constant(cond[idx], dtype=tf.float32)
        loss = float(loss_on_batch(model, xf, xm, xj, y))
        losses.append(loss)
        if (b + 1) % 10 == 0:
            print(f'  batch {b+1}/{n_batches}  running mean={np.mean(losses):.4f}',
                  flush=True)

    mean_loss = float(np.mean(losses))
    std_loss  = float(np.std(losses))
    print(f'  RESULT  {label}: {mean_loss:.4f} ± {std_loss:.4f}  '
          f'(mean over {n_batches} batches of {batch_size})', flush=True)
    return mean_loss


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'Checkpoint : {CKPT_PATH}')
    print(f'Val data   : {args.data_dir}  val_start={args.val_start}  '
          f'n_val={args.n_val} per proc  ({N_PROC} procs)')
    print(f'Eval setup : {args.n_batches} batches × {args.batch_size} events  seed={args.seed}')
    print()

    print('Loading validation data ...', flush=True)
    pf, mask, cond, jet = load_val_data()
    print(f'Total val events: {len(pf):,}', flush=True)

    results = {}
    for label, cls in [
        ('unmodified',               ProcLabelPET),
        ('ablation1_proc_token_zero', Ablation1),
        ('ablation2_proc_emb_zero',   Ablation2),
    ]:
        results[label] = evaluate(
            cls, label, pf, mask, cond, jet,
            args.batch_size, args.n_batches, args.seed)

    print('\n' + '='*60)
    print('ABLATION SUMMARY')
    print('='*60)
    base = results['unmodified']
    for label, val in results.items():
        delta = val - base
        sign  = '+' if delta >= 0 else ''
        print(f'  {label:<35s}  {val:.4f}  ({sign}{delta:.4f})')
    print('='*60)


if __name__ == '__main__':
    main()
