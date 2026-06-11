"""Multi-process training with an explicit one-hot process label in the conditioning.

Extends per_parton_cond_train_nlo_vpar.py with a single key change:

  Conditioning vector format (shape N × num_cond):
    [0 : P*PF]           normalised parton kinematics  (zero-padded)
    [P*PF : P*PF+P]      binary parton mask             (1=valid)
    [P*PF+P : end]       one-hot process label          (n_proc dims)

    num_cond = P*PF + P + n_proc

Without a process label, processes that have kinematically similar parton
configurations (dijet / zjets / wjets all look like 2-4 light partons) are
indistinguishable from the conditioning alone.  The model then learns an average
distribution that matches none of them well.  The one-hot label gives the
generator an unambiguous process identity at both the jet-multiplicity stage and
the particle-cloud stage.

Architecture changes (ProcLabelPET subclass):
  - generator head: process label → Dense(D) embedding added to the global
    conditioning token before tiling over particle positions.  Every
    cross-attention layer sees the process identity.
  - ResNet jet head: process label → Dense(D) embedding added to the masked
    mean-pool of parton tokens before the scale/shift conditioning.

All other hyper-parameters and the training loop are identical to the parent
script.
"""

import horovod.tensorflow.keras as hvd
hvd.init()

import os, sys, argparse, pickle, json, time as _time, gc
import numpy as np
import h5py

import ctypes as _ctypes
for _lib in [
    "/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64/libcudart.so.12",
    "/global/common/software/nersc9/cudnn/8.9.3-cuda12/lib/libcudnn.so.8",
]:
    try: _ctypes.CDLL(_lib, mode=_ctypes.RTLD_GLOBAL)
    except OSError: pass

os.environ["TF_GPU_ALLOCATOR"]     = "cuda_malloc_async"
os.environ["XLA_FLAGS"]            = "--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True)

tf.random.set_seed(1233 + hvd.rank())

if hvd.rank() == 0:
    print(f"Horovod: {hvd.size()} workers", flush=True)
print(f"  rank {hvd.rank()}: local_rank={hvd.local_rank()}, "
      f"GPU={gpus[hvd.local_rank()].name if gpus else 'CPU'}", flush=True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from PET_pp_parton_vpar import PET_pp_parton_vpar
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import schedules, Adam
from tensorflow.keras import layers, Input
from PET import FourierProjection
from layers import LayerScale, StochasticDepth

_FULL_DATA    = '/pscratch/sd/l/lcondren/MCsim/full_event_fpcd_nlo'
_FULL_DATA_LO = '/pscratch/sd/l/lcondren/MCsim/full_event_fpcd'
STATS_PATH    = f'{_FULL_DATA}/normalisation_stats.json'
CKPT_BASE     = f'{_FULL_DATA_LO}/checkpoints_pet_pp'

NUM_FEAT    = 6
NUM_JET     = 1
MAX_PARTONS = 6    # 2 initial + up to 3 final + 1 hard boson slot
PARTON_FEAT = 6
# num_cond is computed at runtime: P*PF + P + n_proc


# ── Process-label–aware model ─────────────────────────────────────────────────

class ProcLabelPET(PET_pp_parton_vpar):
    """PET_pp_parton_vpar with a one-hot process label appended to cond.

    Set self._num_proc_labels BEFORE calling super().__init__() so that
    _build_vpar_generator_head (called inside super.__init__) can extend
    self.num_cond and build correctly-sized Keras Input layers.
    """

    def __init__(self, num_proc_labels, cfg_drop_prob=0.1, **kwargs):
        self._num_proc_labels = num_proc_labels
        self.cfg_drop_prob    = cfg_drop_prob
        super().__init__(**kwargs)

    # ── Updated _split_cond for external callers ──────────────────────────────

    def _split_cond(self, inp_cond):
        n_feat = self.max_partons * self.parton_feat
        n_mask = self.max_partons
        return (inp_cond[:, :n_feat],
                inp_cond[:, n_feat : n_feat + n_mask],
                inp_cond[:, n_feat + n_mask :])

    # ── Generator head: process label → conditioning token ───────────────────

    def _build_vpar_generator_head(self):
        # Extend num_cond *before* the parent wires inputs_cond = Input((num_cond,))
        self.num_cond = (self.max_partons * self.parton_feat +
                         self.max_partons + self._num_proc_labels)

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

        # Split: parton kinematics | parton mask | process label
        parton_feat_flat = inp_cond[:, :P * PF]               # (N, P*PF)
        parton_mask_in   = inp_cond[:, P * PF : P * PF + P]   # (N, P) exact mask bits
        proc_label_in    = inp_cond[:, P * PF + P :]           # (N, n_proc)

        # Per-parton token embedding
        parton_tokens = layers.Reshape((P, PF))(parton_feat_flat)
        parton_emb    = layers.Dense(D)(parton_tokens)
        parton_emb    = StochasticDepth(self.feature_drop)(parton_emb)

        # Process label as a separate cross-attention token (Phase 3)
        proc_token = layers.Dense(D, activation='gelu')(proc_label_in)  # (N, D)
        proc_token = layers.Dense(D)(proc_token)                         # (N, D)
        proc_token = tf.expand_dims(proc_token, axis=1)                  # (N, 1, D)
        cond_set   = tf.concat([parton_emb, proc_token], axis=1)         # (N, P+1, D)

        # Cross-attention key mask: parton positions from parton_mask_in,
        # process token position always valid
        ones_col  = tf.ones_like(parton_mask_in[:, :1])                  # (N, 1)
        attn_mask = tf.cast(
            tf.concat([parton_mask_in, ones_col], axis=1)[:, None, :],
            tf.bool)                                                      # (N, 1, P+1)

        # Global conditioning token: time + log_npart + process label
        time_emb   = FourierProjection(inp_time, D)
        jet_emb    = layers.Dense(D)(inp_jet)
        cond_token = layers.Dense(2 * D, activation="gelu")(time_emb + jet_emb)
        cond_token = layers.Dense(D, activation="gelu")(cond_token)   # (N, D)

        # Process label embedding injected into the conditioning token
        proc_emb   = layers.Dense(D, activation='gelu')(proc_label_in)  # (N, D)
        proc_emb   = layers.Dense(D)(proc_emb)                          # (N, D)
        cond_token = cond_token + proc_emb                              # (N, D)

        # Broadcast over particle positions
        cond_token = tf.tile(cond_token[:, None, :],
                             [1, tf.shape(inp_encoded)[1], 1]) * inp_mask

        encoded = inp_encoded

        for i in range(self.num_gen_layers):
            # 1. Self-attention
            x   = layers.Add()([cond_token, encoded])
            x1  = layers.GroupNormalization(groups=1)(x)
            upd = layers.MultiHeadAttention(num_heads=nh, key_dim=kd)(
                query=x1, key=x1, value=x1)
            if self.layer_scale:
                upd = LayerScale(self.layer_scale_init, D)(upd, inp_mask)
            x2 = layers.Add()([upd, cond_token])

            # 2. Masked cross-attention to parton tokens
            x2n   = layers.GroupNormalization(groups=1)(x2)
            cross = layers.MultiHeadAttention(num_heads=nh, key_dim=kd,
                                              name=f'parton_xattn_{i}')(
                query=x2n, key=cond_set, value=cond_set,
                attention_mask=attn_mask)
            cross = cross * inp_mask
            if self.layer_scale:
                cross = LayerScale(self.layer_scale_init, D)(cross, inp_mask)
            x2 = layers.Add()([cross, x2])

            # 3. FFN
            x3 = layers.GroupNormalization(groups=1)(x2)
            x3 = layers.Dense(2 * D, activation="gelu")(x3)
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

    # ── ResNet jet head: process label added to parton global pool ────────────

    def _resnet_vpar(self, inputs, inputs_time, labels,
                     num_layer=3, mlp_dim=128, dropout=0.0):

        def resnet_dense(input_layer, hidden_size, nlayers=2):
            x        = input_layer
            residual = layers.Dense(hidden_size)(x)
            for _ in range(nlayers):
                x = layers.Dense(hidden_size, activation='swish')(x)
                x = layers.Dropout(dropout)(x)
            x = LayerScale(self.layer_scale_init, hidden_size)(x)
            return residual + x

        D  = self.projection_dim
        P  = self.max_partons
        PF = self.parton_feat

        # Exact mask slice: do not spill into process label dims
        parton_feat_flat = labels[:, :P * PF]               # (N, P*PF)
        parton_mask_in   = labels[:, P * PF : P * PF + P]   # (N, P)
        proc_label_in    = labels[:, P * PF + P :]           # (N, n_proc)

        parton_tokens = tf.reshape(parton_feat_flat, (-1, P, PF))
        parton_emb    = layers.Dense(D)(parton_tokens)

        # Masked mean-pool of valid parton tokens
        mask_expand   = parton_mask_in[:, :, None]
        count         = tf.maximum(
            tf.reduce_sum(parton_mask_in, axis=1, keepdims=True), 1.0)
        parton_global = (tf.reduce_sum(parton_emb * mask_expand, axis=1)
                         / count)                            # (N, D)

        # Add process embedding to the global parton representation
        proc_emb = layers.Dense(D, activation='gelu')(proc_label_in)  # (N, D)
        parton_global = parton_global + proc_emb

        time       = FourierProjection(inputs_time, D)
        cond_token = layers.Dense(D)(parton_global)
        cond_token = layers.Dense(2 * D, activation='gelu')(cond_token + time)
        scale, shift = tf.split(cond_token, 2, -1)

        layer = layers.Dense(D, activation='swish')(inputs)
        layer = layer * (1.0 + scale) + shift

        for _ in range(num_layer - 1):
            layer = layers.LayerNormalization(epsilon=1e-6)(layer)
            layer = resnet_dense(layer, mlp_dim)

        layer   = layers.LayerNormalization(epsilon=1e-6)(layer)
        outputs = layers.Dense(self.num_jet, kernel_initializer="zeros")(layer)
        return outputs


class WeightedPartonShowerPET(PET_pp_parton_vpar):
    """PET_pp_parton_vpar with per-batch |event_weight|-weighted DDPM loss.

    Used when --no_proc_label is set.  Conditioning is parton kinematics +
    parton mask only (num_cond = P*PF + P = 42).  Process identity comes
    entirely from the PDG norm values already encoded in the parton features.
    """

    def _w_norm(self, x):
        abs_w = tf.abs(x['input_weight'])
        return abs_w / (tf.reduce_mean(abs_w) + 1e-10)

    def train_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        mask       = x['input_mask'][:, :, None]
        w          = self._w_norm(x)

        with tf.GradientTape(persistent=True) as tape:
            t = tf.random.uniform((batch_size, 1))
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)

            eps         = tf.random.normal(tf.shape(x['input_features']),
                                           dtype=tf.float32) * mask
            perturbed_x = alpha[:, None] * x['input_features'] + eps * sigma[:, None]

            v_pred_part = self.model_part([
                perturbed_x * mask,
                perturbed_x[:, :, :2] * mask,
                x['input_mask'], x['input_jet'], t, y])
            v_pred_part = tf.reshape(v_pred_part, (batch_size, -1))
            v_part      = alpha[:, None] * eps - sigma[:, None] * x['input_features']
            v_part      = tf.reshape(v_part, (batch_size, -1))

            sq_per_event   = tf.reduce_sum(tf.square(v_part - v_pred_part), axis=1)
            mask_per_event = tf.reduce_sum(x['input_mask'], axis=1)
            mse_per_event  = sq_per_event / (mask_per_event + 1e-10)
            loss_part      = tf.reduce_sum(w * mse_per_event) / (tf.reduce_sum(w) + 1e-10)

            eps         = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
            perturbed_x = alpha * x['input_jet'] + eps * sigma
            v_pred      = self.model_jet([perturbed_x, t, y])
            v_jet       = alpha * eps - sigma * x['input_jet']
            sq_jet      = tf.squeeze(tf.square(v_pred - v_jet), axis=1)
            loss_jet    = tf.reduce_sum(w * sq_jet) / (tf.reduce_sum(w) + 1e-10)

            loss = loss_jet + loss_part

        self.body_optimizer.minimize(loss_part, self.body.trainable_variables, tape=tape)
        trainable_vars = self.model_jet.trainable_variables + self.head.trainable_variables
        self.optimizer.minimize(loss, trainable_vars, tape=tape)

        self.loss_tracker.update_state(loss)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)

        for w_m, ew in zip(self.model_jet.weights, self.ema_jet.weights):
            ew.assign(self.ema * ew + (1 - self.ema) * w_m)
        for w_m, ew in zip(self.head.weights, self.ema_head.weights):
            ew.assign(self.ema * ew + (1 - self.ema) * w_m)
        for w_m, ew in zip(self.body.weights, self.ema_body.weights):
            ew.assign(self.ema * ew + (1 - self.ema) * w_m)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        mask       = x['input_mask'][:, :, None]
        w          = self._w_norm(x)

        t = tf.random.uniform((batch_size, 1))
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)

        eps         = tf.random.normal(tf.shape(x['input_features']),
                                       dtype=tf.float32) * mask
        perturbed_x = alpha[:, None] * x['input_features'] + eps * sigma[:, None]

        v_pred_part = self.model_part([
            perturbed_x * mask,
            perturbed_x[:, :, :2] * mask,
            x['input_mask'], x['input_jet'], t, y])
        v_pred_part = tf.reshape(v_pred_part, (batch_size, -1))
        v_part      = alpha[:, None] * eps - sigma[:, None] * x['input_features']
        v_part      = tf.reshape(v_part, (batch_size, -1))

        sq_per_event   = tf.reduce_sum(tf.square(v_part - v_pred_part), axis=1)
        mask_per_event = tf.reduce_sum(x['input_mask'], axis=1)
        mse_per_event  = sq_per_event / (mask_per_event + 1e-10)
        loss_part      = tf.reduce_sum(w * mse_per_event) / (tf.reduce_sum(w) + 1e-10)

        eps         = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
        perturbed_x = alpha * x['input_jet'] + eps * sigma
        v_pred      = self.model_jet([perturbed_x, t, y])
        v_jet       = alpha * eps - sigma * x['input_jet']
        sq_jet      = tf.squeeze(tf.square(v_pred - v_jet), axis=1)
        loss_jet    = tf.reduce_sum(w * sq_jet) / (tf.reduce_sum(w) + 1e-10)

        loss = loss_jet + loss_part
        self.loss_tracker.update_state(loss)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)
        return {m.name: m.result() for m in self.metrics}


class WeightedProcLabelPET(ProcLabelPET):
    """ProcLabelPET with per-batch |event_weight|-weighted DDPM loss."""

    def _w_norm(self, x):
        abs_w = tf.abs(x['input_weight'])
        return abs_w / (tf.reduce_mean(abs_w) + 1e-10)

    def train_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        mask       = x['input_mask'][:, :, None]
        w          = self._w_norm(x)

        # CFG: randomly null the process label for cfg_drop_prob fraction of events.
        # Both model_part and model_jet see the same dropped conditioning per event.
        P, PF   = self.max_partons, self.parton_feat
        drop    = tf.cast(
            tf.random.uniform([batch_size, 1]) < self.cfg_drop_prob, tf.float32)
        y_train = tf.concat([
            y[:, :P*PF + P],
            y[:, P*PF + P:] * (1.0 - drop),
        ], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            t = tf.random.uniform((batch_size, 1))
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)

            eps         = tf.random.normal(tf.shape(x['input_features']),
                                           dtype=tf.float32) * mask
            perturbed_x = alpha[:, None] * x['input_features'] + eps * sigma[:, None]

            v_pred_part = self.model_part([
                perturbed_x * mask,
                perturbed_x[:, :, :2] * mask,
                x['input_mask'], x['input_jet'], t, y_train])
            v_pred_part = tf.reshape(v_pred_part, (batch_size, -1))
            v_part      = alpha[:, None] * eps - sigma[:, None] * x['input_features']
            v_part      = tf.reshape(v_part, (batch_size, -1))

            sq_per_event   = tf.reduce_sum(tf.square(v_part - v_pred_part), axis=1)
            mask_per_event = tf.reduce_sum(x['input_mask'], axis=1)
            mse_per_event  = sq_per_event / (mask_per_event + 1e-10)
            loss_part      = tf.reduce_sum(w * mse_per_event) / (tf.reduce_sum(w) + 1e-10)

            eps         = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
            perturbed_x = alpha * x['input_jet'] + eps * sigma
            v_pred      = self.model_jet([perturbed_x, t, y_train])
            v_jet       = alpha * eps - sigma * x['input_jet']
            sq_jet      = tf.squeeze(tf.square(v_pred - v_jet), axis=1)
            loss_jet    = tf.reduce_sum(w * sq_jet) / (tf.reduce_sum(w) + 1e-10)

            loss = loss_jet + loss_part

        self.body_optimizer.minimize(loss_part, self.body.trainable_variables, tape=tape)
        trainable_vars = self.model_jet.trainable_variables + self.head.trainable_variables
        self.optimizer.minimize(loss, trainable_vars, tape=tape)

        self.loss_tracker.update_state(loss)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)

        for w_m, ew in zip(self.model_jet.weights, self.ema_jet.weights):
            ew.assign(self.ema * ew + (1 - self.ema) * w_m)
        for w_m, ew in zip(self.head.weights, self.ema_head.weights):
            ew.assign(self.ema * ew + (1 - self.ema) * w_m)
        for w_m, ew in zip(self.body.weights, self.ema_body.weights):
            ew.assign(self.ema * ew + (1 - self.ema) * w_m)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        mask       = x['input_mask'][:, :, None]
        w          = self._w_norm(x)

        t = tf.random.uniform((batch_size, 1))
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)

        eps         = tf.random.normal(tf.shape(x['input_features']),
                                       dtype=tf.float32) * mask
        perturbed_x = alpha[:, None] * x['input_features'] + eps * sigma[:, None]

        v_pred_part = self.model_part([
            perturbed_x * mask,
            perturbed_x[:, :, :2] * mask,
            x['input_mask'], x['input_jet'], t, y])
        v_pred_part = tf.reshape(v_pred_part, (batch_size, -1))
        v_part      = alpha[:, None] * eps - sigma[:, None] * x['input_features']
        v_part      = tf.reshape(v_part, (batch_size, -1))

        sq_per_event   = tf.reduce_sum(tf.square(v_part - v_pred_part), axis=1)
        mask_per_event = tf.reduce_sum(x['input_mask'], axis=1)
        mse_per_event  = sq_per_event / (mask_per_event + 1e-10)
        loss_part      = tf.reduce_sum(w * mse_per_event) / (tf.reduce_sum(w) + 1e-10)

        eps         = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
        perturbed_x = alpha * x['input_jet'] + eps * sigma
        v_pred      = self.model_jet([perturbed_x, t, y])
        v_jet       = alpha * eps - sigma * x['input_jet']
        sq_jet      = tf.squeeze(tf.square(v_pred - v_jet), axis=1)
        loss_jet    = tf.reduce_sum(w * sq_jet) / (tf.reduce_sum(w) + 1e-10)

        loss = loss_jet + loss_part
        self.loss_tracker.update_state(loss)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)
        return {m.name: m.result() for m in self.metrics}


# ── Callbacks ─────────────────────────────────────────────────────────────────

class SaveProgressCallback(keras.callbacks.Callback):
    def __init__(self, state_path, total_epochs):
        super().__init__()
        self.state_path   = state_path
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        epochs_done = epoch + 1
        with open(self.state_path, 'w') as f:
            json.dump({
                'epochs_done':  epochs_done,
                'total_epochs': self.total_epochs,
                'done':         epochs_done >= self.total_epochs,
                'val_loss':     float((logs or {}).get('val_loss', float('inf'))),
            }, f, indent=2)


class TimeLimitCallback(keras.callbacks.Callback):
    def __init__(self, max_seconds):
        super().__init__()
        self.max_seconds = max_seconds
        self._start      = _time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed   = _time.time() - self._start
        remaining = self.max_seconds - elapsed
        if hvd.rank() == 0:
            print(f"  [timer] {elapsed/3600:.2f}h elapsed | {remaining/3600:.2f}h remaining",
                  flush=True)
        if elapsed >= self.max_seconds:
            if hvd.rank() == 0:
                print(f"Time limit {self.max_seconds/3600:.1f}h reached — stopping.",
                      flush=True)
            self.model.stop_training = True


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',         default=_FULL_DATA)
    p.add_argument('--run_name',         default='parton_proc_label')
    p.add_argument('--batch',            type=int,   default=128)
    p.add_argument('--epoch',            type=int,   default=200)
    p.add_argument('--lr',               type=float, default=3e-4)
    p.add_argument('--lr_body',          type=float, default=1e-4)
    p.add_argument('--num_layers',       type=int,   default=8)
    p.add_argument('--num_gen_layers',   type=int,   default=2)
    p.add_argument('--proj_dim',         type=int,   default=128)
    p.add_argument('--num_part',         type=int,   default=500)
    p.add_argument('--max_partons',      type=int,   default=MAX_PARTONS)
    p.add_argument('--local',            action='store_true', default=True)
    p.add_argument('--no_local',         dest='local', action='store_false')
    p.add_argument('--K',                type=int,   default=5)
    p.add_argument('--layer_scale',      action='store_true', default=True)
    p.add_argument('--simple',           action='store_true', default=False)
    p.add_argument('--talking_head',     action='store_true', default=False)
    p.add_argument('--drop_prob',        type=float, default=0.0)
    p.add_argument('--patience',         type=int,   default=30)
    p.add_argument('--val_start',        type=int,   default=400000)
    p.add_argument('--n_train',          type=int,   default=None)
    p.add_argument('--n_val',            type=int,   default=10000)
    p.add_argument('--processes',        nargs='+',  default=['dijet', 'zjets'])
    p.add_argument('--fine_tune',        action='store_true', default=False)
    p.add_argument('--model_name',       default=None)
    p.add_argument('--time_limit_hours', type=float, default=3.5)
    p.add_argument('--no_proc_label',    action='store_true', default=False,
                   help='Remove one-hot process label from conditioning (num_cond=42). '
                        'Process identity is implicit in parton PDG values.')
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_shard(data_dir, stats_path, processes, hvd_rank, hvd_size,
               val_start, n_events_per_proc, num_part, max_partons, split,
               no_proc_label=False):
    with open(stats_path) as f:
        stats = json.load(f)

    part_mean = np.array(stats['part_mean'], dtype=np.float32)
    part_std  = np.array(stats['part_std'],  dtype=np.float32)
    jet_mean  = float(stats['jet_mean'][0])
    jet_std   = float(stats['jet_std'][0])

    cond_mean_raw = np.array(stats['cond_mean'], dtype=np.float32)
    cond_std_raw  = np.array(stats['cond_std'],  dtype=np.float32)
    n_cond_feat   = max_partons * PARTON_FEAT

    cond_mean = np.zeros(n_cond_feat, dtype=np.float32)
    cond_std  = np.ones(n_cond_feat,  dtype=np.float32)
    n_fill    = min(len(cond_mean_raw), n_cond_feat)
    cond_mean[:n_fill] = cond_mean_raw[:n_fill]
    cond_std[:n_fill]  = np.where(cond_std_raw[:n_fill] > 0, cond_std_raw[:n_fill], 1.0)

    n_proc = 0 if no_proc_label else len(processes)
    all_pf, all_mask, all_cond, all_jet, all_w = [], [], [], [], []

    for proc_idx, proc in enumerate(processes):
        path = f'{data_dir}/{proc}.hdf5'
        with h5py.File(path, 'r') as f:
            n_total = f['particle_features'].shape[0]
            if split == 'train':
                start_global, end_global = 0, val_start
            else:
                start_global, end_global = val_start, n_total

            end_global = min(end_global, n_total)
            if n_events_per_proc is not None:
                end_global = min(start_global + n_events_per_proc, end_global)

            n_proc_events = max(end_global - start_global, 0)
            per_rank      = n_proc_events // hvd_size
            r0            = start_global + hvd_rank * per_rank
            r1            = start_global + (hvd_rank + 1) * per_rank

            pf_raw   = f['particle_features'][r0:r1].astype(np.float32)
            part_raw = f['parton_features'][r0:r1].astype(np.float32)
            ew       = f['event_weights'][r0:r1].astype(np.float32)

            if 'n_partons' in f:
                n_par       = f['n_partons'][r0:r1].astype(np.int32)
                parton_mask = (np.arange(max_partons)[None, :] <
                               n_par[:, None]).astype(np.float32)
            else:
                p_norms = np.linalg.norm(part_raw, axis=2)
                valid   = (p_norms > 1e-6).astype(np.float32)
                if valid.shape[1] < max_partons:
                    pad = np.zeros((len(valid), max_partons - valid.shape[1]),
                                   dtype=np.float32)
                    valid = np.concatenate([valid, pad], axis=1)
                parton_mask = valid[:, :max_partons]

        P_file = part_raw.shape[1]
        if P_file < max_partons:
            pad      = np.zeros((len(part_raw), max_partons - P_file, PARTON_FEAT),
                                dtype=np.float32)
            part_raw = np.concatenate([part_raw, pad], axis=1)
        part_raw = part_raw[:, :max_partons, :]

        mask      = pf_raw[:, :num_part, 6].astype(np.float32)
        pf6       = pf_raw[:, :num_part, :6]
        npart     = mask.sum(axis=1, keepdims=True).astype(np.float32)
        log_npart = np.log(np.maximum(npart, 1.0))
        jet       = (log_npart - jet_mean) / jet_std

        cond_raw  = part_raw.reshape(len(part_raw), max_partons * PARTON_FEAT)
        cond_norm = (cond_raw - cond_mean) / cond_std

        # One-hot process label (omitted when --no_proc_label)
        if n_proc > 0:
            proc_label = np.zeros((len(cond_norm), n_proc), dtype=np.float32)
            proc_label[:, proc_idx] = 1.0
            cond = np.concatenate([cond_norm, parton_mask, proc_label], axis=1)
        else:
            cond = np.concatenate([cond_norm, parton_mask], axis=1)

        pf6_norm = (pf6 - part_mean) / part_std * mask[:, :, None]

        pos = ew > 0
        if not pos.all():
            n_neg = int((~pos).sum())
            print(f"  [{proc}] dropped {n_neg}/{len(ew)} negative-weight events "
                  f"({100*n_neg/len(ew):.1f}%)", flush=True)
            pf6_norm    = pf6_norm[pos]
            mask        = mask[pos]
            cond        = cond[pos]
            jet         = jet[pos]
            ew          = ew[pos]

        all_pf.append(pf6_norm)
        all_mask.append(mask)
        all_cond.append(cond)
        all_jet.append(jet)
        all_w.append(ew)
        del pf_raw, part_raw

    pf_all   = np.concatenate(all_pf,   axis=0)
    mask_all = np.concatenate(all_mask, axis=0)
    cond_all = np.concatenate(all_cond, axis=0)
    jet_all  = np.concatenate(all_jet,  axis=0)
    w_all    = np.concatenate(all_w,    axis=0)

    seed = 42 + hvd_rank if split == 'train' else 0
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(pf_all))

    if hvd_rank == 0:
        n_par_vals, counts = np.unique(parton_mask.sum(axis=1), return_counts=True)
        info = ', '.join(f'{int(n)}par:{c}' for n, c in zip(n_par_vals, counts))
        print(f"  parton multiplicity in last proc ({split}): {info}", flush=True)

    return (pf_all[idx], mask_all[idx], cond_all[idx], jet_all[idx], w_all[idx],
            part_mean, part_std, jet_mean, jet_std)


def build_tf_dataset(pf, mask, cond, jet, weights, batch_size, repeat=False):
    tf_x = tf.data.Dataset.from_tensor_slices({
        'input_features': pf,
        'input_points':   pf[:, :, :2],
        'input_mask':     mask,
        'input_jet':      jet,
        'input_weight':   weights,
    })
    tf_y = tf.data.Dataset.from_tensor_slices(cond)
    ds   = (tf.data.Dataset.zip((tf_x, tf_y))
            .cache()
            .shuffle(batch_size * 100)
            .batch(batch_size))
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


def build_lr_schedule(lr, n_train, batch, epochs, resume=False):
    decay_steps = epochs * n_train // batch
    if resume:
        return schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=max(decay_steps, 1))
    warmup_steps = 3 * n_train // batch
    return schedules.CosineDecay(
        initial_learning_rate=lr / 10,
        warmup_target=lr,
        warmup_steps=warmup_steps,
        decay_steps=max(decay_steps, 1))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    flags     = parse_args()
    run_dir    = os.path.join(flags.data_dir, 'checkpoints', flags.run_name)
    ckpt_path  = os.path.join(run_dir, 'pet_pp.weights.h5')
    state_path = os.path.join(run_dir, 'training_state.json')

    n_proc   = 0 if flags.no_proc_label else len(flags.processes)
    num_cond = flags.max_partons * PARTON_FEAT + flags.max_partons + n_proc

    if hvd.rank() == 0:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(flags.data_dir, 'checkpoints', 'histories'), exist_ok=True)

    initial_epoch = 0
    done          = False
    if hvd.rank() == 0 and os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        done          = state.get('done', False)
        initial_epoch = state.get('epochs_done', 0)
        if not done:
            print(f"Resuming from epoch {initial_epoch} "
                  f"(val_loss={state.get('val_loss', float('inf')):.4f})", flush=True)

    done_t  = hvd.broadcast(tf.constant([1 if done else 0], dtype=tf.int32), root_rank=0)
    epoch_t = hvd.broadcast(tf.constant([initial_epoch],    dtype=tf.int32), root_rank=0)
    done          = bool(done_t.numpy()[0])
    initial_epoch = int(epoch_t.numpy()[0])

    if done:
        if hvd.rank() == 0:
            print("Training already complete. Exiting.", flush=True)
        return

    resuming         = initial_epoch > 0 and os.path.exists(ckpt_path)
    per_worker_batch = flags.batch
    global_batch     = per_worker_batch * hvd.size()
    lr_head          = flags.lr      * hvd.size()
    lr_body          = flags.lr_body * hvd.size()

    if hvd.rank() == 0:
        print("Loading training data ...", flush=True)

    stats_path = os.path.join(flags.data_dir, 'normalisation_stats.json')
    tr_pf, tr_mask, tr_cond, tr_jet, tr_w, part_mean, part_std, jet_mean, jet_std = load_shard(
        data_dir=flags.data_dir, stats_path=stats_path, processes=flags.processes,
        hvd_rank=hvd.rank(), hvd_size=hvd.size(), val_start=flags.val_start,
        n_events_per_proc=flags.n_train, num_part=flags.num_part,
        max_partons=flags.max_partons, split='train', no_proc_label=flags.no_proc_label)
    n_local_train = len(tr_pf)

    if hvd.rank() == 0:
        print("Loading validation data ...", flush=True)

    vl_pf, vl_mask, vl_cond, vl_jet, vl_w, _, _, _, _ = load_shard(
        data_dir=flags.data_dir, stats_path=stats_path, processes=flags.processes,
        hvd_rank=hvd.rank(), hvd_size=hvd.size(), val_start=flags.val_start,
        n_events_per_proc=flags.n_val, num_part=flags.num_part,
        max_partons=flags.max_partons, split='val', no_proc_label=flags.no_proc_label)

    steps_per_epoch = n_local_train // per_worker_batch

    if hvd.rank() == 0:
        print(f"processes={flags.processes}  no_proc_label={flags.no_proc_label}", flush=True)
        label_str = "NONE (PDG-implicit)" if flags.no_proc_label else str(n_proc)
        print(f"max_partons={flags.max_partons}  num_cond={num_cond}  "
              f"(P*PF={flags.max_partons*PARTON_FEAT} + mask={flags.max_partons} "
              f"+ proc_label={label_str})", flush=True)
        print(f"Per-GPU batch : {per_worker_batch}  |  Global batch : {global_batch}", flush=True)
        print(f"Workers: {hvd.size()}  |  Local train: {n_local_train:,}  |  "
              f"Steps/epoch: {steps_per_epoch}", flush=True)
        print(f"LR head={lr_head:.2e}  body={lr_body:.2e}  resume={resuming}", flush=True)

    train_ds = build_tf_dataset(tr_pf, tr_mask, tr_cond, tr_jet, tr_w,
                                 per_worker_batch, repeat=True)
    val_ds   = build_tf_dataset(vl_pf, vl_mask, vl_cond, vl_jet, vl_w,
                                 per_worker_batch, repeat=False)
    del tr_pf, tr_mask, vl_pf, vl_mask, tr_w, vl_w
    gc.collect()

    model_cls  = WeightedPartonShowerPET if flags.no_proc_label else WeightedProcLabelPET
    model_kw   = {} if flags.no_proc_label else {'num_proc_labels': n_proc}
    model = model_cls(
        **model_kw,
        num_feat=NUM_FEAT,
        num_jet=NUM_JET,
        max_partons=flags.max_partons,
        parton_feat=PARTON_FEAT,
        num_part=flags.num_part,
        projection_dim=flags.proj_dim,
        local=flags.local,
        K=flags.K,
        num_layers=flags.num_layers,
        num_gen_layers=flags.num_gen_layers,
        drop_probability=flags.drop_prob,
        simple=flags.simple,
        layer_scale=flags.layer_scale,
        talking_head=flags.talking_head,
        mode='generator',
        fine_tune=flags.fine_tune,
        model_name=flags.model_name,
    )

    lr_sched_body = build_lr_schedule(lr_body, n_local_train, per_worker_batch,
                                      flags.epoch, resume=resuming)
    lr_sched_head = build_lr_schedule(lr_head, n_local_train, per_worker_batch,
                                      flags.epoch, resume=resuming)

    optimizer_body = hvd.DistributedOptimizer(Adam(learning_rate=lr_sched_body, clipnorm=1.0))
    optimizer_head = hvd.DistributedOptimizer(Adam(learning_rate=lr_sched_head, clipnorm=1.0))
    model.compile(optimizer_body, optimizer_head)

    if resuming and hvd.rank() == 0:
        model.load_weights(ckpt_path)
        print(f"Loaded checkpoint: {ckpt_path}", flush=True)

    max_seconds = int(flags.time_limit_hours * 3600)
    callbacks   = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        TimeLimitCallback(max_seconds=max_seconds),
    ]

    if hvd.rank() == 0:
        callbacks += [
            ModelCheckpoint(ckpt_path, save_best_only=True, save_weights_only=True,
                            monitor='val_loss'),
            EarlyStopping(patience=flags.patience, restore_best_weights=True,
                          monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', patience=flags.patience // 2,
                              factor=0.5, min_lr=1e-6),
            SaveProgressCallback(state_path, total_epochs=flags.epoch),
        ]

    remaining = flags.epoch - initial_epoch
    if hvd.rank() == 0:
        print(f"Training: {n_local_train:,} local events | "
              f"epochs {initial_epoch}→{flags.epoch} ({remaining} remaining) | "
              f"run={flags.run_name} | time limit={flags.time_limit_hours:.1f}h", flush=True)

    hist = model.fit(
        train_ds,
        initial_epoch=initial_epoch,
        epochs=flags.epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        verbose=1 if hvd.rank() == 0 else 0,
    )

    if hvd.rank() == 0:
        hist_path = os.path.join(flags.data_dir, 'checkpoints', 'histories',
                                  f'{flags.run_name}.pkl')
        if os.path.exists(hist_path):
            with open(hist_path, 'rb') as f:
                prev = pickle.load(f)
            for k, v in hist.history.items():
                prev.setdefault(k, []).extend(v)
            combined = prev
        else:
            combined = hist.history
        with open(hist_path, 'wb') as f:
            pickle.dump(combined, f)
        print(f"History → {hist_path}", flush=True)


if __name__ == '__main__':
    main()
