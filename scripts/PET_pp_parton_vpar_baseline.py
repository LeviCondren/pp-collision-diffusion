"""PET_pp_parton with variable-length parton conditioning and process label.

Extends PET_pp_parton to support a variable number of hard-scatter partons per
event (e.g. 4 for 2→2 LO-like events, 5 for 2→3 NLO real-emission events) and
an optional one-hot process label appended to the conditioning vector.

Conditioning vector format (shape N × num_cond):
  [0 : max_partons*parton_feat]                normalised parton kinematics
  [max_partons*parton_feat : …+max_partons]    binary parton mask (1=valid)
  […+max_partons : end]                        one-hot process label (n_proc dims)

  num_cond = max_partons * parton_feat + max_partons + num_proc_labels

When num_proc_labels=0 the layout reduces to the original kinematics + mask form.

The process label is projected to a single token and appended to the parton
cross-attention key/value set so the generator head attends to both parton
kinematics and process identity simultaneously.  The same token enters the
ResNet jet head via a mean-pool over parton+process tokens.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
from tensorflow.keras.models import Model
from PET import PET, FourierProjection, get_encoding
from layers import LayerScale, StochasticDepth
from tqdm import tqdm


class PET_pp_parton_vpar(keras.Model):
    def __init__(self,
                 num_feat,
                 num_jet,
                 max_partons=5,
                 parton_feat=6,
                 num_proc_labels=5,
                 num_part=500,
                 projection_dim=128,
                 local=True, K=5,
                 num_local=2,
                 num_layers=8,
                 num_gen_layers=2,
                 num_heads=4,
                 drop_probability=0.0,
                 simple=False,
                 layer_scale=True,
                 layer_scale_init=1e-5,
                 talking_head=False,
                 feature_drop=0.1,
                 mode='generator',
                 fine_tune=False,
                 model_name=None):
        super().__init__()

        self.num_feat         = num_feat
        self.num_jet          = num_jet
        self.max_partons      = max_partons
        self.parton_feat      = parton_feat
        self.num_proc_labels  = num_proc_labels
        self.num_cond         = (max_partons * parton_feat
                                 + max_partons + num_proc_labels)
        self.max_part         = num_part
        self.projection_dim   = projection_dim
        self.num_heads        = num_heads
        self.num_gen_layers   = num_gen_layers
        self.layer_scale      = layer_scale
        self.layer_scale_init = layer_scale_init
        self.feature_drop     = feature_drop
        self.num_steps        = 500
        self.ema              = 0.999
        self.shape            = (-1, 1, 1)

        # ── Body encoder (unchanged) ──────────────────────────────────────────
        _scaffold = PET(num_feat=num_feat,
                        num_jet=num_jet,
                        num_classes=self.num_cond,
                        projection_dim=projection_dim,
                        local=local, K=K,
                        num_local=num_local,
                        num_layers=num_layers,
                        drop_probability=drop_probability,
                        simple=simple,
                        layer_scale=layer_scale,
                        talking_head=talking_head,
                        mode=mode)
        self.body = _scaffold.ema_body

        if fine_tune:
            assert model_name is not None
            self.body.load_weights(model_name, by_name=True, skip_mismatch=True)

        # ── Variable-parton generator head ────────────────────────────────────
        self.head = self._build_vpar_generator_head()

        # ── Wire body + head into model_part ──────────────────────────────────
        inputs_time     = Input((1,))
        inputs_cond     = Input((self.num_cond,))
        inputs_jet      = Input((self.num_jet,))
        inputs_mask     = Input((None, 1))
        inputs_features = Input(shape=(None, num_feat))
        inputs_points   = Input(shape=(None, 2))

        output_body  = self.body([inputs_features, inputs_points, inputs_mask, inputs_time])
        outputs_head = self.head([output_body, inputs_jet, inputs_mask, inputs_time, inputs_cond])
        outputs      = inputs_mask * outputs_head

        self.model_part = keras.Model(
            inputs=[inputs_features, inputs_points, inputs_mask, inputs_jet,
                    inputs_time, inputs_cond],
            outputs=outputs)

        # ── Stage-1 ResNet: log_npart from parton conditioning ─────────────────
        outputs_jet = self._resnet_vpar(inputs_jet, inputs_time, inputs_cond,
                                        num_layer=3, mlp_dim=2 * projection_dim)
        self.model_jet = Model(inputs=[inputs_jet, inputs_time, inputs_cond],
                               outputs=outputs_jet)

        # ── EMA shadow models ─────────────────────────────────────────────────
        self.ema_jet  = keras.models.clone_model(self.model_jet)
        self.ema_body = keras.models.clone_model(self.body)
        self.ema_head = keras.models.clone_model(self.head)

        self.loss_tracker      = keras.metrics.Mean(name="loss")
        self.loss_part_tracker = keras.metrics.Mean(name="part")
        self.loss_jet_tracker  = keras.metrics.Mean(name="jet")

    # ── Helpers to split conditioning into features + mask ────────────────────

    def _split_cond(self, inp_cond):
        """Return (parton_feat_flat, parton_mask, proc_label) from the cond vector."""
        n_feat = self.max_partons * self.parton_feat
        n_mask = self.max_partons
        return (inp_cond[:, :n_feat],
                inp_cond[:, n_feat : n_feat + n_mask],
                inp_cond[:, n_feat + n_mask :])

    # ── Variable-parton generator head ────────────────────────────────────────

    def _build_vpar_generator_head(self):
        """Generator head with masked per-parton cross-attention.

        The parton mask gates which parton tokens are valid key/value positions.
        When num_proc_labels > 0 the one-hot process label is projected to a
        single additional token appended to cond_set, always attending (mask=1).
        """
        D   = self.projection_dim
        nh  = self.num_heads
        kd  = D // nh
        P   = self.max_partons
        PF  = self.parton_feat

        inp_encoded = Input(shape=(None, D),          name='vph_encoded')
        inp_jet     = Input(shape=(1,),               name='vph_jet')
        inp_mask    = Input(shape=(None, 1),           name='vph_mask')
        inp_time    = Input(shape=(1,),                name='vph_time')
        inp_cond    = Input(shape=(self.num_cond,),    name='vph_cond')

        # Split: parton features | parton mask | (optional) process label
        parton_feat_flat = inp_cond[:, :P * PF]               # (N, P*PF)
        parton_mask_in   = inp_cond[:, P * PF : P * PF + P]   # (N, P)

        # Per-parton token embedding
        parton_tokens = layers.Reshape((P, PF))(parton_feat_flat)   # (N, P, PF)
        parton_emb    = layers.Dense(D)(parton_tokens)               # (N, P, D)
        parton_emb    = StochasticDepth(self.feature_drop)(parton_emb)

        if self.num_proc_labels > 0:
            proc_label_in = inp_cond[:, P * PF + P :]               # (N, n_proc)
            proc_token    = layers.Dense(D, activation='gelu')(proc_label_in)
            proc_token    = layers.Dense(D)(proc_token)
            proc_token    = tf.expand_dims(proc_token, axis=1)       # (N, 1, D)
            cond_set      = tf.concat([parton_emb, proc_token], axis=1)  # (N, P+1, D)
            ones_col      = tf.ones_like(parton_mask_in[:, :1])      # (N, 1)
            attn_mask     = tf.cast(
                tf.concat([parton_mask_in, ones_col], axis=1)[:, None, :],
                tf.bool)                                              # (N, 1, P+1)
        else:
            cond_set  = parton_emb                                   # (N, P, D)
            attn_mask = tf.cast(parton_mask_in[:, None, :], tf.bool) # (N, 1, P)

        # Time + log_npart global conditioning token
        time_emb   = FourierProjection(inp_time, D)
        jet_emb    = layers.Dense(D)(inp_jet)
        cond_token = layers.Dense(2 * D, activation="gelu")(time_emb + jet_emb)
        cond_token = layers.Dense(D, activation="gelu")(cond_token)  # (N, D)
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

            # 2. Masked cross-attention to parton (+process) tokens
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

    # ── Stage-1 ResNet with masked parton pooling ─────────────────────────────

    def _resnet_vpar(self, inputs, inputs_time, labels,
                     num_layer=3, mlp_dim=128, dropout=0.0):
        """ResNet for log_npart.  Masked mean-pool over valid parton tokens."""

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

        parton_feat_flat = labels[:, :P * PF]               # (N, P*PF)
        parton_mask_in   = labels[:, P * PF : P * PF + P]   # (N, P)

        parton_tokens = tf.reshape(parton_feat_flat, (-1, P, PF))
        parton_emb    = layers.Dense(D)(parton_tokens)       # (N, P, D)

        if self.num_proc_labels > 0:
            proc_label_in = labels[:, P * PF + P :]         # (N, n_proc)
            proc_token    = layers.Dense(D, activation='gelu')(proc_label_in)
            proc_token    = layers.Dense(D)(proc_token)
            proc_token    = tf.expand_dims(proc_token, axis=1)   # (N, 1, D)
            # Treat process token as an always-valid additional parton slot
            tokens        = tf.concat([parton_emb, proc_token], axis=1)  # (N, P+1, D)
            ones_col      = tf.ones_like(parton_mask_in[:, :1])
            full_mask     = tf.concat([parton_mask_in, ones_col], axis=1) # (N, P+1)
        else:
            tokens    = parton_emb                           # (N, P, D)
            full_mask = parton_mask_in                       # (N, P)

        # Masked mean-pool over parton (+process) tokens
        mask_expand   = full_mask[:, :, None]                # (N, P(+1), 1)
        count         = tf.maximum(
            tf.reduce_sum(full_mask, axis=1, keepdims=True), 1.0)   # (N, 1)
        parton_global = (tf.reduce_sum(tokens * mask_expand, axis=1)
                         / count)                            # (N, D)

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

    # ── Training / evaluation (identical to PET_pp_parton) ───────────────────

    @property
    def metrics(self):
        return [self.loss_tracker, self.loss_part_tracker, self.loss_jet_tracker]

    def compile(self, body_optimizer, head_optimizer):
        super().compile(experimental_run_tf_function=False, weighted_metrics=[])
        self.body_optimizer = body_optimizer
        self.optimizer      = head_optimizer

    def prior_sde(self, dimensions):
        return tf.random.normal(dimensions, dtype=tf.float32)

    def train_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        mask       = x['input_mask'][:, :, None]

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
            v_pred_part = tf.reshape(v_pred_part, (tf.shape(v_pred_part)[0], -1))
            v_part      = alpha[:, None] * eps - sigma[:, None] * x['input_features']
            v_part      = tf.reshape(v_part, (tf.shape(v_part)[0], -1))
            loss_part   = (tf.reduce_sum(tf.square(v_part - v_pred_part))
                           / tf.reduce_sum(x['input_mask']))

            eps         = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
            perturbed_x = alpha * x['input_jet'] + eps * sigma
            v_pred      = self.model_jet([perturbed_x, t, y])
            v_jet       = alpha * eps - sigma * x['input_jet']
            loss_jet    = tf.reduce_mean(tf.square(v_pred - v_jet))

            loss = loss_jet + loss_part

        self.body_optimizer.minimize(loss_part, self.body.trainable_variables, tape=tape)
        trainable_vars = self.model_jet.trainable_variables + self.head.trainable_variables
        self.optimizer.minimize(loss, trainable_vars, tape=tape)

        self.loss_tracker.update_state(loss)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)

        for w, ew in zip(self.model_jet.weights, self.ema_jet.weights):
            ew.assign(self.ema * ew + (1 - self.ema) * w)
        for w, ew in zip(self.head.weights, self.ema_head.weights):
            ew.assign(self.ema * ew + (1 - self.ema) * w)
        for w, ew in zip(self.body.weights, self.ema_body.weights):
            ew.assign(self.ema * ew + (1 - self.ema) * w)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        mask       = x['input_mask'][:, :, None]

        t = tf.random.uniform((batch_size, 1))
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)

        eps         = tf.random.normal(tf.shape(x['input_features']),
                                       dtype=tf.float32) * mask
        perturbed_x = alpha[:, None] * x['input_features'] + eps * sigma[:, None]

        v_pred_part = self.model_part([
            perturbed_x * mask,
            perturbed_x[:, :, :2] * mask,
            x['input_mask'], x['input_jet'], t, y])
        v_pred_part = tf.reshape(v_pred_part, (tf.shape(v_pred_part)[0], -1))
        v_part      = alpha[:, None] * eps - sigma[:, None] * x['input_features']
        v_part      = tf.reshape(v_part, (tf.shape(v_part)[0], -1))
        loss_part   = (tf.reduce_sum(tf.square(v_part - v_pred_part))
                       / tf.reduce_sum(x['input_mask']))

        eps         = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
        perturbed_x = alpha * x['input_jet'] + eps * sigma
        v_pred      = self.model_jet([perturbed_x, t, y])
        v_jet       = alpha * eps - sigma * x['input_jet']
        loss_jet    = tf.reduce_mean(tf.square(v_pred - v_jet))

        loss = loss_jet + loss_part
        self.loss_tracker.update_state(loss)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        return self.model_part(x)

    def generate(self, cond, jet_mean, jet_std, nsplit=2, jets=None, use_tqdm=False,
                 num_steps=None):
        """Generate particle clouds.  cond has shape (N, max_partons*parton_feat+max_partons)."""
        part_steps = num_steps if num_steps is not None else self.num_steps
        jet_steps  = num_steps if num_steps is not None else 512

        jet_info  = []
        part_info = []
        jet_split = np.array_split(jets, nsplit) if jets is not None else None
        splits    = np.array_split(cond, nsplit)

        for i, split in (tqdm(enumerate(splits), total=len(splits))
                         if use_tqdm else enumerate(splits)):
            if jets is not None:
                jet = jet_split[i]
            else:
                jet = self.DDPMSampler(split, self.ema_jet,
                                       data_shape=[split.shape[0], self.num_jet],
                                       w=0.0, num_steps=jet_steps,
                                       const_shape=[-1, 1]).numpy()

            jet_info.append(jet)

            log_npart = jet[:, 0] * jet_std + jet_mean
            nparts    = np.expand_dims(
                np.clip(np.round(np.exp(log_npart)).astype(int), 1, self.max_part), -1)
            mask      = np.expand_dims(
                np.tile(np.arange(self.max_part), (nparts.shape[0], 1))
                < np.tile(nparts, (1, self.max_part)), -1)

            parts = self.DDPMSampler(
                split, [self.ema_body, self.ema_head],
                data_shape=[split.shape[0], self.max_part, self.num_feat],
                jet=jet, num_steps=part_steps,
                const_shape=self.shape, w=0.0,
                mask=mask.astype(np.float32)).numpy()
            part_info.append(parts * mask)

        return np.concatenate(part_info), np.concatenate(jet_info)

    # ── Diffusion schedule helpers ────────────────────────────────────────────

    def logsnr_schedule_cosine(self, t, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t, tf.float32) + b))

    def get_logsnr_alpha_sigma(self, time, shape=None):
        logsnr = self.logsnr_schedule_cosine(time)
        alpha  = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma  = tf.sqrt(tf.math.sigmoid(-logsnr))
        if shape is not None:
            alpha  = tf.reshape(alpha,  shape)
            sigma  = tf.reshape(sigma,  shape)
            logsnr = tf.reshape(logsnr, shape)
        return logsnr, tf.cast(alpha, tf.float32), tf.cast(sigma, tf.float32)

    def evaluate_models(self, head, body, x, jet, mask, t, cond, w=0.0):
        x_in   = mask * x
        v_body = body([x_in, x[:, :, :2], mask, t], training=False)
        v      = mask * head([v_body, jet, mask, t, cond], training=False)
        return v

    @tf.function
    def second_order_correction(self, time_step, x, pred_images, pred_noises,
                                alphas, sigmas, w, cond, model,
                                jet=None, mask=None, num_steps=100,
                                second_order_alpha=0.5, shape=None):
        step_size   = 1.0 / num_steps
        t           = time_step - second_order_alpha * step_size
        logsnr, alpha_s, alpha_n = self.get_logsnr_alpha_sigma(t, shape=shape)
        alpha_noisy = alpha_s * pred_images + alpha_n * pred_noises

        if jet is None:
            v = model([alpha_noisy, t, cond], training=False)
        else:
            alpha_noisy  *= mask
            model_body, model_head = model
            v = self.evaluate_models(model_head, model_body,
                                     alpha_noisy, jet, mask, t, cond, w)

        alpha_pred_noises = alpha_n * alpha_noisy + alpha_s * v
        pred_noises = ((1.0 - 1.0 / (2.0 * second_order_alpha)) * pred_noises
                       + 1.0 / (2.0 * second_order_alpha) * alpha_pred_noises)

        mean = (x - sigmas * pred_noises) / alphas
        return mean, pred_noises

    @tf.function
    def DDPMSampler(self, cond, model, data_shape=None, const_shape=None,
                    jet=None, w=0.1, num_steps=100, mask=None):
        batch_size = cond.shape[0]
        x          = self.prior_sde(data_shape)

        for time_step in tf.range(num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / num_steps
            logsnr,  alpha,  sigma  = self.get_logsnr_alpha_sigma(t, shape=const_shape)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(
                tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / num_steps,
                shape=const_shape)

            if jet is None:
                v = model([x, t, cond], training=False)
            else:
                x *= mask
                model_body, model_head = model
                v = self.evaluate_models(model_head, model_body,
                                         x, jet, mask, t, cond, w)

            mean = alpha * x - sigma * v
            eps  = v * alpha + x * sigma
            mean, eps = self.second_order_correction(
                t, x, mean, eps, alpha, sigma, w, cond, model,
                jet, mask, num_steps=num_steps, shape=const_shape)
            x = alpha_ * mean + sigma_ * eps

        return mean
