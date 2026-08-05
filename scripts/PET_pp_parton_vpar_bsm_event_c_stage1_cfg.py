"""PET_pp_parton_vpar_bsm_event_c_stage1_cfg — E034: CFG training and inference for cone mass.

Copied from PET_pp_parton_vpar_bsm_event_c_stage1.py (E023) and modified for E034:
  - Architecture identical to E023 (num_gen_layers=2, standard cross-attention).
  - Adds DDPMSamplerCFG / second_order_correction_cfg / generate_cfg for two-pass CFG.
  - WeightedBSMPET_event_c_cfg subclass applies conditioning dropout (prob 0.15 by default)
    on cone mass features (indices 4 and 6 of the 7-dim event vector) during training.
  - Null conditioning = normalized zero (0.0) for the dropped features.
  - Stage-1 ResNet, EMA, and diffusion schedule unchanged from E023.

Do NOT modify the original PET_pp_parton_vpar_bsm_event_c_stage1.py (E023 canonical).
Do NOT mix checkpoints from this file with E023 checkpoints.

Changelog:
  E034 (2026-08-01): CFG training and inference for cone mass conditioning.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from PET import PET, FourierProjection
from layers import LayerScale, StochasticDepth
from tqdm import tqdm

_NUM_EVENT_FEAT_DEFAULT = 7  # all 7: MET(3) + cone_X(2) + cone_Y(2)

# Indices in the 7-dim event vector (inp_event) corresponding to cone mass.
# [log1p_MET(0), sin_MET(1), cos_MET(2), log1p_cone_pT_X(3), log1p_cone_mass_X(4),
#  log1p_cone_pT_Y(5), log1p_cone_mass_Y(6)]
_CONE_MASS_INDICES = [4, 6]
_EVENT_KEEP_MASK = [1., 1., 1., 1., 0., 1., 0.]  # zeros at indices 4, 6


class PET_pp_parton_vpar_bsm_event_c_stage1(keras.Model):
    def __init__(self,
                 num_feat,
                 num_jet,
                 max_partons=4,
                 parton_feat=7,
                 num_event_feat=_NUM_EVENT_FEAT_DEFAULT,
                 num_part=500,
                 projection_dim=128,
                 num_jet_mlp=512,
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
        self.num_jet_mlp      = num_jet_mlp
        self.max_partons      = max_partons
        self.parton_feat      = parton_feat
        self.num_event_feat   = num_event_feat
        self.num_cond         = max_partons * parton_feat + max_partons  # 32
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

        # ── Body encoder ──────────────────────────────────────────────────────
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

        # ── Variable-parton generator head with event token ───────────────────
        self.head = self._build_vpar_generator_head()

        # ── Wire body + head into model_part ──────────────────────────────────
        inputs_time     = Input((1,))
        inputs_cond     = Input((self.num_cond,))
        inputs_jet_s2   = Input((1,),              name='inp_jet_s2')
        inputs_jet_s1   = Input((self.num_jet,),   name='inp_jet_s1')
        inputs_mask     = Input((None, 1))
        inputs_features = Input(shape=(None, num_feat))
        inputs_points   = Input(shape=(None, 2))
        inputs_event    = Input((self.num_event_feat,))

        output_body  = self.body([inputs_features, inputs_points, inputs_mask, inputs_time])
        outputs_head = self.head([output_body, inputs_jet_s2, inputs_mask, inputs_time,
                                  inputs_cond, inputs_event])
        outputs      = inputs_mask * outputs_head

        self.model_part = keras.Model(
            inputs=[inputs_features, inputs_points, inputs_mask, inputs_jet_s2,
                    inputs_time, inputs_cond, inputs_event],
            outputs=outputs)

        # ── Stage-1 ResNet: 8-dim [log_npart + event features] ────────────────
        outputs_jet = self._resnet_vpar(inputs_jet_s1, inputs_time, inputs_cond,
                                        num_layer=3, mlp_dim=num_jet_mlp)
        self.model_jet = Model(inputs=[inputs_jet_s1, inputs_time, inputs_cond],
                               outputs=outputs_jet)

        # ── EMA shadow models ─────────────────────────────────────────────────
        self.ema_jet  = keras.models.clone_model(self.model_jet)
        self.ema_body = keras.models.clone_model(self.body)
        self.ema_head = keras.models.clone_model(self.head)

        self.loss_tracker      = keras.metrics.Mean(name="loss")
        self.loss_part_tracker = keras.metrics.Mean(name="part")
        self.loss_jet_tracker  = keras.metrics.Mean(name="jet")

    # ── Generator head: parton cross-attention + event token ──────────────────

    def _build_vpar_generator_head(self):
        D   = self.projection_dim
        nh  = self.num_heads
        kd  = D // nh
        P   = self.max_partons
        PF  = self.parton_feat
        NEF = self.num_event_feat

        inp_encoded = Input(shape=(None, D),         name='vph_encoded')
        inp_jet     = Input(shape=(1,),              name='vph_jet')
        inp_mask    = Input(shape=(None, 1),         name='vph_mask')
        inp_time    = Input(shape=(1,),              name='vph_time')
        inp_cond    = Input(shape=(self.num_cond,),  name='vph_cond')
        inp_event   = Input(shape=(NEF,),            name='vph_event')

        parton_feat_flat = inp_cond[:, :P * PF]
        parton_mask_in   = inp_cond[:, P * PF : P * PF + P]

        parton_tokens = layers.Reshape((P, PF))(parton_feat_flat)
        parton_emb    = layers.Dense(D)(parton_tokens)
        parton_emb    = StochasticDepth(self.feature_drop)(parton_emb)

        event_token = layers.Dense(D, activation='gelu',
                                   name='event_token_dense1')(inp_event)
        event_token = layers.Dense(D, name='event_token_dense2')(event_token)
        event_token = tf.expand_dims(event_token, axis=1)

        cond_set  = tf.concat([parton_emb, event_token], axis=1)

        ones_col  = tf.ones_like(parton_mask_in[:, :1])
        full_mask = tf.concat([parton_mask_in, ones_col], axis=1)
        attn_mask = tf.cast(full_mask[:, None, :], tf.bool)

        time_emb   = FourierProjection(inp_time, D)
        jet_emb    = layers.Dense(D)(inp_jet)
        cond_token = layers.Dense(2 * D, activation="gelu")(time_emb + jet_emb)
        cond_token = layers.Dense(D, activation="gelu")(cond_token)
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
            x3 = layers.Dense(2 * D, activation="gelu")(x3)
            x3 = layers.Dense(D)(x3)
            if self.layer_scale:
                x3 = LayerScale(self.layer_scale_init, D)(x3, inp_mask)
            cond_token = layers.Add()([x3, x2])

        out = layers.GroupNormalization(groups=1)(cond_token + encoded)
        out = layers.Dense(self.num_feat)(out) * inp_mask

        return keras.Model(
            inputs=[inp_encoded, inp_jet, inp_mask, inp_time, inp_cond, inp_event],
            outputs=out,
            name='vpar_generator_head_event_a')

    # ── Stage-1 ResNet ────────────────────────────────────────────────────────

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

        parton_feat_flat = labels[:, :P * PF]
        parton_mask_in   = labels[:, P * PF : P * PF + P]

        parton_tokens = tf.reshape(parton_feat_flat, (-1, P, PF))
        parton_emb    = layers.Dense(D)(parton_tokens)

        mask_expand   = parton_mask_in[:, :, None]
        count         = tf.maximum(
            tf.reduce_sum(parton_mask_in, axis=1, keepdims=True), 1.0)
        parton_global = (tf.reduce_sum(parton_emb * mask_expand, axis=1) / count)

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

    # ── Standard (unweighted) train/test steps ────────────────────────────────

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
                x['input_mask'], x['input_jet'][:, 0:1], t, y,
                x['input_event']])
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
            x['input_mask'], x['input_jet'][:, 0:1], t, y,
            x['input_event']])
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

    def generate(self, cond, jet_mean, jet_std, event_feat,
                 nsplit=2, jets=None, use_tqdm=False, num_steps=None,
                 num_jet_steps=None, use_true_event=False):
        """Standard (non-CFG) generation — identical to E023."""
        part_steps      = num_steps if num_steps is not None else self.num_steps
        jet_steps_count = num_jet_steps if num_jet_steps is not None else 512

        jet_info  = []
        part_info = []
        jet_split = np.array_split(jets, nsplit) if jets is not None else None
        splits    = np.array_split(cond, nsplit)
        ev_splits = np.array_split(event_feat, nsplit)

        for i, split in (tqdm(enumerate(splits), total=len(splits))
                         if use_tqdm else enumerate(splits)):
            if jets is not None:
                jet = jet_split[i]
            else:
                jet = self.DDPMSampler(split, self.ema_jet,
                                       data_shape=[split.shape[0], self.num_jet],
                                       w=0.0, num_steps=jet_steps_count,
                                       const_shape=[-1, 1]).numpy()
            jet_info.append(jet)

            log_npart_norm = jet[:, 0:1]
            log_npart      = log_npart_norm[:, 0] * jet_std + jet_mean
            nparts    = np.expand_dims(
                np.clip(np.round(np.exp(log_npart)).astype(int), 1, self.max_part), -1)
            mask      = np.expand_dims(
                np.tile(np.arange(self.max_part), (nparts.shape[0], 1))
                < np.tile(nparts, (1, self.max_part)), -1)

            if use_true_event:
                ev_i = ev_splits[i]
            else:
                ev_i = jet[:, 1:]

            parts = self.DDPMSampler(
                split, [self.ema_body, self.ema_head],
                data_shape=[split.shape[0], self.max_part, self.num_feat],
                jet=log_npart_norm, num_steps=part_steps,
                const_shape=self.shape, w=0.0,
                mask=mask.astype(np.float32),
                event_feat=ev_i).numpy()
            part_info.append(parts * mask)

        return np.concatenate(part_info), np.concatenate(jet_info)

    def generate_cfg(self, cond, jet_mean, jet_std,
                     nsplit=2, jets=None, use_tqdm=False, num_steps=None,
                     num_jet_steps=None, guidance_scale=1.0,
                     use_true_cone=False, event_feat=None):
        """Two-pass CFG generation.

        Runs the denoising model twice per step:
          - Conditional pass: cone mass features (indices 4, 6 of 7-dim event) from stage 1
          - Null pass: those indices zeroed (normalized zero = training mean)
          - Combined: v_guided = v_null + guidance_scale * (v_cond - v_null)

        At guidance_scale=1.0: output equals conditional pass (standard sampling).
        At guidance_scale=0.0: output equals null pass (unconditional).
        At guidance_scale>1.0: extrapolation beyond conditional (the CFG effect).
        """
        part_steps      = num_steps if num_steps is not None else self.num_steps
        jet_steps_count = num_jet_steps if num_jet_steps is not None else 512

        jet_info  = []
        part_info = []
        jet_split = np.array_split(jets, nsplit) if jets is not None else None
        splits    = np.array_split(cond, nsplit)
        ev_splits = np.array_split(event_feat, nsplit) if event_feat is not None else None

        gs = tf.constant(guidance_scale, dtype=tf.float32)

        for i, split in (tqdm(enumerate(splits), total=len(splits))
                         if use_tqdm else enumerate(splits)):
            if jets is not None:
                jet = jet_split[i]
            else:
                jet = self.DDPMSampler(split, self.ema_jet,
                                       data_shape=[split.shape[0], self.num_jet],
                                       w=0.0, num_steps=jet_steps_count,
                                       const_shape=[-1, 1]).numpy()
            jet_info.append(jet)

            log_npart_norm = jet[:, 0:1]
            log_npart      = log_npart_norm[:, 0] * jet_std + jet_mean
            nparts = np.expand_dims(
                np.clip(np.round(np.exp(log_npart)).astype(int), 1, self.max_part), -1)
            mask_arr = np.expand_dims(
                np.tile(np.arange(self.max_part), (nparts.shape[0], 1))
                < np.tile(nparts, (1, self.max_part)), -1)

            # Conditional event features: truth cone masses or stage-1 predictions
            if use_true_cone and ev_splits is not None:
                ev_cond = ev_splits[i]
            else:
                ev_cond = jet[:, 1:]  # stage-1 generated event features (N, 7)

            # Null event: zero out cone mass indices (4 and 6 in 7-dim event vector)
            ev_null = ev_cond.copy()
            ev_null[:, 4] = 0.0  # log1p_cone_mass_X → null (normalized mean)
            ev_null[:, 6] = 0.0  # log1p_cone_mass_Y → null

            parts = self.DDPMSamplerCFG(
                split, [self.ema_body, self.ema_head],
                data_shape=[split.shape[0], self.max_part, self.num_feat],
                jet=log_npart_norm, num_steps=part_steps,
                const_shape=self.shape, w=0.0,
                mask=mask_arr.astype(np.float32),
                event_feat=ev_cond,
                null_event_feat=ev_null,
                guidance_scale=gs).numpy()
            part_info.append(parts * mask_arr)

        return np.concatenate(part_info), np.concatenate(jet_info)

    # ── Diffusion schedule ────────────────────────────────────────────────────

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

    def evaluate_models(self, head, body, x, jet, mask, t, cond, w=0.0,
                        event_feat=None):
        x_in   = mask * x
        v_body = body([x_in, x[:, :, :2], mask, t], training=False)
        v      = mask * head([v_body, jet, mask, t, cond, event_feat],
                              training=False)
        return v

    @tf.function
    def second_order_correction(self, time_step, x, pred_images, pred_noises,
                                alphas, sigmas, w, cond, model,
                                jet=None, mask=None, num_steps=100,
                                second_order_alpha=0.5, shape=None,
                                event_feat=None):
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
                                     alpha_noisy, jet, mask, t, cond, w,
                                     event_feat=event_feat)

        alpha_pred_noises = alpha_n * alpha_noisy + alpha_s * v
        pred_noises = ((1.0 - 1.0 / (2.0 * second_order_alpha)) * pred_noises
                       + 1.0 / (2.0 * second_order_alpha) * alpha_pred_noises)

        mean = (x - sigmas * pred_noises) / alphas
        return mean, pred_noises

    @tf.function
    def second_order_correction_cfg(self, time_step, x, pred_images, pred_noises,
                                     alphas, sigmas, w, cond, model,
                                     jet=None, mask=None, num_steps=100,
                                     second_order_alpha=0.5, shape=None,
                                     event_feat=None, null_event_feat=None,
                                     guidance_scale=1.0):
        """CFG version of second_order_correction: two model evals, combined with guidance."""
        step_size   = 1.0 / num_steps
        t           = time_step - second_order_alpha * step_size
        logsnr, alpha_s, alpha_n = self.get_logsnr_alpha_sigma(t, shape=shape)
        alpha_noisy = alpha_s * pred_images + alpha_n * pred_noises

        alpha_noisy *= mask
        model_body, model_head = model

        v_cond = self.evaluate_models(model_head, model_body,
                                      alpha_noisy, jet, mask, t, cond, w,
                                      event_feat=event_feat)
        v_null = self.evaluate_models(model_head, model_body,
                                      alpha_noisy, jet, mask, t, cond, w,
                                      event_feat=null_event_feat)
        v = v_null + guidance_scale * (v_cond - v_null)

        alpha_pred_noises = alpha_n * alpha_noisy + alpha_s * v
        pred_noises = ((1.0 - 1.0 / (2.0 * second_order_alpha)) * pred_noises
                       + 1.0 / (2.0 * second_order_alpha) * alpha_pred_noises)

        mean = (x - sigmas * pred_noises) / alphas
        return mean, pred_noises

    @tf.function
    def DDPMSampler(self, cond, model, data_shape=None, const_shape=None,
                    jet=None, w=0.1, num_steps=100, mask=None, event_feat=None):
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
                                         x, jet, mask, t, cond, w,
                                         event_feat=event_feat)

            mean = alpha * x - sigma * v
            eps  = v * alpha + x * sigma
            mean, eps = self.second_order_correction(
                t, x, mean, eps, alpha, sigma, w, cond, model,
                jet, mask, num_steps=num_steps, shape=const_shape,
                event_feat=event_feat)
            x = alpha_ * mean + sigma_ * eps

        return mean

    @tf.function
    def DDPMSamplerCFG(self, cond, model, data_shape=None, const_shape=None,
                       jet=None, w=0.1, num_steps=100, mask=None,
                       event_feat=None, null_event_feat=None, guidance_scale=1.0):
        """Two-pass CFG DDPM sampler for particle generation.

        Runs two model evaluations per denoising step:
          v_cond: conditional (with cone mass event features)
          v_null: unconditional (cone mass indices zeroed)
          v_guided = v_null + guidance_scale * (v_cond - v_null)

        At guidance_scale=1.0 this is equivalent to standard DDPMSampler.
        """
        batch_size = cond.shape[0]
        x          = self.prior_sde(data_shape)

        for time_step in tf.range(num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / num_steps
            logsnr,  alpha,  sigma  = self.get_logsnr_alpha_sigma(t, shape=const_shape)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(
                tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / num_steps,
                shape=const_shape)

            x *= mask
            model_body, model_head = model

            v_cond = self.evaluate_models(model_head, model_body,
                                          x, jet, mask, t, cond, w,
                                          event_feat=event_feat)
            v_null = self.evaluate_models(model_head, model_body,
                                          x, jet, mask, t, cond, w,
                                          event_feat=null_event_feat)
            v = v_null + guidance_scale * (v_cond - v_null)

            mean = alpha * x - sigma * v
            eps  = v * alpha + x * sigma
            mean, eps = self.second_order_correction_cfg(
                t, x, mean, eps, alpha, sigma, w, cond, model,
                jet, mask, num_steps=num_steps, shape=const_shape,
                event_feat=event_feat, null_event_feat=null_event_feat,
                guidance_scale=guidance_scale)
            x = alpha_ * mean + sigma_ * eps

        return mean


class WeightedBSMPET_event_c(PET_pp_parton_vpar_bsm_event_c_stage1):
    """|event_weight|-weighted training loss — identical to E023, included for compat."""

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
                x['input_mask'], x['input_jet'][:, 0:1], t, y,
                x['input_event']])
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
            sq_jet      = tf.reduce_mean(tf.square(v_pred - v_jet), axis=1)
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
            x['input_mask'], x['input_jet'][:, 0:1], t, y,
            x['input_event']])
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
        sq_jet      = tf.reduce_mean(tf.square(v_pred - v_jet), axis=1)
        loss_jet    = tf.reduce_sum(w * sq_jet) / (tf.reduce_sum(w) + 1e-10)

        loss = loss_jet + loss_part
        self.loss_tracker.update_state(loss)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)
        return {m.name: m.result() for m in self.metrics}


class WeightedBSMPET_event_c_cfg(WeightedBSMPET_event_c):
    """E034 training class: adds CFG conditioning dropout on cone mass features.

    Only change from WeightedBSMPET_event_c (E023):
      - During train_step, each event independently has its cone mass features
        (indices 4 and 6 of the 7-dim input_event) zeroed with probability
        cfg_dropout_prob. This teaches the model the conditional distribution
        P(particles | log_npart, MET, cone_pT, cone_mass) AND the marginal
        P(particles | log_npart, MET, cone_pT), enabling CFG at inference.
      - Tracks 'cfg_drop' metric: mean fraction of events with dropout applied
        (should be ~cfg_dropout_prob consistently).
      - Validation (test_step) uses full conditioning — val_loss is comparable
        to E023.
    """

    def __init__(self, *args, cfg_dropout_prob=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg_dropout_prob    = cfg_dropout_prob
        self.cfg_dropout_tracker = keras.metrics.Mean(name="cfg_drop")
        # Pre-compute keep mask as a constant tensor (zeros at cone mass indices 4, 6)
        self._event_keep_mask = tf.constant(_EVENT_KEEP_MASK, dtype=tf.float32)

    @property
    def metrics(self):
        return [self.loss_tracker, self.loss_part_tracker,
                self.loss_jet_tracker, self.cfg_dropout_tracker]

    def train_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        mask       = x['input_mask'][:, :, None]
        w          = self._w_norm(x)

        # CFG dropout: zero cone mass features (indices 4, 6) for a random subset of events.
        # Applied before GradientTape — no randomness flows into the gradient computation.
        dropout_mask = tf.random.uniform((batch_size,)) < self.cfg_dropout_prob  # (B,)
        null_event   = x['input_event'] * self._event_keep_mask[None, :]         # (B, 7)
        event_input  = tf.where(dropout_mask[:, None], null_event, x['input_event'])

        self.cfg_dropout_tracker.update_state(tf.cast(dropout_mask, tf.float32))

        with tf.GradientTape(persistent=True) as tape:
            t = tf.random.uniform((batch_size, 1))
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)

            eps         = tf.random.normal(tf.shape(x['input_features']),
                                           dtype=tf.float32) * mask
            perturbed_x = alpha[:, None] * x['input_features'] + eps * sigma[:, None]

            v_pred_part = self.model_part([
                perturbed_x * mask,
                perturbed_x[:, :, :2] * mask,
                x['input_mask'], x['input_jet'][:, 0:1], t, y,
                event_input])  # ← dropout-applied event conditioning
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
            sq_jet      = tf.reduce_mean(tf.square(v_pred - v_jet), axis=1)
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
