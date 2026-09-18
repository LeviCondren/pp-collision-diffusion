"""PET_pp_parton_vpar_bsm_event_c_stage1_cfg_asym — E035: asymmetric per-cone CFG dropout.

Copied from PET_pp_parton_vpar_bsm_event_c_stage1_cfg.py (E034) and modified for E035:
  - Architecture identical to E034/E023 (num_gen_layers=2, standard cross-attention).
  - Replaces symmetric CFG dropout (both cone masses dropped together) with
    independent per-cone dropout: cone_mass_X (index 4) and cone_mass_Y (index 6)
    of the 7-dim event vector are each independently dropped with their own
    probabilities (default 0.10 each).
  - Inference: four-pass per-cone CFG (four model evals per denoising step):
      v_full:  all event features present
      v_no_x:  cone_mass_X zeroed
      v_no_y:  cone_mass_Y zeroed
      v_null:  both zeroed
      v_guided = v_null + gs_x*(v_full - v_no_x) + gs_y*(v_full - v_no_y)
  - WeightedBSMPET_event_c_cfg_asym subclass with 4 dropout-fraction metrics.
  - Stage-1 ResNet, EMA, and diffusion schedule unchanged from E034.
  - E038 addition (additive only — generate_asym_cfg/DDPMSamplerAsymCFG untouched,
    still used by E036/E037): generate_composable_cfg / DDPMSamplerComposableCFG /
    second_order_correction_composable_cfg implement the composable-diffusion
    formula v_guided = v_null + gs_x*(v_no_y-v_null) + gs_y*(v_no_x-v_null),
    a 3-pass alternative to generate_asym_cfg's 4-pass formula that never
    references v_full and so cannot double-count it (generate_asym_cfg's
    coefficient on v_full is gs_x+gs_y = 3.0 at gs_x=gs_y=1.5, which is the
    E037 amplification bug). Intended for use with E038's checkpoint
    (cfg_drop_x_prob=cfg_drop_y_prob=0.35, vs E036's 0.10/0.10), which trains
    v_null on ~12% of batches instead of E036's ~1%.

Do NOT modify E034 canonical scripts.
Do NOT mix checkpoints from this file with E034 checkpoints.
Do NOT modify generate_asym_cfg / DDPMSamplerAsymCFG / second_order_correction_asym_cfg
  — E036's checkpoint and E037's inference results depend on them as-is.

Changelog:
  E034 (2026-08-01): symmetric CFG training and two-pass inference.
  E035/E036 (2026-08-14): asymmetric per-cone CFG dropout (0.10/0.10) and
    four-pass per-cone inference (generate_asym_cfg).
  E038 (2026-09-09): added 3-pass composable-diffusion inference
    (generate_composable_cfg) as an alternative formula that avoids
    double-counting v_full; paired with a retrain at higher per-cone
    dropout (0.35/0.35) to raise v_null's training exposure.
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
_EVENT_KEEP_MASK    = [1., 1., 1., 1., 0., 1., 0.]  # zeros at indices 4 and 6 (both)
_CONE_MASS_X_KEEP   = [1., 1., 1., 1., 0., 1., 1.]  # zero only index 4 (cone_mass_X)
_CONE_MASS_Y_KEEP   = [1., 1., 1., 1., 1., 1., 0.]  # zero only index 6 (cone_mass_Y)


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

    def generate_asym_cfg(self, cond, jet_mean, jet_std,
                          nsplit=2, jets=None, use_tqdm=False, num_steps=None,
                          num_jet_steps=None, guidance_scale_x=1.0, guidance_scale_y=1.0,
                          use_true_cone=False, event_feat=None):
        """Four-pass per-cone CFG generation (E035).

        Four model evaluations per denoising step:
          v_full:  all event features (cone_mass_X and cone_mass_Y present)
          v_no_x:  cone_mass_X zeroed (index 4 of 7-dim event vector)
          v_no_y:  cone_mass_Y zeroed (index 6 of 7-dim event vector)
          v_null:  both cone masses zeroed

          v_guided = v_null
                     + guidance_scale_x * (v_full - v_no_x)
                     + guidance_scale_y * (v_full - v_no_y)

        Limit cases:
          (gs_x=0, gs_y=0) → v_null (fully unconditional on cone masses)
          (gs_x=1, gs_y=1) → approximately v_full (if cones are independent)
          (gs_x>1, gs_y=0) → strong X guidance, no Y guidance
          (gs_x=0, gs_y>1) → strong Y guidance, no X guidance

        Note: 4x model evaluations per step vs E034's 2-pass CFG.
        """
        part_steps      = num_steps if num_steps is not None else self.num_steps
        jet_steps_count = num_jet_steps if num_jet_steps is not None else 512

        jet_info  = []
        part_info = []
        jet_split = np.array_split(jets, nsplit) if jets is not None else None
        splits    = np.array_split(cond, nsplit)
        ev_splits = np.array_split(event_feat, nsplit) if event_feat is not None else None

        gs_x = tf.constant(guidance_scale_x, dtype=tf.float32)
        gs_y = tf.constant(guidance_scale_y, dtype=tf.float32)

        print(f'  generate_asym_cfg: guidance_scale_x={guidance_scale_x}  '
              f'guidance_scale_y={guidance_scale_y}  '
              f'num_steps={part_steps}  nsplit={nsplit}')

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

            # Build four event conditioning arrays
            if use_true_cone and ev_splits is not None:
                ev_full = ev_splits[i]
            else:
                ev_full = jet[:, 1:]  # stage-1 generated event features (N, 7)

            ev_no_x          = ev_full.copy()
            ev_no_x[:, 4]    = 0.0   # log1p_cone_mass_X → null

            ev_no_y          = ev_full.copy()
            ev_no_y[:, 6]    = 0.0   # log1p_cone_mass_Y → null

            ev_null          = ev_full.copy()
            ev_null[:, 4]    = 0.0
            ev_null[:, 6]    = 0.0   # both zeroed

            parts = self.DDPMSamplerAsymCFG(
                split, [self.ema_body, self.ema_head],
                data_shape=[split.shape[0], self.max_part, self.num_feat],
                jet=log_npart_norm, num_steps=part_steps,
                const_shape=self.shape, w=0.0,
                mask=mask_arr.astype(np.float32),
                event_feat=ev_full,
                no_x_event_feat=ev_no_x,
                no_y_event_feat=ev_no_y,
                null_event_feat=ev_null,
                guidance_scale_x=gs_x,
                guidance_scale_y=gs_y).numpy()
            part_info.append(parts * mask_arr)

        return np.concatenate(part_info), np.concatenate(jet_info)

    def generate_composable_cfg(self, cond, jet_mean, jet_std,
                          nsplit=2, jets=None, use_tqdm=False, num_steps=None,
                          num_jet_steps=None, guidance_scale_x=1.0, guidance_scale_y=1.0,
                          use_true_cone=False, event_feat=None):
        """Three-pass composable-diffusion CFG generation (E038).

        Three model evaluations per denoising step (v_full is not used):
          v_no_x:  cone_mass_X zeroed (index 4)   — "Y-only" branch
          v_no_y:  cone_mass_Y zeroed (index 6)   — "X-only" branch
          v_null:  both cone masses zeroed

          v_guided = v_null
                     + guidance_scale_x * (v_no_y - v_null)
                     + guidance_scale_y * (v_no_x - v_null)

        This is the standard composable-diffusion formula (Liu et al.) generalized
        to two independent conditioning signals. Unlike generate_asym_cfg, v_full
        never appears in the combination, so there is no double-counting of v_full
        when both scales are active (generate_asym_cfg's coefficient on v_full is
        gs_x+gs_y, which is 3.0 at gs_x=gs_y=1.5; this formula's coefficient on
        v_null is 1-gs_x-gs_y and v_full is never referenced at all).

        Limit cases:
          (gs_x=0, gs_y=0) → v_null (fully unconditional on cone masses)
          (gs_x=1, gs_y=1) → v_no_y + v_no_x - v_null (additive approx of v_full,
                              exact iff X and Y contribute independently)

        Note: 3x model evaluations per step vs generate_asym_cfg's 4x.
        """
        part_steps      = num_steps if num_steps is not None else self.num_steps
        jet_steps_count = num_jet_steps if num_jet_steps is not None else 512

        jet_info  = []
        part_info = []
        jet_split = np.array_split(jets, nsplit) if jets is not None else None
        splits    = np.array_split(cond, nsplit)
        ev_splits = np.array_split(event_feat, nsplit) if event_feat is not None else None

        gs_x = tf.constant(guidance_scale_x, dtype=tf.float32)
        gs_y = tf.constant(guidance_scale_y, dtype=tf.float32)

        print(f'  generate_composable_cfg: guidance_scale_x={guidance_scale_x}  '
              f'guidance_scale_y={guidance_scale_y}  '
              f'num_steps={part_steps}  nsplit={nsplit}')

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

            if use_true_cone and ev_splits is not None:
                ev_full = ev_splits[i]
            else:
                ev_full = jet[:, 1:]  # stage-1 generated event features (N, 7)

            ev_no_x          = ev_full.copy()
            ev_no_x[:, 4]    = 0.0   # log1p_cone_mass_X → null

            ev_no_y          = ev_full.copy()
            ev_no_y[:, 6]    = 0.0   # log1p_cone_mass_Y → null

            ev_null          = ev_full.copy()
            ev_null[:, 4]    = 0.0
            ev_null[:, 6]    = 0.0   # both zeroed

            parts = self.DDPMSamplerComposableCFG(
                split, [self.ema_body, self.ema_head],
                data_shape=[split.shape[0], self.max_part, self.num_feat],
                jet=log_npart_norm, num_steps=part_steps,
                const_shape=self.shape, w=0.0,
                mask=mask_arr.astype(np.float32),
                no_x_event_feat=ev_no_x,
                no_y_event_feat=ev_no_y,
                null_event_feat=ev_null,
                guidance_scale_x=gs_x,
                guidance_scale_y=gs_y).numpy()
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
    def second_order_correction_asym_cfg(self, time_step, x, pred_images, pred_noises,
                                          alphas, sigmas, w, cond4, model,
                                          jet4=None, mask4=None, num_steps=100,
                                          second_order_alpha=0.5, shape=None,
                                          ev4=None, guidance_scale_x=1.0,
                                          guidance_scale_y=1.0):
        """Asym-CFG second-order correction using a single 4x batched model call.

        Signature uses pre-tiled cond4/jet4/mask4/ev4 (prepared by DDPMSamplerAsymCFG).
        v_guided = v_null + gs_x*(v_full - v_no_x) + gs_y*(v_full - v_no_y)
        """
        step_size   = 1.0 / num_steps
        t_half      = time_step - second_order_alpha * step_size
        logsnr, alpha_s, alpha_n = self.get_logsnr_alpha_sigma(t_half, shape=shape)
        alpha_noisy = alpha_s * pred_images + alpha_n * pred_noises  # [B, parts, feat]

        B      = tf.shape(alpha_noisy)[0]
        mask_B = mask4[:B]          # [B, parts, 1] — first quarter of 4x-tiled mask
        alpha_noisy *= mask_B

        # Single 4x-batched model call at the half-step
        an4    = tf.tile(alpha_noisy, [4, 1, 1])
        t4     = tf.tile(tf.cast(t_half, tf.float32), [4, 1])
        model_body, model_head = model
        v4 = self.evaluate_models(model_head, model_body, an4, jet4, mask4, t4, cond4, w,
                                  event_feat=ev4)

        v_full, v_no_x, v_no_y, v_null = tf.split(v4, 4, axis=0)
        v = (v_null
             + guidance_scale_x * (v_full - v_no_x)
             + guidance_scale_y * (v_full - v_no_y))

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
    def DDPMSamplerAsymCFG(self, cond, model, data_shape=None, const_shape=None,
                            jet=None, w=0.1, num_steps=100, mask=None,
                            event_feat=None, no_x_event_feat=None,
                            no_y_event_feat=None, null_event_feat=None,
                            guidance_scale_x=1.0, guidance_scale_y=1.0):
        """Four-pass asymmetric CFG DDPM sampler using a single 4x batched model call.

        Pre-tiles the four conditioning variants into one 4x batch before the
        denoising loop so the loop compiles as a single @tf.function graph.
        Peak GPU memory per step = one forward pass at 4*chunk_size events,
        matching DDPMSampler at the same effective batch.  Use chunk_size ~N/4
        relative to the symmetric 2-pass sampler (e.g. chunk_size=50 instead
        of 100) to keep the effective batch through the model the same.

        Per denoising step:
          v_guided = v_null
                     + guidance_scale_x * (v_full - v_no_x)
                     + guidance_scale_y * (v_full - v_no_y)
        """
        batch_size = cond.shape[0]
        x          = self.prior_sde(data_shape)

        # Pre-tile the four conditioning arrays (constant across all time steps)
        ev4   = tf.concat([event_feat, no_x_event_feat, no_y_event_feat, null_event_feat],
                          axis=0)
        cond4 = tf.tile(cond, [4, 1])
        jet4  = tf.tile(jet,  [4, 1])
        mask4 = tf.tile(mask, [4, 1, 1])

        for time_step in tf.range(num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / num_steps
            logsnr,  alpha,  sigma  = self.get_logsnr_alpha_sigma(t, shape=const_shape)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(
                tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / num_steps,
                shape=const_shape)

            x *= mask

            # Single 4x-batched model call (one graph node, not four)
            t4 = tf.tile(t, [4, 1])
            x4 = tf.tile(x, [4, 1, 1])
            model_body, model_head = model
            v4 = self.evaluate_models(model_head, model_body, x4, jet4, mask4, t4, cond4, w,
                                      event_feat=ev4)

            v_full, v_no_x, v_no_y, v_null = tf.split(v4, 4, axis=0)
            v = (v_null
                 + guidance_scale_x * (v_full - v_no_x)
                 + guidance_scale_y * (v_full - v_no_y))

            mean = alpha * x - sigma * v
            eps  = v * alpha + x * sigma
            mean, eps = self.second_order_correction_asym_cfg(
                t, x, mean, eps, alpha, sigma, w, cond4, model,
                jet4, mask4, num_steps=num_steps, shape=const_shape,
                ev4=ev4, guidance_scale_x=guidance_scale_x,
                guidance_scale_y=guidance_scale_y)
            x = alpha_ * mean + sigma_ * eps

        return mean

    def second_order_correction_composable_cfg(self, time_step, x, pred_images, pred_noises,
                                          alphas, sigmas, w, cond3, model,
                                          jet3=None, mask3=None, num_steps=100,
                                          second_order_alpha=0.5, shape=None,
                                          ev3=None, guidance_scale_x=1.0,
                                          guidance_scale_y=1.0):
        """Composable-CFG second-order correction using a single 3x batched model call.

        Signature uses pre-tiled cond3/jet3/mask3/ev3 (prepared by DDPMSamplerComposableCFG).
        v_guided = v_null + gs_x*(v_no_y - v_null) + gs_y*(v_no_x - v_null)
        """
        step_size   = 1.0 / num_steps
        t_half      = time_step - second_order_alpha * step_size
        logsnr, alpha_s, alpha_n = self.get_logsnr_alpha_sigma(t_half, shape=shape)
        alpha_noisy = alpha_s * pred_images + alpha_n * pred_noises  # [B, parts, feat]

        B      = tf.shape(alpha_noisy)[0]
        mask_B = mask3[:B]          # [B, parts, 1] — first third of 3x-tiled mask
        alpha_noisy *= mask_B

        # Single 3x-batched model call at the half-step
        an3    = tf.tile(alpha_noisy, [3, 1, 1])
        t3     = tf.tile(tf.cast(t_half, tf.float32), [3, 1])
        model_body, model_head = model
        v3 = self.evaluate_models(model_head, model_body, an3, jet3, mask3, t3, cond3, w,
                                  event_feat=ev3)

        v_no_x, v_no_y, v_null = tf.split(v3, 3, axis=0)
        v = (v_null
             + guidance_scale_x * (v_no_y - v_null)
             + guidance_scale_y * (v_no_x - v_null))

        alpha_pred_noises = alpha_n * alpha_noisy + alpha_s * v
        pred_noises = ((1.0 - 1.0 / (2.0 * second_order_alpha)) * pred_noises
                       + 1.0 / (2.0 * second_order_alpha) * alpha_pred_noises)

        mean = (x - sigmas * pred_noises) / alphas
        return mean, pred_noises

    def DDPMSamplerComposableCFG(self, cond, model, data_shape=None, const_shape=None,
                            jet=None, w=0.1, num_steps=100, mask=None,
                            no_x_event_feat=None, no_y_event_feat=None,
                            null_event_feat=None,
                            guidance_scale_x=1.0, guidance_scale_y=1.0):
        """Three-pass composable-diffusion CFG DDPM sampler (E038), single 3x batched call.

        Pre-tiles the three conditioning variants (no_x, no_y, null — v_full is not
        needed for this formula) into one 3x batch before the denoising loop so the
        loop compiles as a single @tf.function graph.

        Per denoising step:
          v_guided = v_null
                     + guidance_scale_x * (v_no_y - v_null)
                     + guidance_scale_y * (v_no_x - v_null)
        """
        batch_size = cond.shape[0]
        x          = self.prior_sde(data_shape)

        # Pre-tile the three conditioning arrays (constant across all time steps)
        ev3   = tf.concat([no_x_event_feat, no_y_event_feat, null_event_feat], axis=0)
        cond3 = tf.tile(cond, [3, 1])
        jet3  = tf.tile(jet,  [3, 1])
        mask3 = tf.tile(mask, [3, 1, 1])

        for time_step in tf.range(num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / num_steps
            logsnr,  alpha,  sigma  = self.get_logsnr_alpha_sigma(t, shape=const_shape)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(
                tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / num_steps,
                shape=const_shape)

            x *= mask

            # Single 3x-batched model call (one graph node, not three)
            t3 = tf.tile(t, [3, 1])
            x3 = tf.tile(x, [3, 1, 1])
            model_body, model_head = model
            v3 = self.evaluate_models(model_head, model_body, x3, jet3, mask3, t3, cond3, w,
                                      event_feat=ev3)

            v_no_x, v_no_y, v_null = tf.split(v3, 3, axis=0)
            v = (v_null
                 + guidance_scale_x * (v_no_y - v_null)
                 + guidance_scale_y * (v_no_x - v_null))

            mean = alpha * x - sigma * v
            eps  = v * alpha + x * sigma
            mean, eps = self.second_order_correction_composable_cfg(
                t, x, mean, eps, alpha, sigma, w, cond3, model,
                jet3, mask3, num_steps=num_steps, shape=const_shape,
                ev3=ev3, guidance_scale_x=guidance_scale_x,
                guidance_scale_y=guidance_scale_y)
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


class WeightedBSMPET_event_c_cfg_asym(WeightedBSMPET_event_c):
    """E035 training class: independent per-cone CFG dropout on cone mass features.

    Replaces E034's symmetric dropout (both cone masses dropped together at prob p)
    with independent Bernoulli dropout per cone:
      - cone_mass_X (index 4 of 7-dim event vector) dropped with prob cfg_drop_x_prob
      - cone_mass_Y (index 6 of 7-dim event vector) dropped with prob cfg_drop_y_prob

    With default p_x=p_y=0.10 the marginal fractions are:
      drop_X marginal ≈ 0.10  (X dropped regardless of Y)
      drop_Y marginal ≈ 0.10
      drop_both       ≈ 0.01  (both dropped together)
      drop_none       ≈ 0.81  (full conditioning)

    This teaches the model to handle each cone's conditioning independently,
    enabling per-cone CFG at inference via generate_asym_cfg / DDPMSamplerAsymCFG.

    Validation (test_step) uses full conditioning — val_loss is comparable to E034.
    Loss function unchanged from E034 (|w|-weighted MSE).
    """

    def __init__(self, *args, cfg_drop_x_prob=0.10, cfg_drop_y_prob=0.10, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg_drop_x_prob = cfg_drop_x_prob
        self.cfg_drop_y_prob = cfg_drop_y_prob
        # Pre-compute per-cone keep masks as constant tensors
        self._cone_x_keep_mask  = tf.constant(_CONE_MASS_X_KEEP,  dtype=tf.float32)  # zeros index 4
        self._cone_y_keep_mask  = tf.constant(_CONE_MASS_Y_KEEP,  dtype=tf.float32)  # zeros index 6
        # Metric trackers: marginal drop_X, drop_Y, and joint cells
        self.drop_x_tracker    = keras.metrics.Mean(name="drop_X")
        self.drop_y_tracker    = keras.metrics.Mean(name="drop_Y")
        self.drop_both_tracker = keras.metrics.Mean(name="drop_both")
        self.drop_none_tracker = keras.metrics.Mean(name="drop_none")

    @property
    def metrics(self):
        return [self.loss_tracker, self.loss_part_tracker, self.loss_jet_tracker,
                self.drop_x_tracker, self.drop_y_tracker,
                self.drop_both_tracker, self.drop_none_tracker]

    def train_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        mask       = x['input_mask'][:, :, None]
        w          = self._w_norm(x)

        # ── Per-cone independent CFG dropout ──────────────────────────────────
        # Each cone mass is dropped independently; four conditioning states result:
        #   (keep X, keep Y), (drop X, keep Y), (keep X, drop Y), (drop X, drop Y)
        drop_X = tf.random.uniform((batch_size,)) < self.cfg_drop_x_prob  # (B,) bool
        drop_Y = tf.random.uniform((batch_size,)) < self.cfg_drop_y_prob  # (B,) bool

        # Apply X dropout: zero index 4 where drop_X is True
        ev_after_x = tf.where(drop_X[:, None],
                               x['input_event'] * self._cone_x_keep_mask[None, :],
                               x['input_event'])
        # Apply Y dropout to the result: zero index 6 where drop_Y is True
        event_input = tf.where(drop_Y[:, None],
                                ev_after_x * self._cone_y_keep_mask[None, :],
                                ev_after_x)

        # Track dropout fractions (marginal and joint)
        drop_both_mask = drop_X & drop_Y
        drop_none_mask = ~drop_X & ~drop_Y
        self.drop_x_tracker.update_state(tf.cast(drop_X,         tf.float32))
        self.drop_y_tracker.update_state(tf.cast(drop_Y,         tf.float32))
        self.drop_both_tracker.update_state(tf.cast(drop_both_mask, tf.float32))
        self.drop_none_tracker.update_state(tf.cast(drop_none_mask, tf.float32))

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
                event_input])   # ← per-cone dropout-applied event conditioning
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
