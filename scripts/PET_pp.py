"""PET model adapted for full-event pp generation conditioned on MG5 hard-scatter partons.

Conditioning mapping vs original EIC version:
  y  (num_cond=24) = flattened MG5 parton features (4 partons × 6 features), normalised
  input_jet (num_jet=1) = log_npart (generated in stage 1, used as context in stage 2)
  input_features (num_feat=6) = particle cloud [eta, sin_phi, cos_phi, log_pT, pid, charge]
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
from tensorflow.keras.losses import mse
from tensorflow.keras.models import Model
from PET import PET, FourierProjection, get_encoding
from layers import LayerScale
from tqdm import tqdm


class PET_pp(keras.Model):
    def __init__(self,
                 num_feat,
                 num_jet,
                 num_cond=24,
                 num_part=500,
                 projection_dim=128,
                 local=True, K=5,
                 num_local=2,
                 num_layers=8,
                 num_heads=4,
                 drop_probability=0.0,
                 simple=False,
                 layer_scale=True,
                 layer_scale_init=1e-5,
                 talking_head=False,
                 mode='generator',
                 fine_tune=False,
                 model_name=None):
        super(PET_pp, self).__init__()

        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_cond = num_cond
        self.max_part = num_part
        self.projection_dim = projection_dim
        self.layer_scale_init = layer_scale_init
        self.num_steps = 500
        self.ema = 0.999
        self.shape = (-1, 1, 1)

        # PET body + generator head for particle cloud generation
        # num_classes here is the conditioning dim fed as "label" to the generator head
        self.model_part = PET(num_feat=num_feat,
                              num_jet=num_jet,
                              num_classes=num_cond,
                              local=local, K=K,
                              num_layers=num_layers,
                              drop_probability=drop_probability,
                              simple=simple, layer_scale=layer_scale,
                              talking_head=talking_head,
                              mode=mode)

        if fine_tune:
            assert model_name is not None
            self.model_part.load_weights(model_name, by_name=True, skip_mismatch=True)

        self.body = self.model_part.ema_body
        self.head = self.model_part.ema_generator_head

        # Build particle model (body + head wired together)
        inputs_time = Input((1))
        inputs_cond = Input((self.num_cond))
        inputs_jet = Input((self.num_jet))
        inputs_mask = Input((None, 1))
        inputs_features = Input(shape=(None, num_feat))
        inputs_points = Input(shape=(None, 2))

        output_body = self.body([inputs_features, inputs_points, inputs_mask, inputs_time])
        outputs_head = self.head([output_body, inputs_jet, inputs_mask, inputs_time, inputs_cond])
        outputs = inputs_mask * outputs_head

        self.model_part = keras.Model(
            inputs=[inputs_features, inputs_points, inputs_mask, inputs_jet, inputs_time, inputs_cond],
            outputs=outputs)

        # Stage-1 ResNet: generates input_jet (log_npart) from parton conditioning
        outputs_jet = self.Resnet(inputs_jet, inputs_time, inputs_cond,
                                  num_layer=3, mlp_dim=2 * self.projection_dim)
        self.model_jet = Model(inputs=[inputs_jet, inputs_time, inputs_cond], outputs=outputs_jet)

        self.ema_jet = keras.models.clone_model(self.model_jet)
        self.ema_body = keras.models.clone_model(self.body)
        self.ema_head = keras.models.clone_model(self.head)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_part_tracker = keras.metrics.Mean(name="part")
        self.loss_jet_tracker = keras.metrics.Mean(name="jet")

    @property
    def metrics(self):
        return [self.loss_tracker, self.loss_part_tracker, self.loss_jet_tracker]

    def compile(self, body_optimizer, head_optimizer):
        super(PET_pp, self).compile(experimental_run_tf_function=False, weighted_metrics=[])
        self.body_optimizer = body_optimizer
        self.optimizer = head_optimizer

    def Resnet(self, inputs, inputs_time, labels,
               num_layer=3, mlp_dim=128, dropout=0.0):

        def resnet_dense(input_layer, hidden_size, nlayers=2):
            x = input_layer
            residual = layers.Dense(hidden_size)(x)
            for _ in range(nlayers):
                x = layers.Dense(hidden_size, activation='swish')(x)
                x = layers.Dropout(dropout)(x)
            x = LayerScale(self.layer_scale_init, hidden_size)(x)
            return residual + x

        time = FourierProjection(inputs_time, self.projection_dim)
        cond_token = layers.Dense(self.projection_dim)(labels)
        cond_token = layers.Dense(2 * self.projection_dim, activation='gelu')(cond_token + time)
        scale, shift = tf.split(cond_token, 2, -1)

        layer = layers.Dense(self.projection_dim, activation='swish')(inputs)
        layer = layer * (1.0 + scale) + shift

        for _ in range(num_layer - 1):
            layer = layers.LayerNormalization(epsilon=1e-6)(layer)
            layer = resnet_dense(layer, mlp_dim)

        layer = layers.LayerNormalization(epsilon=1e-6)(layer)
        outputs = layers.Dense(self.num_jet, kernel_initializer="zeros")(layer)
        return outputs

    def prior_sde(self, dimensions):
        return tf.random.normal(dimensions, dtype=tf.float32)

    def train_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        mask = x['input_mask'][:, :, None]

        with tf.GradientTape(persistent=True) as tape:
            t = tf.random.uniform((batch_size, 1))
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)

            # Particle diffusion
            eps = tf.random.normal(tf.shape(x['input_features']), dtype=tf.float32) * mask
            perturbed_x = alpha[:, None] * x['input_features'] + eps * sigma[:, None]

            v_pred_part = self.model_part([perturbed_x * mask, perturbed_x[:, :, :2] * mask,
                                           x['input_mask'], x['input_jet'], t, y])
            v_pred_part = tf.reshape(v_pred_part, (tf.shape(v_pred_part)[0], -1))
            v_part = alpha[:, None] * eps - sigma[:, None] * x['input_features']
            v_part = tf.reshape(v_part, (tf.shape(v_part)[0], -1))
            loss_part = tf.reduce_sum(tf.square(v_part - v_pred_part)) / tf.reduce_sum(x['input_mask'])

            # Jet (log_npart) diffusion
            eps = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
            perturbed_x = alpha * x['input_jet'] + eps * sigma
            v_pred = self.model_jet([perturbed_x, t, y])
            v_jet = alpha * eps - sigma * x['input_jet']
            loss_jet = tf.reduce_mean(tf.square(v_pred - v_jet))

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
        mask = x['input_mask'][:, :, None]

        t = tf.random.uniform((batch_size, 1))
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)

        eps = tf.random.normal(tf.shape(x['input_features']), dtype=tf.float32) * mask
        perturbed_x = alpha[:, None] * x['input_features'] + eps * sigma[:, None]

        v_pred_part = self.model_part([perturbed_x * mask, perturbed_x[:, :, :2] * mask,
                                       x['input_mask'], x['input_jet'], t, y])
        v_pred_part = tf.reshape(v_pred_part, (tf.shape(v_pred_part)[0], -1))
        v_part = alpha[:, None] * eps - sigma[:, None] * x['input_features']
        v_part = tf.reshape(v_part, (tf.shape(v_part)[0], -1))
        loss_part = tf.reduce_sum(tf.square(v_part - v_pred_part)) / tf.reduce_sum(x['input_mask'])

        eps = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
        perturbed_x = alpha * x['input_jet'] + eps * sigma
        v_pred = self.model_jet([perturbed_x, t, y])
        v_jet = alpha * eps - sigma * x['input_jet']
        loss_jet = tf.reduce_mean(tf.square(v_pred - v_jet))

        loss = loss_jet + loss_part
        self.loss_tracker.update_state(loss)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        return self.model_part(x)

    def generate(self, cond, jet_mean, jet_std, nsplit=2, jets=None, use_tqdm=False):
        """Generate particle clouds conditioned on normalised parton features.

        Args:
            cond: normalised parton conditioning, shape (N, 24)
            jet_mean, jet_std: scalars for log_npart denormalisation
            nsplit: number of chunks (reduce if OOM)
            jets: pre-computed normalised log_npart; if None, sampled from model_jet
        """
        jet_info = []
        part_info = []
        jet_split = np.array_split(jets, nsplit) if jets is not None else None
        splits = np.array_split(cond, nsplit)

        for i, split in (tqdm(enumerate(splits), total=len(splits)) if use_tqdm else enumerate(splits)):
            if jets is not None:
                jet = jet_split[i]
            else:
                jet = self.DDPMSampler(split, self.ema_jet,
                                       data_shape=[split.shape[0], self.num_jet],
                                       w=0.0, num_steps=512,
                                       const_shape=[-1, 1]).numpy()

            jet_info.append(jet)

            log_npart = jet[:, 0] * jet_std + jet_mean
            nparts = np.expand_dims(
                np.clip(np.round(np.exp(log_npart)).astype(int), 1, self.max_part), -1)
            mask = np.expand_dims(
                np.tile(np.arange(self.max_part), (nparts.shape[0], 1)) < np.tile(nparts, (1, self.max_part)),
                -1)

            parts = self.DDPMSampler(split, [self.ema_body, self.ema_head],
                                     data_shape=[split.shape[0], self.max_part, self.num_feat],
                                     jet=jet, num_steps=self.num_steps,
                                     const_shape=self.shape, w=0.0,
                                     mask=mask.astype(np.float32)).numpy()
            part_info.append(parts * mask)

        return np.concatenate(part_info), np.concatenate(jet_info)

    # ── Diffusion schedule helpers (identical to EIC version) ─────────────────

    def logsnr_schedule_cosine(self, t, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t, tf.float32) + b))

    def get_logsnr_alpha_sigma(self, time, shape=None):
        logsnr = self.logsnr_schedule_cosine(time)
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))
        if shape is not None:
            alpha = tf.reshape(alpha, shape)
            sigma = tf.reshape(sigma, shape)
            logsnr = tf.reshape(logsnr, shape)
        return logsnr, tf.cast(alpha, tf.float32), tf.cast(sigma, tf.float32)

    @tf.function
    def second_order_correction(self, time_step, x, pred_images, pred_noises,
                                alphas, sigmas, w, cond, model,
                                jet=None, mask=None, num_steps=100,
                                second_order_alpha=0.5, shape=None):
        step_size = 1.0 / num_steps
        t = time_step - second_order_alpha * step_size
        logsnr, alpha_s, alpha_n = self.get_logsnr_alpha_sigma(t, shape=shape)
        alpha_noisy = alpha_s * pred_images + alpha_n * pred_noises

        if jet is None:
            v = model([alpha_noisy, t, cond], training=False)
        else:
            alpha_noisy *= mask
            model_body, model_head = model
            v = self.evaluate_models(model_head, model_body, alpha_noisy, jet, mask, t, cond, w)

        alpha_pred_noises = alpha_n * alpha_noisy + alpha_s * v
        pred_noises = ((1.0 - 1.0 / (2.0 * second_order_alpha)) * pred_noises
                       + 1.0 / (2.0 * second_order_alpha) * alpha_pred_noises)

        mean = (x - sigmas * pred_noises) / alphas
        return mean, pred_noises

    def evaluate_models(self, head, body, x, jet, mask, t, cond, w=0.0):
        x_in = mask * x
        v = body([x_in, x[:, :, :2], mask, t], training=False)
        v = head([v, jet, mask, t, cond], training=False)
        return mask * v

    @tf.function
    def DDPMSampler(self, cond, model, data_shape=None, const_shape=None,
                    jet=None, w=0.1, num_steps=100, mask=None):
        batch_size = cond.shape[0]
        x = self.prior_sde(data_shape)

        for time_step in tf.range(num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / num_steps
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t, shape=const_shape)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(
                tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / num_steps,
                shape=const_shape)

            if jet is None:
                v = model([x, t, cond], training=False)
            else:
                x *= mask
                model_body, model_head = model
                v = self.evaluate_models(model_head, model_body, x, jet, mask, t, cond, w)

            mean = alpha * x - sigma * v
            eps = v * alpha + x * sigma
            mean, eps = self.second_order_correction(t, x, mean, eps, alpha, sigma, w,
                                                     cond, model, jet, mask,
                                                     num_steps=num_steps, shape=const_shape)
            x = alpha_ * mean + sigma_ * eps

        return mean
