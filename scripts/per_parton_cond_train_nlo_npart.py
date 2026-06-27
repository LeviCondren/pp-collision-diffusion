"""NLO variant of per_parton_cond_train.py.

Differences from the LO version:
  - Loads event_weights from HDF5 and weights the DDPM loss by |w|/mean(|w|)
  - Default data_dir points to the NLO dataset
  - Default run_name = 'parton_v1_nlo'

The weighted loss is implemented via WeightedPET_pp_parton, a thin subclass
of PET_pp_parton that overrides train_step and test_step.  Per-batch weight
normalisation (w_norm = |w| / mean(|w|)) keeps the loss magnitude stable.

Launch (multi-node via srun):
  srun python3 per_parton_cond_train_nlo.py [args]

Launch (single-node smoke, 1 GPU via horovodrun):
  horovodrun --gloo -np 1 python3 per_parton_cond_train_nlo.py \
      --run_name parton_nlo_smoke --batch 16 --epoch 1 \
      --num_layers 2 --proj_dim 32 --num_part 100 \
      --n_train 50 --n_val 20 --time_limit_hours 0.4
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

from PET_pp_parton import PET_pp_parton
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import schedules, Adam


class WeightedPET_pp_parton(PET_pp_parton):
    """PET_pp_parton with |event_weight|-weighted DDPM loss for NLO training.

    Expects x['input_weight'] shape (batch,) in the input dict.
    Weights are normalised per-batch: w_norm = |w| / mean(|w|).
    """

    def _w_norm(self, x):
        abs_w = tf.abs(x['input_weight'])                      # (batch,)
        return abs_w / (tf.reduce_mean(abs_w) + 1e-10)

    def train_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        mask       = x['input_mask'][:, :, None]
        w          = self._w_norm(x)                           # (batch,)

        with tf.GradientTape(persistent=True) as tape:
            t = tf.random.uniform((batch_size, 1))
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)

            # Particle diffusion
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

            sq_per_event   = tf.reduce_sum(tf.square(v_part - v_pred_part), axis=1)  # (batch,)
            mask_per_event = tf.reduce_sum(x['input_mask'], axis=1)                   # (batch,)
            mse_per_event  = sq_per_event / (mask_per_event + 1e-10)
            loss_part      = tf.reduce_sum(w * mse_per_event) / (tf.reduce_sum(w) + 1e-10)

            # Jet (log_npart) diffusion
            eps         = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
            perturbed_x = alpha * x['input_jet'] + eps * sigma
            v_pred      = self.model_jet([perturbed_x, t, y])
            v_jet       = alpha * eps - sigma * x['input_jet']
            sq_jet      = tf.squeeze(tf.square(v_pred - v_jet), axis=1)             # (batch,)
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

_FULL_DATA     = '/pscratch/sd/l/lcondren/MCsim/full_event_fpcd_nlo'
_FULL_DATA_LO  = '/pscratch/sd/l/lcondren/MCsim/full_event_fpcd'
STATS_PATH     = f'{_FULL_DATA}/normalisation_stats.json'
CKPT_BASE      = f'{_FULL_DATA_LO}/checkpoints_pet_pp'

NUM_FEAT    = 6   # eta, sin_phi, cos_phi, log_pT, pid, charge
NUM_JET     = 1   # log_npart
NUM_COND    = 25  # 4 partons × 6 features + 1 normalised npart
NUM_PARTONS = 4
PARTON_FEAT = 6


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
    """Must run on all ranks to avoid AllReduce deadlock."""
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',          default=_FULL_DATA)
    p.add_argument('--run_name',          default='parton_v1_nlo_npart')
    p.add_argument('--batch',             type=int,   default=128)
    p.add_argument('--epoch',             type=int,   default=200)
    p.add_argument('--lr',                type=float, default=3e-4)
    p.add_argument('--lr_body',           type=float, default=1e-4)
    p.add_argument('--num_layers',        type=int,   default=8)
    p.add_argument('--num_gen_layers',    type=int,   default=2,
                   help='Generator head depth (cross-attn layers per block)')
    p.add_argument('--proj_dim',          type=int,   default=128)
    p.add_argument('--num_part',          type=int,   default=500)
    p.add_argument('--local',             action='store_true', default=True)
    p.add_argument('--no_local',          dest='local', action='store_false')
    p.add_argument('--K',                 type=int,   default=5)
    p.add_argument('--layer_scale',       action='store_true', default=True)
    p.add_argument('--simple',            action='store_true', default=False)
    p.add_argument('--talking_head',      action='store_true', default=False)
    p.add_argument('--drop_prob',         type=float, default=0.0)
    p.add_argument('--patience',          type=int,   default=30)
    p.add_argument('--val_start',         type=int,   default=400000)
    p.add_argument('--n_train',           type=int,   default=None)
    p.add_argument('--n_val',             type=int,   default=10000)
    p.add_argument('--processes',         nargs='+',  default=['dijet', 'zjets'])
    p.add_argument('--fine_tune',         action='store_true', default=False)
    p.add_argument('--model_name',        default=None)
    p.add_argument('--time_limit_hours',  type=float, default=3.5)
    return p.parse_args()


def load_shard(data_dir, stats_path, processes, hvd_rank, hvd_size,
               val_start, n_events_per_proc, num_part, split):
    with open(stats_path) as f:
        stats = json.load(f)
    part_mean = np.array(stats['part_mean'], dtype=np.float32)
    part_std  = np.array(stats['part_std'],  dtype=np.float32)
    jet_mean  = float(stats['jet_mean'][0])
    jet_std   = float(stats['jet_std'][0])
    cond_mean = np.array(stats['cond_mean'], dtype=np.float32)
    cond_std  = np.array(stats['cond_std'],  dtype=np.float32)
    cond_std  = np.where(cond_std > 0, cond_std, 1.0)

    all_pf, all_mask, all_cond, all_jet, all_w, all_npart = [], [], [], [], [], []

    for proc in processes:
        path = f'{data_dir}/{proc}.hdf5'
        with h5py.File(path, 'r') as f:
            n_total = f['particle_features'].shape[0]
            if split == 'train':
                start_global, end_global = 0, val_start
            else:
                start_global, end_global = val_start, n_total

            if n_events_per_proc is not None:
                end_global = min(start_global + n_events_per_proc, end_global)

            n_proc   = end_global - start_global
            per_rank = n_proc // hvd_size
            r0       = start_global + hvd_rank * per_rank
            r1       = start_global + (hvd_rank + 1) * per_rank

            pf   = f['particle_features'][r0:r1].astype(np.float32)
            part = f['parton_features'][r0:r1].astype(np.float32)
            ew   = f['event_weights'][r0:r1].astype(np.float32)

        mask      = pf[:, :num_part, 6].astype(np.float32)
        pf6       = pf[:, :num_part, :6]
        npart     = mask.sum(axis=1, keepdims=True).astype(np.float32)
        log_npart = np.log(np.maximum(npart, 1.0))
        jet       = (log_npart - jet_mean) / jet_std
        cond_raw  = part.reshape(part.shape[0], 24)
        cond      = (cond_raw - cond_mean) / cond_std
        pf6_norm  = (pf6 - part_mean) / part_std * mask[:, :, None]

        # Drop negative-weight events: signed weighting is pathological for diffusion
        # models (loss can go negative, gradients fight each other). Train on
        # positive-weight events and correct for the shape difference at inference.
        pos = ew > 0
        if not pos.all():
            n_neg = int((~pos).sum())
            print(f"  [{proc}] dropped {n_neg}/{len(ew)} negative-weight events "
                  f"({100*n_neg/len(ew):.1f}%)", flush=True)
            pf6_norm = pf6_norm[pos]
            mask     = mask[pos]
            npart    = npart[pos]
            cond     = cond[pos]
            jet      = jet[pos]
            ew       = ew[pos]

        all_pf.append(pf6_norm)
        all_mask.append(mask)
        all_cond.append(cond)
        all_jet.append(jet)
        all_w.append(ew)
        all_npart.append(npart.squeeze(1))
        del pf, part

    pf_all    = np.concatenate(all_pf,    axis=0)
    mask_all  = np.concatenate(all_mask,  axis=0)
    cond_all  = np.concatenate(all_cond,  axis=0)
    jet_all   = np.concatenate(all_jet,   axis=0)
    w_all     = np.concatenate(all_w,     axis=0)
    npart_all = np.concatenate(all_npart, axis=0)

    seed = 42 + hvd_rank if split == 'train' else 0
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(pf_all))

    return (pf_all[idx], mask_all[idx], cond_all[idx], jet_all[idx], w_all[idx],
            npart_all[idx], part_mean, part_std, jet_mean, jet_std)


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


def main():
    flags     = parse_args()
    run_dir    = os.path.join(CKPT_BASE, flags.run_name)
    ckpt_path  = os.path.join(run_dir, 'pet_pp.weights.h5')
    state_path = os.path.join(run_dir, 'training_state.json')

    if hvd.rank() == 0:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(CKPT_BASE, 'histories'), exist_ok=True)

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

    tr_pf, tr_mask, tr_cond, tr_jet, tr_w, tr_npart, part_mean, part_std, jet_mean, jet_std = load_shard(
        data_dir=flags.data_dir, stats_path=STATS_PATH, processes=flags.processes,
        hvd_rank=hvd.rank(), hvd_size=hvd.size(), val_start=flags.val_start,
        n_events_per_proc=flags.n_train, num_part=flags.num_part, split='train')
    n_local_train = len(tr_pf)

    # Compute npart normalisation from training data and append to conditioning
    npart_mean = float(tr_npart.mean())
    npart_std  = float(tr_npart.std()) if tr_npart.std() > 0 else 1.0
    if hvd.rank() == 0:
        with open(os.path.join(run_dir, 'npart_stats.json'), 'w') as f:
            json.dump({'npart_mean': npart_mean, 'npart_std': npart_std}, f)
    tr_cond = np.concatenate(
        [tr_cond, ((tr_npart - npart_mean) / npart_std)[:, None]], axis=1)
    del tr_npart

    if hvd.rank() == 0:
        print("Loading validation data ...", flush=True)

    vl_pf, vl_mask, vl_cond, vl_jet, vl_w, vl_npart, _, _, _, _ = load_shard(
        data_dir=flags.data_dir, stats_path=STATS_PATH, processes=flags.processes,
        hvd_rank=hvd.rank(), hvd_size=hvd.size(), val_start=flags.val_start,
        n_events_per_proc=flags.n_val, num_part=flags.num_part, split='val')
    vl_cond = np.concatenate(
        [vl_cond, ((vl_npart - npart_mean) / npart_std)[:, None]], axis=1)
    del vl_npart

    steps_per_epoch = n_local_train // per_worker_batch

    if hvd.rank() == 0:
        print(f"Per-GPU batch : {per_worker_batch}  |  Global batch : {global_batch}", flush=True)
        print(f"Workers: {hvd.size()}  |  Local train: {n_local_train:,}  |  "
              f"Steps/epoch: {steps_per_epoch}", flush=True)
        print(f"LR head={lr_head:.2e}  body={lr_body:.2e}  resume={resuming}", flush=True)

    train_ds = build_tf_dataset(tr_pf, tr_mask, tr_cond, tr_jet, tr_w, per_worker_batch, repeat=True)
    val_ds   = build_tf_dataset(vl_pf, vl_mask, vl_cond, vl_jet, vl_w, per_worker_batch, repeat=False)
    del tr_pf, tr_mask, vl_pf, vl_mask, tr_w, vl_w
    gc.collect()

    model = WeightedPET_pp_parton(
        num_feat=NUM_FEAT,
        num_jet=NUM_JET,
        num_cond=NUM_COND,
        num_partons=NUM_PARTONS,
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
        hist_path = os.path.join(CKPT_BASE, 'histories', f'{flags.run_name}.pkl')
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
