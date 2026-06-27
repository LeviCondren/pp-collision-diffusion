"""Train PET_pp_parton on the W' mass grid (per-parton cross-attention conditioning).

Loads from a directory of per-mass-point HDF5 files
  signal_mX{mX:04d}_mY{mY:04d}.hdf5
holding out files where mX in [holdout_mX_min, holdout_mX_max] AND
mY in [holdout_mY_min, holdout_mY_max] (inclusive endpoints).

Architecture is identical to per_parton_cond_train.py:
  PET_pp_parton, NUM_PARTONS=4, NUM_COND=24, NUM_FEAT=6.

Normalisation stats are computed on rank 0 from a random sample of training
files if stats_path does not already exist, then broadcast to all ranks.

Launch (multi-node via srun):
  srun python3 per_parton_cond_train_wprime.py [args]
"""

import horovod.tensorflow.keras as hvd
hvd.init()

import os, sys, re, argparse, pickle, json, time as _time, gc
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

_DEFAULT_SIGNAL_DIR = '/pscratch/sd/l/lcondren/MCsim/wprime_signal'

NUM_FEAT    = 6
NUM_JET     = 1
NUM_PARTONS = 4
PARTON_FEAT = 6
NUM_COND    = NUM_PARTONS * PARTON_FEAT  # 24

_FILE_PAT = re.compile(r'signal_mX(\d+)_mY(\d+)\.hdf5$')


class SaveProgressCallback(keras.callbacks.Callback):
    def __init__(self, state_path, total_epochs):
        super().__init__()
        self.state_path   = state_path
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        with open(self.state_path, 'w') as f:
            json.dump({
                'epochs_done':  epoch + 1,
                'total_epochs': self.total_epochs,
                'done':         (epoch + 1) >= self.total_epochs,
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
                print(f"Time limit {self.max_seconds/3600:.1f}h reached — stopping.", flush=True)
            self.model.stop_training = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--signal_dir',       default=_DEFAULT_SIGNAL_DIR)
    p.add_argument('--run_name',         default='wprimeGrid')
    p.add_argument('--holdout_mX_min',   type=int, default=300)
    p.add_argument('--holdout_mX_max',   type=int, default=350)
    p.add_argument('--holdout_mY_min',   type=int, default=300)
    p.add_argument('--holdout_mY_max',   type=int, default=350)
    p.add_argument('--n_events_per_file', type=int, default=10000,
                   help='Events to load per mass-point file (train+val combined, None=all)')
    p.add_argument('--val_frac',         type=float, default=0.1,
                   help='Fraction of n_events_per_file reserved for validation')
    p.add_argument('--batch',            type=int,   default=128)
    p.add_argument('--epoch',            type=int,   default=200)
    p.add_argument('--lr',               type=float, default=3e-4)
    p.add_argument('--lr_body',          type=float, default=1e-4)
    p.add_argument('--num_layers',       type=int,   default=8)
    p.add_argument('--num_gen_layers',   type=int,   default=2)
    p.add_argument('--proj_dim',         type=int,   default=128)
    p.add_argument('--num_part',         type=int,   default=500)
    p.add_argument('--local',            action='store_true', default=True)
    p.add_argument('--no_local',         dest='local', action='store_false')
    p.add_argument('--K',                type=int,   default=5)
    p.add_argument('--layer_scale',      action='store_true', default=True)
    p.add_argument('--simple',           action='store_true', default=False)
    p.add_argument('--talking_head',     action='store_true', default=False)
    p.add_argument('--drop_prob',        type=float, default=0.0)
    p.add_argument('--patience',         type=int,   default=30)
    p.add_argument('--fine_tune',        action='store_true', default=False)
    p.add_argument('--model_name',       default=None)
    p.add_argument('--time_limit_hours', type=float, default=3.5)
    return p.parse_args()


def list_grid_files(signal_dir, holdout_mX_min, holdout_mX_max,
                    holdout_mY_min, holdout_mY_max):
    """Return (train_files, holdout_files) lists of absolute paths."""
    train, holdout = [], []
    for fname in sorted(os.listdir(signal_dir)):
        m = _FILE_PAT.match(fname)
        if not m:
            continue
        mx, my = int(m.group(1)), int(m.group(2))
        fpath = os.path.join(signal_dir, fname)
        if (holdout_mX_min <= mx <= holdout_mX_max and
                holdout_mY_min <= my <= holdout_mY_max):
            holdout.append((mx, my, fpath))
        else:
            train.append((mx, my, fpath))
    return train, holdout


def compute_stats(train_files, num_part, num_partons, parton_feat,
                  sample_per_file=500):
    """Compute normalisation stats from a sample of training files."""
    all_pf6  = []
    all_npart = []
    all_cond = []

    for _mx, _my, fpath in train_files:
        with h5py.File(fpath, 'r') as f:
            n = min(sample_per_file, f['particle_features'].shape[0])
            pf  = f['particle_features'][:n, :num_part, :].astype(np.float32)
            pt  = f['parton_features'][:n, :num_partons, :parton_feat].astype(np.float32)

        mask = pf[:, :, 6]
        pf6  = pf[:, :, :6]
        valid = mask.astype(bool)
        all_pf6.append(pf6[valid])
        all_npart.append(np.log(np.maximum(mask.sum(axis=1), 1.0)))
        all_cond.append(pt.reshape(n, num_partons * parton_feat))

    pf6_all   = np.concatenate(all_pf6,   axis=0)
    npart_all = np.concatenate(all_npart, axis=0)
    cond_all  = np.concatenate(all_cond,  axis=0)

    part_std = pf6_all.std(axis=0)
    cond_std = cond_all.std(axis=0)

    return {
        'part_mean': pf6_all.mean(axis=0).tolist(),
        'part_std':  np.where(part_std > 0, part_std, 1.0).tolist(),
        'jet_mean':  [float(npart_all.mean())],
        'jet_std':   [max(float(npart_all.std()), 1e-6)],
        'cond_mean': cond_all.mean(axis=0).tolist(),
        'cond_std':  np.where(cond_std > 0, cond_std, 1.0).tolist(),
    }


def load_split(train_files, stats, n_events_per_file, val_frac,
               num_part, num_partons, parton_feat, hvd_rank, hvd_size, split):
    """Load train or val events from all training files, sharded by hvd rank."""
    part_mean = np.array(stats['part_mean'], dtype=np.float32)
    part_std  = np.array(stats['part_std'],  dtype=np.float32)
    jet_mean  = float(stats['jet_mean'][0])
    jet_std   = float(stats['jet_std'][0])
    cond_mean = np.array(stats['cond_mean'], dtype=np.float32)
    cond_std  = np.array(stats['cond_std'],  dtype=np.float32)

    n_val_per_file   = max(1, int(n_events_per_file * val_frac))
    n_train_per_file = n_events_per_file - n_val_per_file

    all_pf, all_mask, all_cond_n, all_jet = [], [], [], []

    for _mx, _my, fpath in train_files:
        with h5py.File(fpath, 'r') as f:
            n_file = f['particle_features'].shape[0]

        if split == 'train':
            start_g, end_g = 0, min(n_train_per_file, n_file)
        else:
            start_g = min(n_train_per_file, n_file)
            end_g   = min(n_train_per_file + n_val_per_file, n_file)

        n_proc   = end_g - start_g
        per_rank = n_proc // hvd_size
        r0 = start_g + hvd_rank * per_rank
        r1 = start_g + (hvd_rank + 1) * per_rank
        if r0 >= r1:
            continue

        with h5py.File(fpath, 'r') as f:
            pf  = f['particle_features'][r0:r1, :num_part, :].astype(np.float32)
            pt  = f['parton_features'][r0:r1, :num_partons, :parton_feat].astype(np.float32)

        mask      = pf[:, :, 6].astype(np.float32)
        pf6       = pf[:, :, :6]
        npart     = mask.sum(axis=1, keepdims=True).astype(np.float32)
        log_npart = np.log(np.maximum(npart, 1.0))
        jet       = (log_npart - jet_mean) / jet_std
        cond_raw  = pt.reshape(pt.shape[0], num_partons * parton_feat)
        cond_n    = (cond_raw - cond_mean) / cond_std
        pf6_norm  = (pf6 - part_mean) / part_std * mask[:, :, None]

        all_pf.append(pf6_norm)
        all_mask.append(mask)
        all_cond_n.append(cond_n)
        all_jet.append(jet)
        del pf, pt

    pf_all   = np.concatenate(all_pf,     axis=0)
    mask_all = np.concatenate(all_mask,   axis=0)
    cond_all = np.concatenate(all_cond_n, axis=0)
    jet_all  = np.concatenate(all_jet,    axis=0)

    seed = 42 + hvd_rank if split == 'train' else 0
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(pf_all))

    return pf_all[idx], mask_all[idx], cond_all[idx], jet_all[idx]


def build_tf_dataset(pf, mask, cond, jet, batch_size, repeat=False):
    tf_x = tf.data.Dataset.from_tensor_slices({
        'input_features': pf,
        'input_points':   pf[:, :, :2],
        'input_mask':     mask,
        'input_jet':      jet,
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
    flags = parse_args()

    train_files, holdout_files = list_grid_files(
        flags.signal_dir,
        flags.holdout_mX_min, flags.holdout_mX_max,
        flags.holdout_mY_min, flags.holdout_mY_max,
    )

    ckpt_base  = os.path.join(flags.signal_dir, 'checkpoints')
    run_dir    = os.path.join(ckpt_base, flags.run_name)
    ckpt_path  = os.path.join(run_dir, 'pet_pp.weights.h5')
    state_path = os.path.join(run_dir, 'training_state.json')
    stats_path = os.path.join(run_dir, 'normalisation_stats.json')

    if hvd.rank() == 0:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(ckpt_base, 'histories'), exist_ok=True)
        print(f"Training files  : {len(train_files)}", flush=True)
        print(f"Holdout files   : {len(holdout_files)} "
              f"(mX∈[{flags.holdout_mX_min},{flags.holdout_mX_max}] ∩ "
              f"mY∈[{flags.holdout_mY_min},{flags.holdout_mY_max}])", flush=True)
        for mx, my, _ in holdout_files:
            print(f"  holdout: mX={mx}  mY={my}", flush=True)

    # ── Resume check ──────────────────────────────────────────────────────────
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

    # ── Normalisation stats ────────────────────────────────────────────────────
    if hvd.rank() == 0:
        if os.path.exists(stats_path):
            print(f"Loading existing stats: {stats_path}", flush=True)
        else:
            print("Computing normalisation stats from training files ...", flush=True)
            stats = compute_stats(train_files, flags.num_part, NUM_PARTONS, PARTON_FEAT,
                                  sample_per_file=500)
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Stats saved: {stats_path}", flush=True)

    # Barrier: wait for rank 0 to finish writing stats before other ranks read
    hvd.broadcast(tf.constant([0], dtype=tf.int32), root_rank=0)

    with open(stats_path) as f:
        stats = json.load(f)

    # ── Load data ─────────────────────────────────────────────────────────────
    resuming = initial_epoch > 0 and os.path.exists(ckpt_path)

    per_worker_batch = flags.batch
    global_batch     = per_worker_batch * hvd.size()
    lr_head          = flags.lr      * hvd.size()
    lr_body          = flags.lr_body * hvd.size()

    if hvd.rank() == 0:
        print("Loading training data ...", flush=True)
    tr_pf, tr_mask, tr_cond, tr_jet = load_split(
        train_files, stats, flags.n_events_per_file, flags.val_frac,
        flags.num_part, NUM_PARTONS, PARTON_FEAT,
        hvd.rank(), hvd.size(), split='train')
    n_local_train = len(tr_pf)

    if hvd.rank() == 0:
        print("Loading validation data ...", flush=True)
    vl_pf, vl_mask, vl_cond, vl_jet = load_split(
        train_files, stats, flags.n_events_per_file, flags.val_frac,
        flags.num_part, NUM_PARTONS, PARTON_FEAT,
        hvd.rank(), hvd.size(), split='val')

    steps_per_epoch = n_local_train // per_worker_batch

    if hvd.rank() == 0:
        print(f"Per-GPU batch : {per_worker_batch}  |  Global batch : {global_batch}", flush=True)
        print(f"Workers: {hvd.size()}  |  Local train: {n_local_train:,}  |  "
              f"Steps/epoch: {steps_per_epoch}", flush=True)
        print(f"LR head={lr_head:.2e}  body={lr_body:.2e}  resume={resuming}", flush=True)

    train_ds = build_tf_dataset(tr_pf, tr_mask, tr_cond, tr_jet, per_worker_batch, repeat=True)
    val_ds   = build_tf_dataset(vl_pf, vl_mask, vl_cond, vl_jet, per_worker_batch, repeat=False)
    del tr_pf, tr_mask, vl_pf, vl_mask
    gc.collect()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PET_pp_parton(
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
        hist_path = os.path.join(ckpt_base, 'histories', f'{flags.run_name}.pkl')
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
