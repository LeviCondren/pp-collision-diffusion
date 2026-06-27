#!/usr/bin/env python3
"""
TheorySpec → Hard Parton Generator via OT-CFM.

Learns P(hard partons | theorySpec) from (parton_array, theory_vec) pairs.
SM events are assigned theorySpec = 0 ("empty BSM").
BSM events (e.g. w-prime) are assigned theorySpec = [mX/scale, mY/scale, ...].

Model: Optimal-Transport Conditional Flow Matching with an adaLN MLP
       velocity network.

Target: MAX_PARTONS × PARTON_FEAT = 6 × 6 = 36-dim parton array.
        Features: [log E, sin φ, cos φ, pz/E, pdg_norm(÷16), occupancy]
"""

try:
    import horovod.tensorflow.keras as hvd
    hvd.init()
    _HVD = True
except ImportError:
    # Single-process stub for interactive testing without Horovod
    class _HvdStub:
        def init(self): pass
        def size(self): return 1
        def rank(self): return 0
        def local_rank(self): return 0
        def broadcast(self, tensor, root_rank=0): return tensor
        class callbacks:
            class BroadcastGlobalVariablesCallback:
                def __init__(self, *a, **kw): pass
            class MetricAverageCallback:
                def __init__(self, *a, **kw): pass
        class DistributedOptimizer:
            def __new__(cls, opt, **kw): return opt
    hvd = _HvdStub()
    _HVD = False

import os, sys, re, argparse, json, time as _time, math, gc
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
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus and _HVD:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True)
elif gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

tf.random.set_seed(42 + hvd.rank())

if hvd.rank() == 0:
    mode = f"Horovod {hvd.size()} workers" if _HVD else "single-process (no Horovod)"
    print(f"{mode} | GPU count: {len(gpus)}", flush=True)

MAX_PARTONS  = 6
PARTON_FEAT  = 6
PARTON_DIM   = MAX_PARTONS * PARTON_FEAT   # 36
PDG_STEP     = 1.0 / 16.0                  # rounding unit for pdg_norm


# ── Fourier time embedding ────────────────────────────────────────────────────

def fourier_embed(t, dim=128):
    """(N, 1) float → (N, dim) sin/cos embedding."""
    half  = dim // 2
    freqs = tf.exp(
        -math.log(10000.0) * tf.cast(tf.range(half), tf.float32) / float(max(half - 1, 1))
    )
    args = t * freqs[None]        # (N, half)
    return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)   # (N, dim)


# ── Model components ──────────────────────────────────────────────────────────

class TheorySpecEncoder(keras.layers.Layer):
    """MLP: theorySpec (D_theory,) → context (context_dim,)."""
    def __init__(self, theory_dim, context_dim=256, hidden=256, **kw):
        super().__init__(**kw)
        self.net = keras.Sequential([
            layers.Dense(hidden, activation='swish'),
            layers.Dense(hidden, activation='swish'),
            layers.Dense(context_dim),
        ])

    def call(self, theory):
        return self.net(theory)


class AdaLNBlock(keras.layers.Layer):
    """Residual block with adaptive LayerNorm conditioning from context."""
    def __init__(self, hidden, context_dim, **kw):
        super().__init__(**kw)
        self.norm    = layers.LayerNormalization(epsilon=1e-6)
        self.cond    = layers.Dense(2 * hidden)
        self.linear1 = layers.Dense(hidden, activation='swish')
        self.linear2 = layers.Dense(hidden)

    def call(self, x, context):
        xn             = self.norm(x)
        scale, shift   = tf.split(self.cond(context), 2, axis=-1)
        xn             = xn * (1.0 + scale) + shift
        h              = self.linear2(self.linear1(xn))
        return x + h


class VelocityNet(keras.layers.Layer):
    """
    v_θ(x_t, t, context) → velocity (PARTON_DIM,)

    Input concat: [x_t (36), t_emb (t_emb_dim)] → Dense(hidden)
                  → N adaLN blocks with context
                  → LayerNorm → Dense(36, init=0)
    """
    def __init__(self, parton_dim, context_dim,
                 hidden=256, num_layers=4, t_emb_dim=128, **kw):
        super().__init__(**kw)
        self.t_emb_dim = t_emb_dim
        self.in_proj   = layers.Dense(hidden)
        self.blocks    = [AdaLNBlock(hidden, context_dim, name=f'adln_{i}')
                          for i in range(num_layers)]
        self.out_norm  = layers.LayerNormalization(epsilon=1e-6)
        self.out_proj  = layers.Dense(parton_dim, kernel_initializer='zeros')

    def call(self, x_t, t, context):
        t_emb = fourier_embed(t, self.t_emb_dim)       # (N, t_emb_dim)
        h     = tf.concat([x_t, t_emb], axis=-1)       # (N, 36+t_emb_dim)
        h     = self.in_proj(h)                         # (N, hidden)
        for block in self.blocks:
            h = block(h, context)
        return self.out_proj(self.out_norm(h))          # (N, 36)


class TheorySpecPartonGen(keras.Model):
    """
    OT-CFM model for P(partons | theorySpec).

    sigma_min=1e-4 (tight OT path, minimal noise).
    Training: standard CFM MSE loss on velocity residuals.
    Generation: Euler integration x_0→x_1 with num_steps steps.
    """
    sigma_min = 1e-4

    def __init__(self, theory_dim, parton_dim=PARTON_DIM,
                 context_dim=256, hidden=256, num_layers=4, t_emb_dim=128, **kw):
        super().__init__(**kw)
        self.parton_dim  = parton_dim
        self.encoder     = TheorySpecEncoder(theory_dim, context_dim, hidden)
        self.velocity    = VelocityNet(parton_dim, context_dim, hidden, num_layers, t_emb_dim)
        self.loss_tracker = keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self.loss_tracker]

    def _cfm_loss(self, x1, theory):
        N   = tf.shape(x1)[0]
        x0  = tf.random.normal((N, self.parton_dim))
        t   = tf.random.uniform((N, 1))
        eps = tf.random.normal((N, self.parton_dim))

        x_t   = (1.0 - (1.0 - self.sigma_min) * t) * x0 + t * x1 + self.sigma_min * eps
        v_gt  = x1 - (1.0 - self.sigma_min) * x0

        ctx    = self.encoder(theory)
        v_pred = self.velocity(x_t, t, ctx)
        return tf.reduce_mean(tf.square(v_pred - v_gt))

    def train_step(self, data):
        x1, theory = data
        with tf.GradientTape() as tape:
            loss = self._cfm_loss(x1, theory)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def test_step(self, data):
        x1, theory = data
        self.loss_tracker.update_state(self._cfm_loss(x1, theory))
        return {'loss': self.loss_tracker.result()}

    def generate(self, theory_vec, n_events, num_steps=50):
        """
        Euler integration from x~N(0,I) to parton array.

        theory_vec: (n_events, theory_dim) or (1, theory_dim)
        Returns: (n_events, 36) in *normalised* space.
        """
        if theory_vec.shape[0] == 1:
            theory_vec = tf.tile(theory_vec, [n_events, 1])
        x  = tf.random.normal((n_events, self.parton_dim))
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = tf.fill((n_events, 1), float(step) * dt)
            x = x + self.velocity(x, t, self.encoder(theory_vec)) * dt
        return x


# ── Callbacks ─────────────────────────────────────────────────────────────────

class SaveProgressCallback(keras.callbacks.Callback):
    def __init__(self, state_path, total_epochs):
        super().__init__()
        self.state_path   = state_path
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        with open(self.state_path, 'w') as f:
            json.dump({'epochs_done': ep, 'total_epochs': self.total_epochs,
                       'done': ep >= self.total_epochs,
                       'val_loss': float((logs or {}).get('val_loss', float('inf')))}, f, indent=2)


class TimeLimitCallback(keras.callbacks.Callback):
    def __init__(self, max_seconds):
        super().__init__()
        self.max_seconds = max_seconds
        self._start      = _time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = _time.time() - self._start
        if hvd.rank() == 0:
            print(f"  [timer] {elapsed/3600:.2f}h / {self.max_seconds/3600:.1f}h", flush=True)
        if elapsed >= self.max_seconds:
            if hvd.rank() == 0:
                print("Time limit reached — stopping.", flush=True)
            self.model.stop_training = True


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sm_dir',          default='/pscratch/sd/l/lcondren/MCsim/full_event_mixed')
    p.add_argument('--sm_processes',    nargs='+', default=['dijet', 'zjets', 'ttbar', 'wjets'])
    p.add_argument('--bsm_dir',         default=None,
                   help='Directory of BSM HDF5 files named signal_mX{M1}_mY{M2}.hdf5')
    p.add_argument('--bsm_type',        default='wprime',
                   help='BSM type keyword; currently supports "wprime" (2-param: mX, mY)')
    p.add_argument('--theory_dim',      type=int,   default=2)
    p.add_argument('--theory_ref',      nargs='+',  type=float, default=None,
                   help='Reference scale per theory dim for normalization (default: max in data)')
    p.add_argument('--max_bsm_files',   type=int,   default=None,
                   help='Cap on BSM files to load (for quick tests)')
    p.add_argument('--run_name',        default='theoryspec_parton_gen')
    p.add_argument('--ckpt_base',       default='/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints')
    p.add_argument('--batch',           type=int,   default=512)
    p.add_argument('--epoch',           type=int,   default=200)
    p.add_argument('--lr',              type=float, default=3e-4)
    p.add_argument('--context_dim',     type=int,   default=256)
    p.add_argument('--hidden',          type=int,   default=256)
    p.add_argument('--num_layers',      type=int,   default=6)
    p.add_argument('--t_emb_dim',       type=int,   default=128)
    p.add_argument('--patience',        type=int,   default=30)
    p.add_argument('--val_start',       type=int,   default=400000,
                   help='SM train/val split index (per process)')
    p.add_argument('--n_sm_per_proc',   type=int,   default=None)
    p.add_argument('--n_bsm_per_file',  type=int,   default=None)
    p.add_argument('--n_val_per_proc',  type=int,   default=10000)
    p.add_argument('--time_limit_hours', type=float, default=3.5)
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_wprime_masses(filename):
    """signal_mX0300_mY0200.hdf5 → (300.0, 200.0)"""
    m = re.search(r'mX(\d+)_mY(\d+)', filename)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def load_partons_from_hdf5(path, start, end):
    """Load parton_features[start:end] → (N, 36) float32."""
    with h5py.File(path, 'r') as f:
        n = f['parton_features'].shape[0]
        end = min(end, n)
        if end <= start:
            return np.zeros((0, PARTON_DIM), dtype=np.float32)
        raw = f['parton_features'][start:end].astype(np.float32)   # (N, 6, 6)
    # Pad to MAX_PARTONS if file has fewer slots
    if raw.shape[1] < MAX_PARTONS:
        pad  = np.zeros((len(raw), MAX_PARTONS - raw.shape[1], PARTON_FEAT), dtype=np.float32)
        raw  = np.concatenate([raw, pad], axis=1)
    return raw[:, :MAX_PARTONS, :].reshape(-1, PARTON_DIM)          # (N, 36)


def load_dataset(flags, split, hvd_rank, hvd_size):
    """
    Returns:
        partons_norm : (N, 36) normalised parton arrays
        theory_vecs  : (N, theory_dim) theorySpec vectors (un-normalised)
        parton_mean  : (36,)
        parton_std   : (36,)
    """
    all_partons, all_theory = [], []

    # ── SM processes: theorySpec = zeros ─────────────────────────────────────
    for proc in flags.sm_processes:
        path = os.path.join(flags.sm_dir, f'{proc}.hdf5')
        if not os.path.exists(path):
            if hvd_rank == 0:
                print(f"  [warn] SM file not found: {path}", flush=True)
            continue

        if split == 'train':
            start, end = 0, flags.val_start
        else:
            with h5py.File(path, 'r') as f:
                n_total = f['parton_features'].shape[0]
            start, end = flags.val_start, n_total

        if split == 'train' and flags.n_sm_per_proc is not None:
            end = min(start + flags.n_sm_per_proc, end)
        elif split == 'val':
            end = min(start + flags.n_val_per_proc, end)

        n_avail  = max(end - start, 0)
        per_rank = n_avail // hvd_size
        r0       = start + hvd_rank * per_rank
        r1       = r0 + per_rank

        pts = load_partons_from_hdf5(path, r0, r1)
        tv  = np.zeros((len(pts), flags.theory_dim), dtype=np.float32)  # empty BSM
        all_partons.append(pts)
        all_theory.append(tv)
        if hvd_rank == 0:
            print(f"  SM [{proc}] {split}: {len(pts):,} events", flush=True)

    # ── BSM processes: theorySpec from filename ───────────────────────────────
    if flags.bsm_dir:
        import glob
        bsm_files = sorted(glob.glob(os.path.join(flags.bsm_dir, 'signal_m*.hdf5')))
        if flags.max_bsm_files:
            bsm_files = bsm_files[:flags.max_bsm_files]

        for path in bsm_files:
            masses = _parse_wprime_masses(os.path.basename(path))
            if masses is None:
                continue
            mX, mY = masses
            tv_raw = np.zeros(flags.theory_dim, dtype=np.float32)
            if flags.theory_dim >= 1: tv_raw[0] = mX
            if flags.theory_dim >= 2: tv_raw[1] = mY

            with h5py.File(path, 'r') as f:
                n_total = f['parton_features'].shape[0]

            if split == 'train':
                start, end = 0, min(int(n_total * 0.8), n_total)
            else:
                start, end = int(n_total * 0.8), n_total

            if flags.n_bsm_per_file is not None:
                end = min(start + flags.n_bsm_per_file, end)

            n_avail  = max(end - start, 0)
            per_rank = n_avail // hvd_size
            r0       = start + hvd_rank * per_rank
            r1       = r0 + per_rank

            pts = load_partons_from_hdf5(path, r0, r1)
            tv  = np.tile(tv_raw, (len(pts), 1))
            all_partons.append(pts)
            all_theory.append(tv)

        if hvd_rank == 0:
            print(f"  BSM [{flags.bsm_type}] {split}: {len(bsm_files)} files loaded", flush=True)

    if not all_partons:
        raise RuntimeError("No training data loaded.")

    partons_all = np.concatenate(all_partons, axis=0)   # (N, 36)
    theory_all  = np.concatenate(all_theory,  axis=0)   # (N, theory_dim)

    # Shuffle
    rng = np.random.default_rng(42 + hvd_rank if split == 'train' else 0)
    idx = rng.permutation(len(partons_all))
    return partons_all[idx], theory_all[idx]


def compute_norm_stats(partons, theory):
    """Compute mean/std for parton array and theory vector."""
    parton_mean = partons.mean(axis=0).astype(np.float32)
    parton_std  = partons.std(axis=0).clip(1e-8).astype(np.float32)
    theory_max  = np.maximum(theory.max(axis=0), 1.0).astype(np.float32)
    return parton_mean, parton_std, theory_max


def normalize(partons, theory, parton_mean, parton_std, theory_max):
    return ((partons - parton_mean) / parton_std,
            theory / theory_max)


# ── LR schedule ───────────────────────────────────────────────────────────────

def build_lr_schedule(lr, n_train, batch, epochs, resume=False):
    decay_steps  = max(epochs * n_train // batch, 1)
    warmup_steps = max(3 * n_train // batch, 1)
    if resume:
        return tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps)
    return tf.keras.optimizers.schedules.CosineDecay(
        lr / 10, warmup_target=lr,
        warmup_steps=warmup_steps, decay_steps=decay_steps)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    flags     = parse_args()
    run_dir    = os.path.join(flags.ckpt_base, flags.run_name)
    ckpt_path  = os.path.join(run_dir, 'parton_gen.weights.h5')
    state_path = os.path.join(run_dir, 'training_state.json')
    stats_path = os.path.join(run_dir, 'parton_gen_stats.json')

    if hvd.rank() == 0:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(flags.ckpt_base, 'histories'), exist_ok=True)

    # ── Resume state ─────────────────────────────────────────────────────────
    initial_epoch, done = 0, False
    if hvd.rank() == 0 and os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        done          = state.get('done', False)
        initial_epoch = state.get('epochs_done', 0)
        if not done:
            print(f"Resuming from epoch {initial_epoch} "
                  f"(val_loss={state.get('val_loss', float('inf')):.4f})", flush=True)

    done_t  = hvd.broadcast(tf.constant([1 if done  else 0], tf.int32), root_rank=0)
    epoch_t = hvd.broadcast(tf.constant([initial_epoch],     tf.int32), root_rank=0)
    done          = bool(done_t.numpy()[0])
    initial_epoch = int(epoch_t.numpy()[0])

    if done:
        if hvd.rank() == 0:
            print("Training complete. Exiting.", flush=True)
        return

    resuming = initial_epoch > 0 and os.path.exists(ckpt_path)

    # ── Load data ─────────────────────────────────────────────────────────────
    if hvd.rank() == 0:
        print("Loading training data ...", flush=True)

    tr_par, tr_thy = load_dataset(flags, 'train', hvd.rank(), hvd.size())
    vl_par, vl_thy = load_dataset(flags, 'val',   hvd.rank(), hvd.size())

    # ── Norm stats (rank 0 computes, broadcasts) ──────────────────────────────
    if not resuming or not os.path.exists(stats_path):
        pmean, pstd, tmax = compute_norm_stats(tr_par, tr_thy)
        # Override with user-supplied reference scales if provided
        if flags.theory_ref is not None:
            ref = np.array(flags.theory_ref[:flags.theory_dim], dtype=np.float32)
            tmax[:len(ref)] = ref
        if hvd.rank() == 0:
            js = {'parton_mean': pmean.tolist(), 'parton_std': pstd.tolist(),
                  'theory_max':  tmax.tolist(),  'theory_dim': flags.theory_dim,
                  'parton_dim':  PARTON_DIM}
            with open(stats_path, 'w') as f:
                json.dump(js, f, indent=2)
            print(f"Saved norm stats → {stats_path}", flush=True)
    else:
        with open(stats_path) as f:
            js = json.load(f)
        pmean = np.array(js['parton_mean'], dtype=np.float32)
        pstd  = np.array(js['parton_std'],  dtype=np.float32)
        tmax  = np.array(js['theory_max'],  dtype=np.float32)

    # Broadcast stats across workers
    pmean = hvd.broadcast(tf.constant(pmean), root_rank=0).numpy()
    pstd  = hvd.broadcast(tf.constant(pstd),  root_rank=0).numpy()
    tmax  = hvd.broadcast(tf.constant(tmax),  root_rank=0).numpy()

    tr_par_n, tr_thy_n = normalize(tr_par, tr_thy, pmean, pstd, tmax)
    vl_par_n, vl_thy_n = normalize(vl_par, vl_thy, pmean, pstd, tmax)
    del tr_par, tr_thy, vl_par, vl_thy
    gc.collect()

    n_local_train = len(tr_par_n)

    if hvd.rank() == 0:
        print(f"theory_dim={flags.theory_dim}  parton_dim={PARTON_DIM}", flush=True)
        print(f"theory_max (reference scales): {tmax.tolist()}", flush=True)
        print(f"Local train: {n_local_train:,}  Local val: {len(vl_par_n):,}", flush=True)
        print(f"Batch: {flags.batch}  Global batch: {flags.batch * hvd.size()}", flush=True)

    # ── TF datasets ───────────────────────────────────────────────────────────
    def _make_ds(par, thy, repeat=False):
        ds = (tf.data.Dataset.from_tensor_slices((
                  tf.constant(par, dtype=tf.float32),
                  tf.constant(thy, dtype=tf.float32)))
              .cache()
              .shuffle(flags.batch * 200)
              .batch(flags.batch))
        if repeat:
            ds = ds.repeat()
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = _make_ds(tr_par_n, tr_thy_n, repeat=True)
    val_ds   = _make_ds(vl_par_n, vl_thy_n)
    del tr_par_n, vl_par_n, tr_thy_n, vl_thy_n
    gc.collect()

    steps_per_epoch = n_local_train // flags.batch

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TheorySpecPartonGen(
        theory_dim  = flags.theory_dim,
        parton_dim  = PARTON_DIM,
        context_dim = flags.context_dim,
        hidden      = flags.hidden,
        num_layers  = flags.num_layers,
        t_emb_dim   = flags.t_emb_dim,
    )

    lr_sched  = build_lr_schedule(flags.lr * hvd.size(), n_local_train,
                                  flags.batch, flags.epoch, resume=resuming)
    optimizer = hvd.DistributedOptimizer(
        tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=1.0))
    model.compile(optimizer=optimizer)

    if resuming and hvd.rank() == 0:
        model.load_weights(ckpt_path)
        print(f"Loaded checkpoint: {ckpt_path}", flush=True)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    max_seconds = int(flags.time_limit_hours * 3600)
    callbacks   = [TimeLimitCallback(max_seconds=max_seconds)]
    if _HVD:
        callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                     hvd.callbacks.MetricAverageCallback()] + callbacks
    if hvd.rank() == 0:
        callbacks += [
            keras.callbacks.ModelCheckpoint(
                ckpt_path, save_best_only=True, save_weights_only=True, monitor='val_loss'),
            keras.callbacks.EarlyStopping(
                patience=flags.patience, restore_best_weights=True, monitor='val_loss'),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', patience=flags.patience // 2, factor=0.5, min_lr=1e-6),
            SaveProgressCallback(state_path, total_epochs=flags.epoch),
        ]

    if hvd.rank() == 0:
        print(f"Training epochs {initial_epoch}→{flags.epoch} | "
              f"run={flags.run_name} | time limit={flags.time_limit_hours:.1f}h", flush=True)

    model.fit(
        train_ds,
        initial_epoch      = initial_epoch,
        epochs             = flags.epoch,
        validation_data    = val_ds,
        callbacks          = callbacks,
        steps_per_epoch    = steps_per_epoch,
        verbose            = 1 if hvd.rank() == 0 else 0,
    )

    if hvd.rank() == 0:
        import pickle
        hist_path = os.path.join(flags.ckpt_base, 'histories', f'{flags.run_name}.pkl')
        with open(hist_path, 'wb') as f:
            pickle.dump({}, f)   # placeholder; history merging not needed for single run
        print(f"Done. Checkpoint: {ckpt_path}", flush=True)


if __name__ == '__main__':
    main()
