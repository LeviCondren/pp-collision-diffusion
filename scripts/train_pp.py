"""Train PET_pp for full-event pp generation.

Usage (single GPU, resuming automatically from checkpoint):
  python train_pp.py --run_name pet_pp_v1

Checkpoints saved to: DATA_DIR/checkpoints_pet_pp/<run_name>/

Resume logic:
  If training_state.json exists for this run_name, training resumes from the
  saved epoch count and loads the best checkpoint weights.  The LR warmup is
  skipped on resume so the optimiser starts at the full target LR.

Time-limit callback:
  --time_limit_hours (default 3.5) stops model.fit() cleanly before SLURM
  kills the job, so training_state.json is always written with the correct
  epoch count.  Set to match your wall-time minus ~30 min.
"""

import os, sys, argparse, pickle, json, time as _time
import numpy as np

# ── CUDA lib preload (must happen before TF import) ───────────────────────────
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
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy()
n_gpus   = strategy.num_replicas_in_sync
print(f"GPUs visible : {[g.name for g in gpus]}", flush=True)
print(f"Replicas     : {n_gpus}", flush=True)

# ── Local imports ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from PET_pp import PET_pp
from dataloader_pp import PPDataLoader

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import schedules, Adam

# ── Paths ──────────────────────────────────────────────────────────────────────
_FULL_DATA = '/pscratch/sd/l/lcondren/MCsim/full_event_fpcd'
STATS_PATH = f'{_FULL_DATA}/normalisation_stats.json'
CKPT_BASE  = f'{_FULL_DATA}/checkpoints_pet_pp'


# ── Custom callbacks ───────────────────────────────────────────────────────────

class SaveProgressCallback(keras.callbacks.Callback):
    """Writes training_state.json after every epoch so resume works even if
    the job is killed mid-run."""
    def __init__(self, state_path, total_epochs):
        super().__init__()
        self.state_path  = state_path
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        epochs_done = epoch + 1  # epoch is absolute (initial_epoch + relative)
        with open(self.state_path, 'w') as f:
            json.dump({
                'epochs_done':  epochs_done,
                'total_epochs': self.total_epochs,
                'done':         epochs_done >= self.total_epochs,
                'val_loss':     float((logs or {}).get('val_loss', float('inf'))),
            }, f, indent=2)


class TimeLimitCallback(keras.callbacks.Callback):
    """Stops model.fit() cleanly when the wall-clock limit is approached,
    leaving time for the job script to finalise and exit before SLURM kills it."""
    def __init__(self, max_seconds):
        super().__init__()
        self.max_seconds = max_seconds
        self._start      = _time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed   = _time.time() - self._start
        remaining = self.max_seconds - elapsed
        print(f"  [timer] {elapsed/3600:.2f}h elapsed | {remaining/3600:.2f}h remaining",
              flush=True)
        if elapsed >= self.max_seconds:
            print(f"Time limit {self.max_seconds/3600:.1f}h reached — stopping cleanly.",
                  flush=True)
            self.model.stop_training = True


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',          default=_FULL_DATA)
    p.add_argument('--run_name',          default='pet_pp_v1')
    p.add_argument('--batch',             type=int,   default=128,
                   help='Per-GPU batch size')
    p.add_argument('--epoch',             type=int,   default=200,
                   help='Total epochs across all runs')
    p.add_argument('--lr',                type=float, default=3e-4)
    p.add_argument('--lr_body',           type=float, default=1e-4)
    p.add_argument('--num_layers',        type=int,   default=8)
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
    p.add_argument('--time_limit_hours',  type=float, default=3.5,
                   help='Stop training this many hours after script start '
                        '(leave ~30 min buffer vs SLURM wall time)')
    return p.parse_args()


def build_lr_schedule(lr, n_train, global_batch, epochs, resume=False):
    """CosineDecay with linear warmup on first run; no warmup on resume."""
    decay_steps = epochs * n_train // global_batch
    if resume:
        # Optimizer step counter resets each job — start at full LR, no warmup
        return schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=max(decay_steps, 1),
        )
    warmup_steps = 3 * n_train // global_batch
    return schedules.CosineDecay(
        initial_learning_rate=lr / 10,
        warmup_target=lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )


def main():
    flags = parse_args()
    run_dir    = os.path.join(CKPT_BASE, flags.run_name)
    ckpt_path  = os.path.join(run_dir, 'pet_pp.ckpt')   # TF checkpoint format
    state_path = os.path.join(run_dir, 'training_state.json')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(CKPT_BASE, 'histories'), exist_ok=True)

    # ── Resume state ──────────────────────────────────────────────────────────
    initial_epoch = 0
    resuming      = False
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        if state.get('done', False):
            print(f"Training already complete ({state['epochs_done']} epochs). Exiting.",
                  flush=True)
            return
        initial_epoch = state.get('epochs_done', 0)
        resuming      = initial_epoch > 0 and os.path.exists(ckpt_path)
        print(f"Resuming from epoch {initial_epoch} "
              f"(best val_loss={state.get('val_loss', float('inf')):.4f})", flush=True)

    global_batch = flags.batch * n_gpus
    lr_head      = flags.lr      * n_gpus
    lr_body      = flags.lr_body * n_gpus

    print(f"Per-GPU batch : {flags.batch}  |  Global batch : {global_batch}", flush=True)
    print(f"LR head={lr_head:.2e}  body={lr_body:.2e}  resume={resuming}", flush=True)

    print("Loading training data ...", flush=True)
    train_loader = PPDataLoader(
        data_dir=flags.data_dir,
        stats_path=STATS_PATH,
        processes=flags.processes,
        batch_size=global_batch,
        val_start=flags.val_start,
        n_events=flags.n_train,
        split='train',
        num_part=flags.num_part,
    )
    print("Loading validation data ...", flush=True)
    val_loader = PPDataLoader(
        data_dir=flags.data_dir,
        stats_path=STATS_PATH,
        processes=flags.processes,
        batch_size=global_batch,
        val_start=flags.val_start,
        n_events=flags.n_val,
        split='val',
        num_part=flags.num_part,
    )

    n_train_total = train_loader.nevts

    with strategy.scope():
        model = PET_pp(
            num_feat=train_loader.num_feat,
            num_jet=train_loader.num_jet,
            num_cond=train_loader.num_cond,
            num_part=flags.num_part,
            projection_dim=flags.proj_dim,
            local=flags.local,
            K=flags.K,
            num_layers=flags.num_layers,
            drop_probability=flags.drop_prob,
            simple=flags.simple,
            layer_scale=flags.layer_scale,
            talking_head=flags.talking_head,
            mode='generator',
            fine_tune=flags.fine_tune,
            model_name=flags.model_name,
        )

        lr_sched_body = build_lr_schedule(lr_body, n_train_total, global_batch,
                                          flags.epoch, resume=resuming)
        lr_sched_head = build_lr_schedule(lr_head, n_train_total, global_batch,
                                          flags.epoch, resume=resuming)
        optimizer_body = Adam(learning_rate=lr_sched_body, clipnorm=1.0)
        optimizer_head = Adam(learning_rate=lr_sched_head, clipnorm=1.0)
        model.compile(optimizer_body, optimizer_head)

        if resuming:
            model.load_weights(ckpt_path)
            print(f"Loaded checkpoint: {ckpt_path}", flush=True)

    max_seconds = int(flags.time_limit_hours * 3600)
    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True, save_weights_only=True,
                        monitor='val_loss'),
        EarlyStopping(patience=flags.patience, restore_best_weights=True,
                      monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', patience=flags.patience // 2,
                          factor=0.5, min_lr=1e-6),
        SaveProgressCallback(state_path, total_epochs=flags.epoch),
        TimeLimitCallback(max_seconds=max_seconds),
    ]

    remaining = flags.epoch - initial_epoch
    print(f"Training: {n_train_total} events | epochs {initial_epoch}→{flags.epoch} "
          f"({remaining} remaining) | run={flags.run_name} | "
          f"time limit={flags.time_limit_hours:.1f}h", flush=True)

    hist = model.fit(
        train_loader.make_tfdata(),
        initial_epoch=initial_epoch,
        epochs=flags.epoch,
        validation_data=val_loader.make_tfdata(),
        callbacks=callbacks,
        verbose=1,
    )

    # Append history across runs
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
