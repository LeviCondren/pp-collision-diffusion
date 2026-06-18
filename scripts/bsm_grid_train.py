"""Phase 2 BSM grid training script.

Trains a DDPM on (hard-scatter parton, final state) pairs from the W'→XY→4q
mass grid, with optional QCD background.

Data directory: /pscratch/sd/l/lcondren/MCsim/wprime_signal/
  signal_mXXXXX_mYYYYY.hdf5  — 144 files on a 12×12 mass grid (all m_X, m_Y ∈ 50..600 GeV)
  background.hdf5              — QCD multijet background (mass_x=mass_y=0)

Holdout: 4 grid points are excluded from training and reserved for interpolation validation.
  HELDOUT_POINTS = {(250, 250), (250, 300), (300, 250), (300, 300)}  — 2×2 block near grid centre.
  Files exist at disk and are never touched during training; evaluate with infer_bsm_grid.py.

Conditioning vector (32-dim):
  [0:28]   normalised parton features (4 partons × 7 features)
            7 features per parton: log_E, sin_phi, cos_phi, pz/E, pdg_norm, occ, mass_norm
  [28:32]  binary parton mask (always [1,1,1,1] in wprime_signal data)

The 7th parton feature is mass / 600:
  slot 0  incoming quark beam-A  → 0
  slot 1  incoming quark beam-B  → 0
  slot 2  X (m=m_X)              → m_X / 600
  slot 3  Y (m=m_Y)              → m_Y / 600

Background events have mass_x=mass_y=0 → mass feature = 0 in all slots.
No process label. Signal vs background distinguished by mass alone.

Do NOT mix with Phase 1 checkpoints (different num_cond / parton_feat).
"""

import horovod.tensorflow.keras as hvd
hvd.init()

import os, sys, argparse, pickle, json, glob, time as _time, gc
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

from PET_pp_parton_vpar_bsm import WeightedBSMPET
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import schedules, Adam
from PET import FourierProjection
from layers import LayerScale, StochasticDepth

_GRID_DIR_DEFAULT = '/pscratch/sd/l/lcondren/MCsim/wprime_signal'

NUM_FEAT    = 6
NUM_JET     = 1
MAX_PARTONS = 4
PARTON_FEAT = 7   # 6 kinematic + mass
MASS_NORM   = 600.0

# 2×2 block held out of training for interpolation validation (E008).
# Files exist on disk; evaluate post-training with infer_bsm_grid.py.
HELDOUT_POINTS = frozenset({(250, 250), (250, 300), (300, 250), (300, 300)})


def _is_heldout(fpath):
    """Return True if this signal file's mass point is in HELDOUT_POINTS."""
    import re
    m = re.search(r'signal_mX(\d+)_mY(\d+)', os.path.basename(fpath))
    if not m:
        return False
    return (int(m.group(1)), int(m.group(2))) in HELDOUT_POINTS


# ── Stats computation ─────────────────────────────────────────────────────────

def compute_stats(grid_dir, include_background, n_signal_files, num_part, val_start):
    """Compute normalisation stats over training events (rows 0..val_start per file).

    Returns a dict matching the normalisation_stats.json schema:
      part_mean / part_std : (6,)  — particle feature stats (mask bit excluded)
      jet_mean  / jet_std  : [v]   — log(npart) stats
      cond_mean / cond_std : (28,) — flattened parton feature stats (4 × 7)
    """
    signal_files = sorted(glob.glob(f'{grid_dir}/signal_mX*.hdf5'))
    if n_signal_files is not None:
        signal_files = signal_files[:n_signal_files]
    # Exclude holdout points from stats computation (stats must reflect training data only)
    signal_files = [f for f in signal_files
                    if not _is_heldout(f)]
    bg_files = [f'{grid_dir}/background.hdf5'] if include_background else []
    all_files = signal_files + bg_files

    if not all_files:
        raise RuntimeError(f"No HDF5 files found in {grid_dir}")

    cond_dim = MAX_PARTONS * PARTON_FEAT  # 28

    part_sum   = np.zeros(NUM_FEAT, dtype=np.float64)
    part_sq    = np.zeros(NUM_FEAT, dtype=np.float64)
    part_count = 0

    cond_sum   = np.zeros(cond_dim, dtype=np.float64)
    cond_sq    = np.zeros(cond_dim, dtype=np.float64)
    cond_count = 0

    jet_sum   = 0.0
    jet_sq    = 0.0
    jet_count = 0

    for fpath in all_files:
        with h5py.File(fpath, 'r') as f:
            n_total = f['particle_features'].shape[0]
            e = min(val_start, n_total)
            if e == 0:
                continue

            mass_x = float(f.attrs.get('mass_x', 0.0))
            mass_y = float(f.attrs.get('mass_y', 0.0))

            pf_raw   = f['particle_features'][:e].astype(np.float64)
            part_raw = f['parton_features'][:e].astype(np.float64)

        N = len(part_raw)

        # Build 7-feature parton vector
        mass_col = np.zeros((N, MAX_PARTONS, 1), dtype=np.float64)
        mass_col[:, 2, 0] = mass_x / MASS_NORM
        mass_col[:, 3, 0] = mass_y / MASS_NORM
        part7    = np.concatenate([part_raw[:, :MAX_PARTONS, :], mass_col], axis=2)
        cond_raw = part7.reshape(N, cond_dim)

        cond_sum   += cond_raw.sum(axis=0)
        cond_sq    += (cond_raw ** 2).sum(axis=0)
        cond_count += N

        # Particle features — valid particles only
        mask       = pf_raw[:, :num_part, 6].astype(bool)
        pf6        = pf_raw[:, :num_part, :6]
        pf_valid   = pf6[mask]

        part_sum   += pf_valid.sum(axis=0)
        part_sq    += (pf_valid ** 2).sum(axis=0)
        part_count += len(pf_valid)

        # log(npart)
        npart = mask.sum(axis=1).astype(np.float64)
        log_n = np.log(np.maximum(npart, 1.0))
        jet_sum   += log_n.sum()
        jet_sq    += (log_n ** 2).sum()
        jet_count += N

    if part_count == 0 or cond_count == 0:
        raise RuntimeError("No valid events found for stats computation.")

    part_mean = (part_sum / part_count).astype(np.float32)
    part_var  = part_sq / part_count - (part_sum / part_count) ** 2
    part_std  = np.sqrt(np.maximum(part_var, 1e-10)).astype(np.float32)

    cond_mean = (cond_sum / cond_count).astype(np.float32)
    cond_var  = cond_sq / cond_count - (cond_sum / cond_count) ** 2
    cond_std  = np.sqrt(np.maximum(cond_var, 1e-10)).astype(np.float32)
    # Features that are always 0 (e.g. mass in beam slots) → set std=1 to avoid /0
    cond_std  = np.where(cond_std < 1e-6, np.float32(1.0), cond_std)

    jet_mean_v = float(jet_sum / jet_count)
    jet_var    = jet_sq / jet_count - jet_mean_v ** 2
    jet_std_v  = float(np.sqrt(max(jet_var, 1e-10)))

    return {
        'part_mean': part_mean.tolist(),
        'part_std':  part_std.tolist(),
        'jet_mean':  [jet_mean_v],
        'jet_std':   [jet_std_v],
        'cond_mean': cond_mean.tolist(),
        'cond_std':  cond_std.tolist(),
    }


def load_stats(stats_path):
    """Load and validate normalisation stats for Phase 2 (32-dim cond)."""
    with open(stats_path) as f:
        stats = json.load(f)

    cond_mean = np.array(stats['cond_mean'], dtype=np.float32)
    cond_std  = np.array(stats['cond_std'],  dtype=np.float32)

    expected_cond = MAX_PARTONS * PARTON_FEAT  # 28
    if len(cond_mean) != expected_cond:
        raise ValueError(
            f"FATAL: Stats file '{stats_path}' has cond_mean with {len(cond_mean)} dims, "
            f"expected {expected_cond} (= {MAX_PARTONS} partons × {PARTON_FEAT} features). "
            f"A Phase 1 stats file has 36 dims (6×6). Load the correct Phase 2 stats file.")

    part_mean = np.array(stats['part_mean'], dtype=np.float32)
    if len(part_mean) != NUM_FEAT:
        raise ValueError(
            f"FATAL: Stats file has part_mean with {len(part_mean)} dims, expected {NUM_FEAT}.")

    return stats


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bsm_shard(grid_dir, stats, hvd_rank, hvd_size,
                   val_start, include_background, n_signal_files,
                   n_events, num_part, split):
    """Load and shard BSM grid data for one Horovod rank.

    Returns:
        pf_norm, mask, cond, jet, weights,
        part_mean, part_std, jet_mean, jet_std
    """
    part_mean = np.array(stats['part_mean'], dtype=np.float32)
    part_std  = np.array(stats['part_std'],  dtype=np.float32)
    jet_mean  = float(stats['jet_mean'][0])
    jet_std   = float(stats['jet_std'][0])
    cond_mean = np.array(stats['cond_mean'], dtype=np.float32)
    cond_std  = np.array(stats['cond_std'],  dtype=np.float32)
    cond_dim  = MAX_PARTONS * PARTON_FEAT  # 28

    signal_files = sorted(glob.glob(f'{grid_dir}/signal_mX*.hdf5'))
    if n_signal_files is not None:
        signal_files = signal_files[:n_signal_files]
    held = [f for f in signal_files if _is_heldout(f)]
    signal_files = [f for f in signal_files if not _is_heldout(f)]
    bg_files = [f'{grid_dir}/background.hdf5'] if include_background else []
    all_files = signal_files + bg_files

    if not all_files:
        raise RuntimeError(f"No files to load from {grid_dir}")

    if hvd_rank == 0:
        pts = ', '.join(f'({int(mx)},{int(my)})' for mx, my in sorted(HELDOUT_POINTS))
        print(f"  [holdout] excluding {len(held)} files: {pts}", flush=True)

    # ── First pass: collect file slices without reading particle data ────────────
    # Avoids 2× memory spike from np.concatenate(list_of_arrays).
    file_slices = []  # (fpath, r0, r1, mass_x, mass_y)
    for fpath in all_files:
        with h5py.File(fpath, 'r') as f:
            n_total = f['particle_features'].shape[0]

            if split == 'train':
                s, e = 0, min(val_start, n_total)
            else:
                s, e = val_start, n_total

            if n_events is not None:
                e = min(s + n_events, e)

            if s >= e:
                continue

            n_file = e - s
            per_rank = n_file // hvd_size
            r0 = s + hvd_rank * per_rank
            r1 = s + (hvd_rank + 1) * per_rank
            if r0 >= r1:
                continue

            mass_x = float(f.attrs.get('mass_x', 0.0))
            mass_y = float(f.attrs.get('mass_y', 0.0))
            file_slices.append((fpath, r0, r1, mass_x, mass_y))

    if not file_slices:
        raise RuntimeError(
            f"No events loaded for split='{split}' at rank {hvd_rank}. "
            f"Check --val_start ({val_start}) vs file sizes, and that files exist.")

    total_n = sum(r1 - r0 for _, r0, r1, _, _ in file_slices)
    num_cond = MAX_PARTONS * PARTON_FEAT + MAX_PARTONS  # 32

    # ── Pre-allocate output arrays (no list-then-concatenate) ─────────────────
    pf_all   = np.empty((total_n, num_part, NUM_FEAT), dtype=np.float32)
    mask_all = np.empty((total_n, num_part),            dtype=np.float32)
    cond_all = np.empty((total_n, num_cond),            dtype=np.float32)
    jet_all  = np.empty((total_n, 1),                   dtype=np.float32)
    w_all    = np.empty((total_n,),                     dtype=np.float32)

    # ── Second pass: read data and fill in-place ──────────────────────────────
    offset = 0
    for fpath, r0, r1, mass_x, mass_y in file_slices:
        with h5py.File(fpath, 'r') as f:
            pf_raw   = f['particle_features'][r0:r1].astype(np.float32)
            part_raw = f['parton_features'][r0:r1].astype(np.float32)
            ew       = f['event_weights'][r0:r1].astype(np.float32)

        N = r1 - r0

        # Build 7-feature parton vector with mass as 7th feature
        mass_col = np.zeros((N, MAX_PARTONS, 1), dtype=np.float32)
        mass_col[:, 2, 0] = mass_x / MASS_NORM
        mass_col[:, 3, 0] = mass_y / MASS_NORM
        part7    = np.concatenate([part_raw[:, :MAX_PARTONS, :], mass_col], axis=2)

        # Normalise conditioning
        cond_raw  = part7.reshape(N, cond_dim)
        cond_norm = (cond_raw - cond_mean) / cond_std

        # Parton mask: always all-ones for wprime_signal data
        parton_mask = np.ones((N, MAX_PARTONS), dtype=np.float32)

        # Particle features
        mask  = pf_raw[:, :num_part, 6]
        pf6   = pf_raw[:, :num_part, :6]
        npart     = mask.sum(axis=1, keepdims=True)
        log_npart = np.log(np.maximum(npart, 1.0))
        jet       = (log_npart - jet_mean) / jet_std
        pf6_norm  = (pf6 - part_mean) / part_std * mask[:, :, None]

        pf_all  [offset:offset+N] = pf6_norm
        mask_all[offset:offset+N] = mask
        cond_all[offset:offset+N] = np.concatenate([cond_norm, parton_mask], axis=1)
        jet_all [offset:offset+N] = jet
        w_all   [offset:offset+N] = ew
        offset += N

        del pf_raw, part_raw, pf6, pf6_norm, mass_col, part7, cond_raw, cond_norm, parton_mask, mask, jet, ew

    rng = np.random.default_rng(42 + hvd_rank if split == 'train' else 0)
    idx = rng.permutation(total_n)

    if hvd_rank == 0:
        print(f"  [{split}] loaded {total_n:,} events from {len(file_slices)} file(s) "
              f"(rank 0 shard)", flush=True)

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


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--grid_dir',          default=_GRID_DIR_DEFAULT,
                   help='Directory containing signal_mX*.hdf5 and background.hdf5')
    p.add_argument('--ckpt_dir',          default=None,
                   help='Checkpoint root (default: {grid_dir}/checkpoints_bsm_grid)')
    p.add_argument('--run_name',          default='bsm_grid')
    p.add_argument('--stats_path',        default=None,
                   help='Path to normalisation_stats.json '
                        '(default: {grid_dir}/normalisation_stats.json)')
    p.add_argument('--include_background', action='store_true',  default=True)
    p.add_argument('--no_background',      dest='include_background', action='store_false')
    p.add_argument('--n_signal_files',    type=int, default=None,
                   help='Limit number of signal files loaded (default: all 144)')
    p.add_argument('--val_start',         type=int, default=80000,
                   help='First event index used for validation within each file')
    p.add_argument('--n_train',           type=int, default=None,
                   help='Max training events per file (default: all up to val_start)')
    p.add_argument('--n_val',             type=int, default=10000,
                   help='Max validation events per file')
    p.add_argument('--batch',             type=int,   default=128)
    p.add_argument('--epoch',             type=int,   default=200)
    p.add_argument('--lr',                type=float, default=3e-4)
    p.add_argument('--lr_body',           type=float, default=1e-4)
    p.add_argument('--num_layers',        type=int,   default=8)
    p.add_argument('--num_gen_layers',    type=int,   default=2)
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
    p.add_argument('--fine_tune',         action='store_true', default=False)
    p.add_argument('--model_name',        default=None)
    p.add_argument('--time_limit_hours',  type=float, default=3.5)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    flags = parse_args()

    grid_dir   = flags.grid_dir
    ckpt_dir   = flags.ckpt_dir or os.path.join(grid_dir, 'checkpoints_bsm_grid')
    stats_path = flags.stats_path or os.path.join(grid_dir, 'normalisation_stats.json')

    run_dir    = os.path.join(ckpt_dir, flags.run_name)
    ckpt_path  = os.path.join(run_dir, 'pet_pp.weights.h5')
    state_path = os.path.join(run_dir, 'training_state.json')

    if hvd.rank() == 0:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(ckpt_dir, 'histories'), exist_ok=True)

    # ── Compute stats if missing (rank 0 only, all ranks wait) ────────────────
    if hvd.rank() == 0 and not os.path.exists(stats_path):
        print(f"Stats file not found. Computing from training data...", flush=True)
        stats_dict = compute_stats(
            grid_dir=grid_dir,
            include_background=flags.include_background,
            n_signal_files=flags.n_signal_files,
            num_part=flags.num_part,
            val_start=flags.val_start)
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"Stats saved to {stats_path}", flush=True)
        cond_m = np.array(stats_dict['cond_mean'])
        print(f"  cond_mean dims={len(cond_m)}  part_mean dims={len(stats_dict['part_mean'])}",
              flush=True)

    hvd.allreduce(tf.constant(0.0), name='stats_barrier')

    # ── Load and validate stats ────────────────────────────────────────────────
    stats = load_stats(stats_path)
    if hvd.rank() == 0:
        print(f"Stats loaded from {stats_path}  "
              f"(cond dims={len(stats['cond_mean'])}, part dims={len(stats['part_mean'])})",
              flush=True)

    # ── Resume state ──────────────────────────────────────────────────────────
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

    # ── Load data ─────────────────────────────────────────────────────────────
    if hvd.rank() == 0:
        print("Loading training data ...", flush=True)

    tr_pf, tr_mask, tr_cond, tr_jet, tr_w, part_mean, part_std, jet_mean, jet_std = \
        load_bsm_shard(
            grid_dir=grid_dir, stats=stats,
            hvd_rank=hvd.rank(), hvd_size=hvd.size(),
            val_start=flags.val_start,
            include_background=flags.include_background,
            n_signal_files=flags.n_signal_files,
            n_events=flags.n_train,
            num_part=flags.num_part,
            split='train')
    n_local_train = len(tr_pf)

    if hvd.rank() == 0:
        print("Loading validation data ...", flush=True)

    vl_pf, vl_mask, vl_cond, vl_jet, vl_w, _, _, _, _ = \
        load_bsm_shard(
            grid_dir=grid_dir, stats=stats,
            hvd_rank=hvd.rank(), hvd_size=hvd.size(),
            val_start=flags.val_start,
            include_background=flags.include_background,
            n_signal_files=flags.n_signal_files,
            n_events=flags.n_val,
            num_part=flags.num_part,
            split='val')

    steps_per_epoch = n_local_train // per_worker_batch

    num_cond = MAX_PARTONS * PARTON_FEAT + MAX_PARTONS  # 32
    if hvd.rank() == 0:
        n_sig = flags.n_signal_files or len(sorted(glob.glob(f'{grid_dir}/signal_mX*.hdf5')))
        print(f"grid_dir={grid_dir}  signal_files={n_sig}  "
              f"include_background={flags.include_background}", flush=True)
        print(f"max_partons={MAX_PARTONS}  parton_feat={PARTON_FEAT}  "
              f"num_cond={num_cond}  (no process label)", flush=True)
        print(f"Per-GPU batch: {per_worker_batch}  |  Global batch: {global_batch}", flush=True)
        print(f"Workers: {hvd.size()}  |  Local train: {n_local_train:,}  |  "
              f"Steps/epoch: {steps_per_epoch}", flush=True)
        print(f"LR head={lr_head:.2e}  body={lr_body:.2e}  resume={resuming}", flush=True)

    train_ds = build_tf_dataset(tr_pf, tr_mask, tr_cond, tr_jet, tr_w,
                                 per_worker_batch, repeat=True)
    del tr_pf, tr_mask, tr_cond, tr_jet, tr_w
    gc.collect()

    val_ds   = build_tf_dataset(vl_pf, vl_mask, vl_cond, vl_jet, vl_w,
                                 per_worker_batch, repeat=False)
    del vl_pf, vl_mask, vl_cond, vl_jet, vl_w
    gc.collect()

    # ── Build model ───────────────────────────────────────────────────────────
    model = WeightedBSMPET(
        num_feat=NUM_FEAT,
        num_jet=NUM_JET,
        max_partons=MAX_PARTONS,
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
        hist_path = os.path.join(ckpt_dir, 'histories', f'{flags.run_name}.pkl')
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
