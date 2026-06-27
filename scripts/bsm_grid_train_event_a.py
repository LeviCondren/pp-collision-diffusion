"""Phase 2 BSM grid training — E020a: event-level MET conditioning.

Copied from bsm_grid_train.py and extended for E020a:
  - Computes truth event features (MET) from particle data during loading.
  - Event features are passed as a 3-dim input to the generator head.
  - Training bypasses stage-1 event-feature prediction; truth values always used.
  - Background events receive all-zero event features.
  - Normalization stats for event features stored in normalisation_stats_event_a.json.

Do NOT modify the original bsm_grid_train.py.
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

from PET_pp_parton_vpar_bsm_event_a import WeightedBSMPET_event_a
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import schedules, Adam
from PET import FourierProjection
from layers import LayerScale, StochasticDepth

_GRID_DIR_DEFAULT = '/pscratch/sd/l/lcondren/MCsim/wprime_signal'

NUM_FEAT        = 6
NUM_JET         = 1
MAX_PARTONS     = 4
PARTON_FEAT     = 7
MASS_NORM       = 600.0
NUM_EVENT_FEAT  = 3   # E020a: MET_mag(log), sin(MET_phi), cos(MET_phi)
R_CONE          = 1.0  # cone radius for parton-cone features

# Indices in the 3-feature vector that are sin/cos (bounded; set mean=0, std=1)
_SINCOS_INDICES = [1, 2]

HELDOUT_POINTS = frozenset({(250, 250), (250, 300), (300, 250), (300, 300)})


def _is_heldout(fpath):
    import re
    m = re.search(r'signal_mX(\d+)_mY(\d+)', os.path.basename(fpath))
    if not m:
        return False
    return (int(m.group(1)), int(m.group(2))) in HELDOUT_POINTS


# ── Event feature helpers ─────────────────────────────────────────────────────

def _compute_event_raw_all7(pf_raw, part_raw, num_part):
    """Compute 7 log-space event features from raw (un-normalised) particle data.

    Returns float32 array (N, 7):
      [log(MET_mag+1), sin(MET_phi), cos(MET_phi),
       log(cone_pT_X+1), log(cone_mass_X+1),
       log(cone_pT_Y+1), log(cone_mass_Y+1)]
    """
    N     = len(pf_raw)
    valid = pf_raw[:, :num_part, 6].astype(bool)          # (N, P)
    pT    = np.exp(np.clip(pf_raw[:, :num_part, 3], -10, 10)) * valid  # (N, P)
    sp    = pf_raw[:, :num_part, 1]   # sin_phi
    cp    = pf_raw[:, :num_part, 2]   # cos_phi
    eta   = pf_raw[:, :num_part, 0]   # eta
    phi   = np.arctan2(sp, cp)        # (N, P)

    # MET
    MET_x   = (pT * cp).sum(1)
    MET_y   = (pT * sp).sum(1)
    met_mag = np.sqrt(MET_x**2 + MET_y**2)
    met_phi = np.arctan2(MET_y, MET_x)

    feats = [np.log1p(met_mag), np.sin(met_phi), np.cos(met_phi)]

    # Parton-cone features for X (slot 2) and Y (slot 3)
    eta_clip = np.clip(eta, -8, 8)
    for slot in [2, 3]:
        pze    = np.clip(part_raw[:, slot, 3], -1 + 1e-7, 1 - 1e-7)
        eta_p  = 0.5 * np.log((1 + pze) / (1 - pze))   # (N,)
        phi_p  = np.arctan2(part_raw[:, slot, 1], part_raw[:, slot, 2])  # (N,)

        deta = eta   - eta_p[:, None]
        dphi = phi   - phi_p[:, None]
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dR   = np.sqrt(deta**2 + dphi**2)

        in_c = (dR < R_CONE) & valid
        wt   = pT * in_c                              # (N, P) — zero outside cone

        pT_cone  = wt.sum(1)
        E_c  = (wt * np.cosh(eta_clip)).sum(1)
        px_c = (wt * cp).sum(1)
        py_c = (wt * sp).sum(1)
        pz_c = (wt * np.sinh(eta_clip)).sum(1)
        m2   = np.maximum(E_c**2 - px_c**2 - py_c**2 - pz_c**2, 0.0)

        feats.append(np.log1p(pT_cone))
        feats.append(np.log1p(np.sqrt(m2)))

    return np.stack(feats, axis=1).astype(np.float32)  # (N, 7)


def _assemble_event_feat(raw7):
    """Extract variant-a features (MET only) from full 7-feature array."""
    return raw7[:, :3]  # log(MET_mag+1), sin(MET_phi), cos(MET_phi)


def _normalize_event(raw, stats):
    mean = np.array(stats['event_mean'], dtype=np.float32)
    std  = np.array(stats['event_std'],  dtype=np.float32)
    return (raw - mean) / std


# ── Event stats computation (signal data only) ────────────────────────────────

def compute_event_stats(grid_dir, num_part, val_start, n_signal_files):
    """Compute event feature normalization stats from signal training data."""
    signal_files = sorted(glob.glob(f'{grid_dir}/signal_mX*.hdf5'))
    if n_signal_files is not None:
        signal_files = signal_files[:n_signal_files]
    signal_files = [f for f in signal_files if not _is_heldout(f)]

    if not signal_files:
        raise RuntimeError(f"No signal files found in {grid_dir}")

    feat_sum = np.zeros(NUM_EVENT_FEAT, dtype=np.float64)
    feat_sq  = np.zeros(NUM_EVENT_FEAT, dtype=np.float64)
    N_total  = 0

    for fpath in signal_files:
        with h5py.File(fpath, 'r') as f:
            n      = min(val_start, f['particle_features'].shape[0])
            pf_raw = f['particle_features'][:n].astype(np.float32)
            pr_raw = f['parton_features'][:n].astype(np.float32)

        raw7 = _compute_event_raw_all7(pf_raw, pr_raw, num_part)
        feat = _assemble_event_feat(raw7)   # (N, 3)

        feat_sum += feat.sum(0).astype(np.float64)
        feat_sq  += (feat ** 2).sum(0).astype(np.float64)
        N_total  += len(feat)

    mean = (feat_sum / N_total).astype(np.float32)
    var  = feat_sq / N_total - (feat_sum / N_total) ** 2
    std  = np.sqrt(np.maximum(var, 1e-10)).astype(np.float32)

    # sin/cos features are already bounded in [-1, 1]; use identity normalization
    for i in _SINCOS_INDICES:
        mean[i] = 0.0
        std[i]  = 1.0

    return {'event_mean': mean.tolist(), 'event_std': std.tolist()}


# ── Original stats (unchanged) ────────────────────────────────────────────────

def compute_stats(grid_dir, include_background, n_signal_files, num_part, val_start):
    signal_files = sorted(glob.glob(f'{grid_dir}/signal_mX*.hdf5'))
    if n_signal_files is not None:
        signal_files = signal_files[:n_signal_files]
    signal_files = [f for f in signal_files if not _is_heldout(f)]
    bg_files = [f'{grid_dir}/background.hdf5'] if include_background else []
    all_files = signal_files + bg_files

    if not all_files:
        raise RuntimeError(f"No HDF5 files found in {grid_dir}")

    cond_dim = MAX_PARTONS * PARTON_FEAT

    part_sum = np.zeros(NUM_FEAT, dtype=np.float64)
    part_sq  = np.zeros(NUM_FEAT, dtype=np.float64)
    part_count = 0
    cond_sum = np.zeros(cond_dim, dtype=np.float64)
    cond_sq  = np.zeros(cond_dim, dtype=np.float64)
    cond_count = 0
    jet_sum = 0.0; jet_sq = 0.0; jet_count = 0

    for fpath in all_files:
        with h5py.File(fpath, 'r') as f:
            n_total = f['particle_features'].shape[0]
            e = min(val_start, n_total)
            if e == 0: continue
            mass_x = float(f.attrs.get('mass_x', 0.0))
            mass_y = float(f.attrs.get('mass_y', 0.0))
            pf_raw   = f['particle_features'][:e].astype(np.float64)
            part_raw = f['parton_features'][:e].astype(np.float64)

        N = len(part_raw)
        mass_col = np.zeros((N, MAX_PARTONS, 1), dtype=np.float64)
        mass_col[:, 2, 0] = mass_x / MASS_NORM
        mass_col[:, 3, 0] = mass_y / MASS_NORM
        part7    = np.concatenate([part_raw[:, :MAX_PARTONS, :], mass_col], axis=2)
        cond_raw = part7.reshape(N, cond_dim)

        cond_sum   += cond_raw.sum(axis=0)
        cond_sq    += (cond_raw ** 2).sum(axis=0)
        cond_count += N

        mask     = pf_raw[:, :num_part, 6].astype(bool)
        pf6      = pf_raw[:, :num_part, :6]
        pf_valid = pf6[mask]
        part_sum   += pf_valid.sum(axis=0)
        part_sq    += (pf_valid ** 2).sum(axis=0)
        part_count += len(pf_valid)

        npart = mask.sum(axis=1).astype(np.float64)
        log_n = np.log(np.maximum(npart, 1.0))
        jet_sum   += log_n.sum(); jet_sq += (log_n**2).sum(); jet_count += N

    part_mean = (part_sum / part_count).astype(np.float32)
    part_std  = np.sqrt(np.maximum(part_sq/part_count - (part_sum/part_count)**2,
                                    1e-10)).astype(np.float32)
    cond_mean = (cond_sum / cond_count).astype(np.float32)
    cond_std  = np.sqrt(np.maximum(cond_sq/cond_count - (cond_sum/cond_count)**2,
                                    1e-10)).astype(np.float32)
    cond_std  = np.where(cond_std < 1e-6, np.float32(1.0), cond_std)
    jet_mean_v = float(jet_sum / jet_count)
    jet_std_v  = float(np.sqrt(max(jet_sq/jet_count - jet_mean_v**2, 1e-10)))

    return {'part_mean': part_mean.tolist(), 'part_std': part_std.tolist(),
            'jet_mean':  [jet_mean_v],        'jet_std':  [jet_std_v],
            'cond_mean': cond_mean.tolist(),  'cond_std': cond_std.tolist()}


def load_stats(stats_path):
    with open(stats_path) as f:
        stats = json.load(f)
    cond_mean = np.array(stats['cond_mean'], dtype=np.float32)
    expected_cond = MAX_PARTONS * PARTON_FEAT
    if len(cond_mean) != expected_cond:
        raise ValueError(
            f"Stats file '{stats_path}' has {len(cond_mean)} cond dims, "
            f"expected {expected_cond}.")
    part_mean = np.array(stats['part_mean'], dtype=np.float32)
    if len(part_mean) != NUM_FEAT:
        raise ValueError(f"Stats file has {len(part_mean)} part dims, expected {NUM_FEAT}.")
    return stats


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bsm_shard(grid_dir, stats, event_stats, hvd_rank, hvd_size,
                   val_start, include_background, n_signal_files,
                   n_events, num_part, split):
    part_mean = np.array(stats['part_mean'], dtype=np.float32)
    part_std  = np.array(stats['part_std'],  dtype=np.float32)
    jet_mean  = float(stats['jet_mean'][0])
    jet_std   = float(stats['jet_std'][0])
    cond_mean = np.array(stats['cond_mean'], dtype=np.float32)
    cond_std  = np.array(stats['cond_std'],  dtype=np.float32)
    cond_dim  = MAX_PARTONS * PARTON_FEAT

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

    file_slices = []
    for fpath in all_files:
        with h5py.File(fpath, 'r') as f:
            n_total = f['particle_features'].shape[0]
            if split == 'train':
                s, e = 0, min(val_start, n_total)
            else:
                s, e = val_start, n_total
            if n_events is not None:
                e = min(s + n_events, e)
            if s >= e: continue
            n_file   = e - s
            per_rank = n_file // hvd_size
            r0 = s + hvd_rank * per_rank
            r1 = s + (hvd_rank + 1) * per_rank
            if r0 >= r1: continue
            mass_x = float(f.attrs.get('mass_x', 0.0))
            mass_y = float(f.attrs.get('mass_y', 0.0))
            file_slices.append((fpath, r0, r1, mass_x, mass_y))

    if not file_slices:
        raise RuntimeError(
            f"No events loaded for split='{split}' at rank {hvd_rank}.")

    total_n  = sum(r1 - r0 for _, r0, r1, _, _ in file_slices)
    num_cond = MAX_PARTONS * PARTON_FEAT + MAX_PARTONS

    pf_all    = np.empty((total_n, num_part, NUM_FEAT),   dtype=np.float32)
    mask_all  = np.empty((total_n, num_part),              dtype=np.float32)
    cond_all  = np.empty((total_n, num_cond),              dtype=np.float32)
    jet_all   = np.empty((total_n, 1),                     dtype=np.float32)
    w_all     = np.empty((total_n,),                       dtype=np.float32)
    event_all = np.empty((total_n, NUM_EVENT_FEAT),        dtype=np.float32)

    offset = 0
    for fpath, r0, r1, mass_x, mass_y in file_slices:
        with h5py.File(fpath, 'r') as f:
            pf_raw   = f['particle_features'][r0:r1].astype(np.float32)
            part_raw = f['parton_features'][r0:r1].astype(np.float32)
            ew       = f['event_weights'][r0:r1].astype(np.float32)

        N = r1 - r0
        is_background = (mass_x == 0.0 and mass_y == 0.0)

        # Parton conditioning
        mass_col = np.zeros((N, MAX_PARTONS, 1), dtype=np.float32)
        mass_col[:, 2, 0] = mass_x / MASS_NORM
        mass_col[:, 3, 0] = mass_y / MASS_NORM
        part7    = np.concatenate([part_raw[:, :MAX_PARTONS, :], mass_col], axis=2)
        cond_raw  = part7.reshape(N, cond_dim)
        cond_norm = (cond_raw - cond_mean) / cond_std
        parton_mask = np.ones((N, MAX_PARTONS), dtype=np.float32)

        # Particle features
        mask  = pf_raw[:, :num_part, 6]
        pf6   = pf_raw[:, :num_part, :6]
        npart     = mask.sum(axis=1, keepdims=True)
        log_npart = np.log(np.maximum(npart, 1.0))
        jet       = (log_npart - jet_mean) / jet_std
        pf6_norm  = (pf6 - part_mean) / part_std * mask[:, :, None]

        # Event features — zero for background, truth for signal
        if is_background:
            ev_norm = np.zeros((N, NUM_EVENT_FEAT), dtype=np.float32)
        else:
            raw7    = _compute_event_raw_all7(pf_raw, part_raw, num_part)
            ev_raw  = _assemble_event_feat(raw7)
            ev_norm = _normalize_event(ev_raw, event_stats)

        pf_all   [offset:offset+N] = pf6_norm
        mask_all [offset:offset+N] = mask
        cond_all [offset:offset+N] = np.concatenate([cond_norm, parton_mask], axis=1)
        jet_all  [offset:offset+N] = jet
        w_all    [offset:offset+N] = ew
        event_all[offset:offset+N] = ev_norm
        offset += N

        del pf_raw, part_raw, pf6, pf6_norm, mass_col, part7, cond_raw, cond_norm
        del parton_mask, mask, jet, ew
        if not is_background:
            del raw7, ev_raw
        del ev_norm

    rng = np.random.default_rng(42 + hvd_rank if split == 'train' else 0)
    idx = rng.permutation(total_n)

    if hvd_rank == 0:
        print(f"  [{split}] loaded {total_n:,} events (rank 0 shard)", flush=True)
        print(f"  [{split}] event_feat sample[0]: {event_all[idx[0]]}", flush=True)

    return (pf_all[idx], mask_all[idx], cond_all[idx], jet_all[idx],
            w_all[idx], event_all[idx],
            part_mean, part_std, jet_mean, jet_std)


def build_tf_dataset(pf, mask, cond, jet, weights, event_feat,
                     batch_size, repeat=False):
    tf_x = tf.data.Dataset.from_tensor_slices({
        'input_features': pf,
        'input_points':   pf[:, :, :2],
        'input_mask':     mask,
        'input_jet':      jet,
        'input_weight':   weights,
        'input_event':    event_feat,
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
        return schedules.CosineDecay(initial_learning_rate=lr,
                                     decay_steps=max(decay_steps, 1))
    warmup_steps = 3 * n_train // batch
    return schedules.CosineDecay(initial_learning_rate=lr / 10,
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
            json.dump({'epochs_done': epochs_done, 'total_epochs': self.total_epochs,
                       'done': epochs_done >= self.total_epochs,
                       'val_loss': float((logs or {}).get('val_loss', float('inf')))},
                      f, indent=2)


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
    p.add_argument('--grid_dir',           default=_GRID_DIR_DEFAULT)
    p.add_argument('--ckpt_dir',           default=None)
    p.add_argument('--run_name',           default='bsm_grid_event_a')
    p.add_argument('--stats_path',         default=None)
    p.add_argument('--stats_event_path',   default=None,
                   help='Path to normalisation_stats_event_a.json '
                        '(default: {grid_dir}/normalisation_stats_event_a.json)')
    p.add_argument('--include_background', action='store_true',  default=True)
    p.add_argument('--no_background',      dest='include_background', action='store_false')
    p.add_argument('--n_signal_files',     type=int, default=None)
    p.add_argument('--val_start',          type=int, default=80000)
    p.add_argument('--n_train',            type=int, default=None)
    p.add_argument('--n_val',              type=int, default=10000)
    p.add_argument('--batch',              type=int,   default=128)
    p.add_argument('--epoch',              type=int,   default=200)
    p.add_argument('--lr',                 type=float, default=3e-4)
    p.add_argument('--lr_body',            type=float, default=1e-4)
    p.add_argument('--num_layers',         type=int,   default=8)
    p.add_argument('--num_gen_layers',     type=int,   default=2)
    p.add_argument('--proj_dim',           type=int,   default=128)
    p.add_argument('--num_part',           type=int,   default=500)
    p.add_argument('--local',              action='store_true', default=True)
    p.add_argument('--no_local',           dest='local', action='store_false')
    p.add_argument('--K',                  type=int,   default=5)
    p.add_argument('--layer_scale',        action='store_true', default=True)
    p.add_argument('--simple',             action='store_true', default=False)
    p.add_argument('--talking_head',       action='store_true', default=False)
    p.add_argument('--drop_prob',          type=float, default=0.0)
    p.add_argument('--patience',           type=int,   default=30)
    p.add_argument('--fine_tune',          action='store_true', default=False)
    p.add_argument('--model_name',         default=None)
    p.add_argument('--time_limit_hours',   type=float, default=3.5)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    flags = parse_args()

    grid_dir        = flags.grid_dir
    ckpt_dir        = flags.ckpt_dir or os.path.join(grid_dir, 'checkpoints_bsm_grid')
    stats_path      = flags.stats_path or os.path.join(grid_dir, 'normalisation_stats.json')
    stats_event_path = (flags.stats_event_path
                        or os.path.join(grid_dir, 'normalisation_stats_event_a.json'))

    run_dir    = os.path.join(ckpt_dir, flags.run_name)
    ckpt_path  = os.path.join(run_dir, 'pet_pp.weights.h5')
    state_path = os.path.join(run_dir, 'training_state.json')

    if hvd.rank() == 0:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(ckpt_dir, 'histories'), exist_ok=True)

    # ── Compute parton conditioning stats (rank 0, shared) ────────────────────
    if hvd.rank() == 0 and not os.path.exists(stats_path):
        print("Computing parton conditioning stats ...", flush=True)
        stats_dict = compute_stats(grid_dir, flags.include_background,
                                   flags.n_signal_files, flags.num_part,
                                   flags.val_start)
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"Stats saved to {stats_path}", flush=True)

    # ── Compute event feature stats (rank 0, signal only) ─────────────────────
    if hvd.rank() == 0 and not os.path.exists(stats_event_path):
        print(f"Computing event feature stats (E020a) ...", flush=True)
        ev_stats_dict = compute_event_stats(grid_dir, flags.num_part, flags.val_start,
                                            flags.n_signal_files)
        with open(stats_event_path, 'w') as f:
            json.dump(ev_stats_dict, f, indent=2)
        print(f"Event stats saved to {stats_event_path}", flush=True)
        print(f"  event_mean: {ev_stats_dict['event_mean']}", flush=True)
        print(f"  event_std:  {ev_stats_dict['event_std']}", flush=True)

    hvd.allreduce(tf.constant(0.0), name='stats_barrier')

    stats = load_stats(stats_path)
    with open(stats_event_path) as f:
        event_stats = json.load(f)
    if hvd.rank() == 0:
        print(f"Event stats: mean={event_stats['event_mean']}  "
              f"std={event_stats['event_std']}", flush=True)

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

    (tr_pf, tr_mask, tr_cond, tr_jet, tr_w, tr_ev,
     part_mean, part_std, jet_mean, jet_std) = load_bsm_shard(
        grid_dir=grid_dir, stats=stats, event_stats=event_stats,
        hvd_rank=hvd.rank(), hvd_size=hvd.size(),
        val_start=flags.val_start,
        include_background=flags.include_background,
        n_signal_files=flags.n_signal_files,
        n_events=flags.n_train,
        num_part=flags.num_part, split='train')
    n_local_train = len(tr_pf)

    if hvd.rank() == 0:
        print("Loading validation data ...", flush=True)

    (vl_pf, vl_mask, vl_cond, vl_jet, vl_w, vl_ev,
     _, _, _, _) = load_bsm_shard(
        grid_dir=grid_dir, stats=stats, event_stats=event_stats,
        hvd_rank=hvd.rank(), hvd_size=hvd.size(),
        val_start=flags.val_start,
        include_background=flags.include_background,
        n_signal_files=flags.n_signal_files,
        n_events=flags.n_val,
        num_part=flags.num_part, split='val')

    steps_per_epoch = n_local_train // per_worker_batch

    num_cond = MAX_PARTONS * PARTON_FEAT + MAX_PARTONS
    if hvd.rank() == 0:
        print(f"num_event_feat={NUM_EVENT_FEAT}  run={flags.run_name}", flush=True)
        print(f"Per-GPU batch: {per_worker_batch}  |  Global batch: {global_batch}", flush=True)
        print(f"Workers: {hvd.size()}  |  Local train: {n_local_train:,}  |  "
              f"Steps/epoch: {steps_per_epoch}", flush=True)

    train_ds = build_tf_dataset(tr_pf, tr_mask, tr_cond, tr_jet, tr_w, tr_ev,
                                 per_worker_batch, repeat=True)
    del tr_pf, tr_mask, tr_cond, tr_jet, tr_w, tr_ev
    gc.collect()

    val_ds = build_tf_dataset(vl_pf, vl_mask, vl_cond, vl_jet, vl_w, vl_ev,
                               per_worker_batch, repeat=False)
    del vl_pf, vl_mask, vl_cond, vl_jet, vl_w, vl_ev
    gc.collect()

    # ── Build model ───────────────────────────────────────────────────────────
    model = WeightedBSMPET_event_a(
        num_feat=NUM_FEAT,
        num_jet=NUM_JET,
        max_partons=MAX_PARTONS,
        parton_feat=PARTON_FEAT,
        num_event_feat=NUM_EVENT_FEAT,
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
