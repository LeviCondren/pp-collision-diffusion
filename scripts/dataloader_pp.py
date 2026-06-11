"""Data loader for full-event pp generation (dijet + zjets).

Reads the same HDF5 files used by the GSGM pipeline:
  /pscratch/.../full_event_fpcd/dijet.hdf5
  /pscratch/.../full_event_fpcd/zjets.hdf5

Each file has:
  particle_features : (N, NPART, 7)  — [eta, sin_phi, cos_phi, log_pT, pid, charge, mask]
  parton_features   : (N, 4, 6)      — [log_E, sin_phi, cos_phi, pz/E, PDG/10, is_valid]

PET inputs:
  input_features : (N, NPART, 6)   normalised particle features (mask column stripped)
  input_points   : (N, NPART, 2)   first two features (eta, sin_phi) for local graph
  input_mask     : (N, NPART)      binary mask (particle_features[:,:,6])
  input_jet      : (N, 1)          normalised log_npart (stage-1 diffusion target)
  y (label)      : (N, 24)         normalised flattened parton conditioning
"""

import numpy as np
import h5py
import json
import tensorflow as tf
import gc


class PPDataLoader:
    def __init__(self,
                 data_dir,
                 stats_path,
                 processes=('dijet', 'zjets'),
                 batch_size=512,
                 val_start=400000,
                 n_events=None,
                 split='train',
                 num_part=500):
        """
        Args:
            data_dir: directory containing dijet.hdf5 and zjets.hdf5
            stats_path: path to normalisation_stats.json
            processes: tuple of process names to load
            batch_size: batch size for tf.data pipeline
            val_start: index where validation split begins
            n_events: events per process to use (None = all available in split)
            split: 'train' or 'val'
            num_part: max particles per event (padding size)
        """
        self.batch_size = batch_size
        self.num_part = num_part

        with open(stats_path) as f:
            stats = json.load(f)
        self.part_mean = np.array(stats['part_mean'], dtype=np.float32)
        self.part_std  = np.array(stats['part_std'],  dtype=np.float32)
        self.jet_mean  = float(stats['jet_mean'][0])
        self.jet_std   = float(stats['jet_std'][0])
        self.cond_mean = np.array(stats['cond_mean'], dtype=np.float32)
        self.cond_std  = np.array(stats['cond_std'],  dtype=np.float32)
        # avoid divide-by-zero for constant cond features
        self.cond_std  = np.where(self.cond_std > 0, self.cond_std, 1.0)

        all_pf, all_cond, all_jet = [], [], []

        for proc in processes:
            path = f'{data_dir}/{proc}.hdf5'
            with h5py.File(path, 'r') as f:
                n_total = f['particle_features'].shape[0]
                if split == 'train':
                    start, end = 0, val_start
                else:
                    start, end = val_start, n_total

                if n_events is not None:
                    end = min(start + n_events, end)

                pf   = f['particle_features'][start:end].astype(np.float32)
                part = f['parton_features'][start:end].astype(np.float32)

            # mask: particle_features[:,:,6]
            mask = pf[:, :, 6].astype(np.float32)
            # particle features: first 6 channels
            pf6  = pf[:, :, :6]

            # log_npart
            npart = mask.sum(axis=1, keepdims=True).astype(np.float32)
            log_npart = np.log(np.maximum(npart, 1))
            jet = (log_npart - self.jet_mean) / self.jet_std  # (N, 1)

            # parton conditioning: flatten (N,4,6) -> (N,24)
            cond = part.reshape(part.shape[0], 24)
            cond = (cond - self.cond_mean) / self.cond_std

            all_pf.append((pf6, mask))
            all_cond.append(cond)
            all_jet.append(jet)

        # Concatenate across processes
        pf6_all  = np.concatenate([x[0] for x in all_pf], axis=0)
        mask_all = np.concatenate([x[1] for x in all_pf], axis=0)
        cond_all = np.concatenate(all_cond, axis=0)
        jet_all  = np.concatenate(all_jet,  axis=0)

        # Shuffle once
        rng = np.random.default_rng(42 if split == 'train' else 0)
        idx = rng.permutation(len(pf6_all))
        pf6_all, mask_all, cond_all, jet_all = (
            pf6_all[idx], mask_all[idx], cond_all[idx], jet_all[idx])

        self.X      = pf6_all    # (N, NPART, 6)
        self.mask   = mask_all   # (N, NPART)
        self.y      = cond_all   # (N, 24)
        self.jet    = jet_all    # (N, 1)

        self.nevts       = len(self.X)
        self.num_feat    = self.X.shape[2]
        self.num_jet     = self.jet.shape[1]
        self.num_cond    = self.y.shape[1]
        self.steps_per_epoch = None

        print(f"[PPDataLoader] split={split} nevts={self.nevts} "
              f"num_feat={self.num_feat} num_jet={self.num_jet} num_cond={self.num_cond}")

    def preprocess(self):
        X = (self.X - self.part_mean) / self.part_std
        X = self.mask[:, :, None] * X
        return X.astype(np.float32)

    def make_tfdata(self):
        X = self.preprocess()
        tf_x = tf.data.Dataset.from_tensor_slices({
            'input_features': X,
            'input_points':   X[:, :, :2],
            'input_mask':     self.mask,
            'input_jet':      self.jet,
        })
        tf_y = tf.data.Dataset.from_tensor_slices(self.y)
        del self.X, self.mask
        gc.collect()
        return (tf.data.Dataset.zip((tf_x, tf_y))
                .cache()
                .shuffle(self.batch_size * 100)
                .batch(self.batch_size)
                .prefetch(tf.data.AUTOTUNE))

    def make_eval_data(self):
        """Returns (tf_dataset, y_cond) for inference, without shuffling."""
        X = self.preprocess()
        tf_x = tf.data.Dataset.from_tensor_slices({
            'input_features': X,
            'input_points':   X[:, :, :2],
            'input_mask':     self.mask,
            'input_jet':      self.jet,
            'input_time':     np.zeros((self.nevts, 1), dtype=np.float32),
        })
        return (tf_x.cache().batch(self.batch_size).prefetch(tf.data.AUTOTUNE), self.y)

    def revert_preprocess(self, x, mask):
        """Undo particle normalisation (for evaluation)."""
        return mask[:, :, None] * (x * self.part_std + self.part_mean)

    def revert_jet(self, jet_norm):
        """Convert normalised log_npart back to integer particle counts."""
        log_npart = jet_norm[:, 0] * self.jet_std + self.jet_mean
        return np.clip(np.round(np.exp(log_npart)).astype(int), 1, self.num_part)
