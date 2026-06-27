"""
compute_norm_stats.py — compute normalisation_stats.json for per_parton_cond_train.py

Reads a sample of events from each listed HDF5 file, computes mean/std for:
  part_mean/std  : particle features [η, sin_φ, cos_φ, log_pT, pid_cat, charge]  (6,)
  jet_mean/std   : log(n_particles per event)                                      (1,)
  cond_mean/std  : flattened parton features, first 4 partons × 6 = 24            (24,)

Usage:
  python compute_norm_stats.py --data_dir DIR --processes p1 p2 ... \
      --n_sample N --out DIR/normalisation_stats.json
"""

import argparse, json
import numpy as np
import h5py

NUM_PARTONS = 4
PARTON_FEAT = 6
NUM_COND    = NUM_PARTONS * PARTON_FEAT  # 24


def compute_stats(data_dir, processes, n_sample, out_path):
    part_rows  = []   # (n_particles_total, 6)
    cond_rows  = []   # (n_events_total, 24)
    jet_rows   = []   # (n_events_total,)

    for proc in processes:
        path = f'{data_dir}/{proc}.hdf5'
        print(f'  {proc}: {path}', flush=True)
        with h5py.File(path, 'r') as f:
            n_total = f['particle_features'].shape[0]
            n       = min(n_sample, n_total)
            rng     = np.random.default_rng(0)
            idx     = np.sort(rng.choice(n_total, n, replace=False))

            pf   = f['particle_features'][idx].astype(np.float32)   # (n, 500, 7)
            part = f['parton_features'][idx].astype(np.float32)      # (n, ?, 6)

        mask      = pf[:, :, 6]                          # (n, 500)
        pf6       = pf[:, :, :6]                         # (n, 500, 6)
        occupied  = mask > 0.5                           # (n, 500) bool

        # particle features — only occupied slots
        part_rows.append(pf6[occupied])                  # (n_occ, 6)

        # parton conditioning — first 4 partons flattened
        cond_raw = part[:, :NUM_PARTONS, :].reshape(n, NUM_COND)
        cond_rows.append(cond_raw)

        # jet feature: log(n_particles)
        npart = mask.sum(axis=1)                         # (n,)
        jet_rows.append(np.log(np.maximum(npart, 1.0)))

        del pf, part
        print(f'    sampled {n:,} events', flush=True)

    part_all = np.concatenate(part_rows, axis=0)
    cond_all = np.concatenate(cond_rows, axis=0)
    jet_all  = np.concatenate(jet_rows,  axis=0)

    stats = {
        'part_mean': part_all.mean(axis=0).tolist(),
        'part_std':  part_all.std(axis=0).tolist(),
        'jet_mean':  [float(jet_all.mean())],
        'jet_std':   [float(jet_all.std())],
        'cond_mean': cond_all.mean(axis=0).tolist(),
        'cond_std':  cond_all.std(axis=0).tolist(),
    }

    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\nWrote {out_path}')
    print(f'  part shape : {part_all.shape}')
    print(f'  cond shape : {cond_all.shape}')
    print(f'  jet shape  : {jet_all.shape}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',  required=True)
    p.add_argument('--processes', nargs='+', required=True)
    p.add_argument('--n_sample',  type=int, default=50000,
                   help='Events per process to use for stats (default 50k)')
    p.add_argument('--out',       required=True)
    args = p.parse_args()

    print(f'Computing normalisation stats over: {args.processes}')
    print(f'  data_dir : {args.data_dir}')
    print(f'  n_sample : {args.n_sample} per process')
    compute_stats(args.data_dir, args.processes, args.n_sample, args.out)


if __name__ == '__main__':
    main()
