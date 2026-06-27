#!/usr/bin/env python3
"""
Compute normalisation statistics from the NLO training data.

Only uses training events (rows 0..val_start-1) after filtering to positive-weight
events, matching exactly what the model trains on.

Output: {data_dir}/normalisation_stats.json with keys:
  part_mean / part_std  — per-feature mean/std over all valid (unmasked) particles
  jet_mean  / jet_std   — mean/std of log(npart) over all training events
  cond_mean / cond_std  — per-dim mean/std of flattened parton features (24-dim)

Usage:
  conda run -n mg5_new python3 compute_nlo_stats.py \
      --data_dir /pscratch/sd/l/lcondren/MCsim/full_event_fpcd_nlo
"""

import argparse, json, os
import numpy as np
import h5py


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='/pscratch/sd/l/lcondren/MCsim/full_event_fpcd_nlo')
    p.add_argument('--val_start',  type=int, default=400000)
    p.add_argument('--chunk_size', type=int, default=50000)
    p.add_argument('--processes',  nargs='+', default=['dijet', 'zjets'])
    return p.parse_args()


def main():
    args = parse_args()

    NUM_PART_FEAT = 6
    NUM_COND      = 24

    sum_part    = np.zeros(NUM_PART_FEAT, dtype=np.float64)
    sum_sq_part = np.zeros(NUM_PART_FEAT, dtype=np.float64)
    count_part  = 0

    sum_jet    = 0.0
    sum_sq_jet = 0.0
    count_jet  = 0

    sum_cond    = np.zeros(NUM_COND, dtype=np.float64)
    sum_sq_cond = np.zeros(NUM_COND, dtype=np.float64)
    count_cond  = 0

    for proc in args.processes:
        path = f'{args.data_dir}/{proc}.hdf5'
        print(f'Processing {proc} ...', flush=True)

        with h5py.File(path, 'r') as f:
            n_total = f['particle_features'].shape[0]
            n_train = min(args.val_start, n_total)

            n_neg_total    = 0
            n_events_total = 0

            for start in range(0, n_train, args.chunk_size):
                end  = min(start + args.chunk_size, n_train)
                pf   = f['particle_features'][start:end].astype(np.float32)  # (N, 500, 7)
                part = f['parton_features'][start:end].astype(np.float32)    # (N, 4, 6)
                ew   = f['event_weights'][start:end].astype(np.float32)      # (N,)

                pos = ew > 0
                n_neg_total    += int((~pos).sum())
                n_events_total += len(ew)

                pf   = pf[pos]
                part = part[pos]
                N    = len(pf)
                if N == 0:
                    continue

                # --- particle features (valid particles only) ---
                mask      = pf[:, :, 6]    # (N, 500)
                pf6       = pf[:, :, :6]   # (N, 500, 6)
                valid     = mask > 0
                valid_pf6 = pf6[valid]     # (n_valid, 6)
                sum_part    += valid_pf6.sum(axis=0).astype(np.float64)
                sum_sq_part += (valid_pf6**2).sum(axis=0).astype(np.float64)
                count_part  += len(valid_pf6)

                # --- log(npart) per event ---
                npart     = mask.sum(axis=1)
                log_npart = np.log(np.maximum(npart, 1.0))
                sum_jet    += float(log_npart.sum())
                sum_sq_jet += float((log_npart**2).sum())
                count_jet  += N

                # --- flattened parton features (24-dim) ---
                cond_raw = part.reshape(N, 24)
                sum_cond    += cond_raw.sum(axis=0).astype(np.float64)
                sum_sq_cond += (cond_raw**2).sum(axis=0).astype(np.float64)
                count_cond  += N

                del pf, part

                if (start // args.chunk_size) % 4 == 0:
                    print(f'  {proc} {end}/{n_train} events processed', flush=True)

        print(f'  {proc}: {n_events_total} total, '
              f'dropped {n_neg_total} neg-weight ({100*n_neg_total/n_events_total:.1f}%)',
              flush=True)

    # --- finalise stats ---
    part_mean = sum_part / count_part
    part_std  = np.sqrt(np.maximum(sum_sq_part / count_part - part_mean**2, 0.0))

    jet_mean = sum_jet / count_jet
    jet_std  = float(np.sqrt(max(sum_sq_jet / count_jet - jet_mean**2, 0.0)))

    cond_mean = sum_cond / count_cond
    cond_std  = np.sqrt(np.maximum(sum_sq_cond / count_cond - cond_mean**2, 0.0))

    stats = {
        'part_mean': part_mean.tolist(),
        'part_std':  part_std.tolist(),
        'jet_mean':  [float(jet_mean)],
        'jet_std':   [float(jet_std)],
        'cond_mean': cond_mean.tolist(),
        'cond_std':  cond_std.tolist(),
    }

    out_path = os.path.join(args.data_dir, 'normalisation_stats.json')
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'\nSaved → {out_path}')
    print(f'  part_mean : {np.array(stats["part_mean"]).round(4).tolist()}')
    print(f'  part_std  : {np.array(stats["part_std"]).round(4).tolist()}')
    print(f'  jet_mean  : {stats["jet_mean"]}  (npart ≈ {np.exp(jet_mean):.1f})')
    print(f'  jet_std   : {stats["jet_std"]}')
    print(f'  cond_mean : {np.array(stats["cond_mean"]).round(4).tolist()}')
    print(f'  cond_std  : {np.array(stats["cond_std"]).round(4).tolist()}')
    print(f'\n  Total valid particles : {count_part:,}')
    print(f'  Total training events : {count_jet:,}')


if __name__ == '__main__':
    main()
