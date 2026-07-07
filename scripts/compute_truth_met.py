#!/usr/bin/env python3
"""
Compute truth MET distributions from held-out signal HDF5 files.

Saves one NPZ per mass point containing MET_magnitude and MET_phi.
Particle features layout: [eta, sin_phi, cos_phi, log_pT, pid, charge, mask]

Usage:
  python3 compute_truth_met.py \
      --signal_dir /pscratch/sd/l/lcondren/MCsim/wprime_signal \
      --out_dir    /pscratch/sd/l/lcondren/MCsim/wprime_signal/truth_met \
      --mass_points 250_250 250_300 300_250 300_300
"""
import argparse, os
import numpy as np
import h5py

HOLDOUT_POINTS = [
    (250, 250),
    (250, 300),
    (300, 250),
    (300, 300),
]


def compute_met(hdf5_path, n_events=None):
    with h5py.File(hdf5_path, 'r') as f:
        pf = f['particle_features'][:]
    if n_events is not None:
        pf = pf[:n_events]

    mask  = pf[:, :, 6].astype(np.float32)
    sp    = pf[:, :, 1]
    cp    = pf[:, :, 2]
    logpT = pf[:, :, 3]

    pT = np.exp(logpT) * mask

    MET_x   = (pT * cp).sum(axis=1)
    MET_y   = (pT * sp).sum(axis=1)
    MET_mag = np.sqrt(MET_x**2 + MET_y**2)
    MET_phi = np.arctan2(MET_y, MET_x)

    return MET_mag, MET_phi


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--signal_dir', required=True)
    pa.add_argument('--out_dir',    required=True)
    pa.add_argument('--n_events', type=int, default=None,
                    help='Max events to load (default: all)')
    args = pa.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'{"Mass point":<15}  {"N":>6}  {"MET mean":>10}  {"MET std":>10}'
          f'  {"Q25":>8}  {"Q50":>8}  {"Q75":>8}  {"Q99":>8}')
    print('-' * 85)

    for mX, mY in HOLDOUT_POINTS:
        hdf5_path = os.path.join(
            args.signal_dir,
            f'signal_mX{mX:04d}_mY{mY:04d}.hdf5')
        if not os.path.exists(hdf5_path):
            print(f'  MISSING: {hdf5_path}')
            continue

        MET_mag, MET_phi = compute_met(hdf5_path, args.n_events)

        q25, q50, q75, q99 = np.percentile(MET_mag, [25, 50, 75, 99])
        print(f'mX={mX} mY={mY}      {len(MET_mag):>6}  '
              f'{MET_mag.mean():>10.2f}  {MET_mag.std():>10.2f}  '
              f'{q25:>8.2f}  {q50:>8.2f}  {q75:>8.2f}  {q99:>8.2f}')

        out_path = os.path.join(args.out_dir, f'{mX}_{mY}_truth_met.npz')
        np.savez_compressed(out_path,
                            MET_magnitude=MET_mag,
                            MET_phi=MET_phi)
        print(f'  -> {out_path}')

        # phi uniformity check
        phi_std = MET_phi.std()
        print(f'  MET_phi std={phi_std:.4f}  (pi/sqrt(3)={np.pi/np.sqrt(3):.4f}  '
              f'uniform ref)')

    print('\nDone.')


if __name__ == '__main__':
    main()
