#!/usr/bin/env python3
"""
MET post-processing for generated BSM diffusion events.

For each generated event:
  1. Compute current (MET_x_gen, MET_y_gen) from generated particles.
  2. Sample target MET_magnitude from the truth distribution.
  3. Sample target MET_phi uniformly (truth phi is perfectly uniform).
  4. Distribute correction uniformly across valid particles:
       delta_x = MET_x_target - MET_x_gen
       delta_y = MET_y_target - MET_y_gen
       px_new_i = px_i + delta_x / N_valid
       py_new_i = py_i + delta_y / N_valid
  5. Convert back to (logpT, sin_phi, cos_phi) and save.

Input NPZ layout (physical / denormalized space):
  parts_gen : (N, npart, 6)  [eta, sin_phi, cos_phi, log_pT, pid, charge]
  mask_gen  : (N, npart)     1 = valid particle

Output: same NPZ with '_metpp' suffix; parts_gen replaced with corrected version.

Usage:
  python3 postprocess_met.py \
      --infer_dir  /pscratch/.../bsm_grid_event_c/infer_holdout_truth \
      --met_dir    /pscratch/.../truth_met \
      --out_dir    /pscratch/.../bsm_grid_event_c/infer_holdout_truth_metpp \
      [--seed 42]
"""

import argparse, glob, os
import numpy as np

PT_MIN = 1e-4   # GeV, lower-clamp before log to avoid log(0)


def compute_met_xy(parts, mask):
    """Compute (MET_x, MET_y) from physical particle features."""
    pT = np.exp(parts[:, :, 3]) * mask
    sp = parts[:, :, 1]
    cp = parts[:, :, 2]
    MET_x = (pT * cp).sum(axis=1)
    MET_y = (pT * sp).sum(axis=1)
    return MET_x, MET_y


def postprocess(parts_gen, mask_gen, target_MET_mag, target_MET_phi):
    """Apply MET correction.

    parts_gen       : (N, npart, 6) physical features — modified in-place copy
    mask_gen        : (N, npart)    float mask
    target_MET_mag  : (N,)          sampled target magnitudes
    target_MET_phi  : (N,)          sampled target phi values

    Returns corrected parts_gen (N, npart, 6).
    """
    out = parts_gen.copy()
    N   = len(out)

    MET_x_gen, MET_y_gen = compute_met_xy(out, mask_gen)

    MET_x_tgt = target_MET_mag * np.cos(target_MET_phi)
    MET_y_tgt = target_MET_mag * np.sin(target_MET_phi)

    delta_x = MET_x_tgt - MET_x_gen   # (N,)
    delta_y = MET_y_tgt - MET_y_gen   # (N,)

    N_valid = mask_gen.sum(axis=1).clip(min=1)  # (N,) avoid /0

    # Per-particle correction in x, y
    corr_x = (delta_x / N_valid)[:, None] * mask_gen   # (N, npart)
    corr_y = (delta_y / N_valid)[:, None] * mask_gen

    pT_old = np.exp(out[:, :, 3]) * mask_gen   # (N, npart)
    sp_old = out[:, :, 1]
    cp_old = out[:, :, 2]

    px_old = pT_old * cp_old
    py_old = pT_old * sp_old

    px_new = px_old + corr_x
    py_new = py_old + corr_y

    pT_new = np.sqrt(px_new**2 + py_new**2)
    # Clamp to avoid log(0) for very soft particles that get flipped
    pT_new = np.where(mask_gen > 0, np.maximum(pT_new, PT_MIN), 0.0)

    phi_new = np.arctan2(py_new, px_new)   # (N, npart)
    sp_new  = np.sin(phi_new)
    cp_new  = np.cos(phi_new)

    # Only update valid particles; leave padding slots at zero
    valid = mask_gen.astype(bool)
    logpT_new = np.where(valid, np.log(pT_new + 1e-12), 0.0)

    out[:, :, 3] = logpT_new
    out[:, :, 1] = np.where(valid, sp_new, 0.0)
    out[:, :, 2] = np.where(valid, cp_new, 0.0)

    return out


def load_truth_met(met_dir, mX, mY):
    path = os.path.join(met_dir, f'{mX}_{mY}_truth_met.npz')
    d    = np.load(path)
    return d['MET_magnitude'], d['MET_phi']


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--infer_dir', required=True,
                    help='Input directory with bsm_mX*_rank00_of01.npz files')
    pa.add_argument('--met_dir',   required=True,
                    help='Directory with {mX}_{mY}_truth_met.npz files')
    pa.add_argument('--out_dir',   required=True,
                    help='Output directory for post-processed NPZs')
    pa.add_argument('--seed', type=int, default=42)
    args = pa.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    npz_files = sorted(glob.glob(
        os.path.join(args.infer_dir, 'bsm_mX*_rank00_of01.npz')))
    print(f'Found {len(npz_files)} mass point(s)')

    for npz_path in npz_files:
        fname  = os.path.basename(npz_path)
        parts  = fname.replace('_rank00_of01.npz', '').split('_')
        mX_str = parts[1].replace('mX', '')   # e.g. '0250'
        mY_str = parts[2].replace('mY', '')
        mX     = int(mX_str.lstrip('0') or '0')
        mY     = int(mY_str.lstrip('0') or '0')

        print(f'\nmX={mX} mY={mY}')
        d = np.load(npz_path)

        parts_gen = d['parts_gen'].astype(np.float32)
        mask_gen  = d['mask_gen'].astype(np.float32)
        N         = len(parts_gen)

        # Current MET stats
        MET_x_gen, MET_y_gen = compute_met_xy(parts_gen, mask_gen)
        MET_gen = np.sqrt(MET_x_gen**2 + MET_y_gen**2)
        print(f'  Before:  MET mean={MET_gen.mean():.2f}  '
              f'std={MET_gen.std():.2f}  '
              f'Q50={np.percentile(MET_gen, 50):.2f}')

        # Sample targets from truth distribution
        truth_mag, truth_phi = load_truth_met(args.met_dir, mX, mY)
        idx_mag  = rng.integers(0, len(truth_mag), size=N)
        tgt_mag  = truth_mag[idx_mag]
        tgt_phi  = rng.uniform(-np.pi, np.pi, size=N)

        # Apply correction
        parts_pp = postprocess(parts_gen, mask_gen, tgt_mag, tgt_phi)

        # Verify MET after
        MET_x_pp, MET_y_pp = compute_met_xy(parts_pp, mask_gen)
        MET_pp = np.sqrt(MET_x_pp**2 + MET_y_pp**2)
        print(f'  After:   MET mean={MET_pp.mean():.2f}  '
              f'std={MET_pp.std():.2f}  '
              f'Q50={np.percentile(MET_pp, 50):.2f}')
        print(f'  Target:  MET mean={tgt_mag.mean():.2f}  '
              f'std={tgt_mag.std():.2f}  '
              f'Q50={np.percentile(tgt_mag, 50):.2f}')

        # Check for clamp-triggered events (pT_min was applied)
        n_clamp = (np.exp(parts_pp[:,:,3]) < PT_MIN * 1.01).sum()
        if n_clamp > 0:
            print(f'  WARNING: {n_clamp} particle-slots hit PT_MIN clamp')

        # Save — preserve all original arrays, replace parts_gen
        out_dict = {k: d[k] for k in d.files}
        out_dict['parts_gen'] = parts_pp
        out_path = os.path.join(args.out_dir, fname.replace(
            '_rank00_of01.npz', '_metpp_rank00_of01.npz'))
        np.savez_compressed(out_path, **out_dict)
        print(f'  saved -> {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
