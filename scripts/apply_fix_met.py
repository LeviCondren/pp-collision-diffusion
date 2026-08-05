#!/usr/bin/env python3
"""
Post-process A024 inference NPZ files to fix MET closure.

Loads each bsm_*.npz from INFER_DIR, applies the uniform-redistribution
MET correction (same logic as --fix_met in infer_bsm_grid_event_c_stage1.py),
and writes corrected NPZs to OUT_DIR.  All other keys are copied unchanged.

Usage:
    python apply_fix_met.py \
        --stats  /pub/lcondren/.../normalisation_stats_event_c_stage1.json \
        --infer_dir /pub/lcondren/.../infer_holdout_e2e_hpc3 \
        --out_dir   /pub/lcondren/.../infer_holdout_e2e_fixmet
"""

import argparse, glob, json, os
import numpy as np

def fix_met(parts_phys, mask_gen, jets_gen, ev_mean, ev_std):
    """Correct particle momenta so MET matches stage-1 prediction."""
    parts_out = parts_phys.copy()
    mask = mask_gen.astype(bool)

    # Decode stage-1 predicted MET from normalised jets_gen
    log1p_met_pred = jets_gen[:, 1] * float(ev_std[0]) + float(ev_mean[0])
    met_mag_pred   = np.expm1(np.clip(log1p_met_pred, -10, 20))
    sp_pred = jets_gen[:, 2] * float(ev_std[1]) + float(ev_mean[1])
    cp_pred = jets_gen[:, 3] * float(ev_std[2]) + float(ev_mean[2])
    rn = np.sqrt(sp_pred**2 + cp_pred**2 + 1e-8)
    sp_pred /= rn;  cp_pred /= rn
    met_x_tgt = met_mag_pred * cp_pred
    met_y_tgt = met_mag_pred * sp_pred

    # Current particle-level MET
    pT = np.exp(np.clip(parts_phys[:, :, 3], -10, 10)) * mask
    met_x_cur = (pT * parts_phys[:, :, 2]).sum(1)   # cos_phi column
    met_y_cur = (pT * parts_phys[:, :, 1]).sum(1)   # sin_phi column

    # Uniform momentum correction across all valid particles
    n_valid = mask.sum(1).clip(min=1).astype(float)
    d_px = ((met_x_tgt - met_x_cur) / n_valid)[:, None] * mask
    d_py = ((met_y_tgt - met_y_cur) / n_valid)[:, None] * mask

    px_new = pT * parts_phys[:, :, 2] + d_px
    py_new = pT * parts_phys[:, :, 1] + d_py
    pT_new = np.sqrt(px_new**2 + py_new**2)

    safe = (pT_new > 1e-2) & mask
    pT_safe = np.where(safe, pT_new, 1e-6)
    parts_out[:, :, 3] = np.where(safe, np.log(pT_safe),        parts_phys[:, :, 3])
    parts_out[:, :, 2] = np.where(safe, px_new / pT_safe,       parts_phys[:, :, 2])
    parts_out[:, :, 1] = np.where(safe, py_new / pT_safe,       parts_phys[:, :, 1])

    n_skip = (~safe & mask).sum()
    if n_skip:
        print(f'  [fix_met] {n_skip} particles skipped (corrected pT < 10 MeV)')
    return parts_out


def met_residual(parts_phys, mask_gen, jets_gen, ev_mean, ev_std):
    mask = mask_gen.astype(bool)
    log1p_met = jets_gen[:, 1] * float(ev_std[0]) + float(ev_mean[0])
    met_mag_pred = np.expm1(np.clip(log1p_met, -10, 20))
    sp = jets_gen[:, 2] * float(ev_std[1]) + float(ev_mean[1])
    cp = jets_gen[:, 3] * float(ev_std[2]) + float(ev_mean[2])
    rn = np.sqrt(sp**2 + cp**2 + 1e-8)
    met_x_tgt = met_mag_pred * cp / rn
    met_y_tgt = met_mag_pred * sp / rn
    pT = np.exp(np.clip(parts_phys[:, :, 3], -10, 10)) * mask
    met_x_cur = (pT * parts_phys[:, :, 2]).sum(1)
    met_y_cur = (pT * parts_phys[:, :, 1]).sum(1)
    diff = np.sqrt((met_x_cur - met_x_tgt)**2 + (met_y_cur - met_y_tgt)**2)
    return float(diff.mean())


p = argparse.ArgumentParser()
p.add_argument('--stats',     default='/pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid/normalisation_stats_event_c_stage1.json')
p.add_argument('--infer_dir', default='/pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid/bsm_grid_event_c_stage1_mpi_snap_e127/infer_holdout_e2e_hpc3')
p.add_argument('--out_dir',   default='/pub/lcondren/wprime_signal_mpi/checkpoints_bsm_grid/bsm_grid_event_c_stage1_mpi_snap_e127/infer_holdout_e2e_fixmet')
args = p.parse_args()

with open(args.stats) as f:
    stats = json.load(f)

# jet_mean/jet_std indices 1,2,3 → log1p(MET), sin(phi), cos(phi)
ev_mean = np.array(stats['jet_mean'][1:4], dtype=np.float64)
ev_std  = np.array(stats['jet_std'][1:4],  dtype=np.float64)
print(f'ev_mean={ev_mean}, ev_std={ev_std}')

os.makedirs(args.out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(args.infer_dir, 'bsm_*.npz')))
assert files, f'No bsm_*.npz in {args.infer_dir}'

for fpath in files:
    name = os.path.basename(fpath)
    print(f'\n{name}')
    d = dict(np.load(fpath))

    before = met_residual(d['parts_gen'], d['mask_gen'], d['jets_gen'], ev_mean, ev_std)
    print(f'  MET residual before: {before:.4f} GeV')

    d['parts_gen'] = fix_met(d['parts_gen'], d['mask_gen'], d['jets_gen'], ev_mean, ev_std)

    after = met_residual(d['parts_gen'], d['mask_gen'], d['jets_gen'], ev_mean, ev_std)
    print(f'  MET residual after:  {after:.6f} GeV')

    out_path = os.path.join(args.out_dir, name)
    np.savez_compressed(out_path, **d)
    print(f'  Saved: {out_path}')

print('\nDone.')
