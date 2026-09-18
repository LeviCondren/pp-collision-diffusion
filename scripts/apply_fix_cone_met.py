#!/usr/bin/env python3
"""
apply_fix_cone_met.py — Post-process E035 inference NPZs to fix cone masses + MET.

Applies two sequential corrections to the generated particle cloud:

  1. Cone mass correction (new):
       For each decay cone X (parton slot 2) and Y (parton slot 3), scale the
       pT of all particles within dR < R_CONE = 1.0 of the parton direction so
       that the reconstructed cone invariant mass matches the stage-1 prediction.
       Scaling is uniform across in-cone particles:
           pT_i → λ * pT_i   where λ = m_target / m_current
       This leaves particle directions unchanged and exactly hits the target mass.
       Events where m_current = 0 (empty cone) are left unmodified.

  2. MET correction (same logic as apply_fix_met.py):
       After cone scaling changes the total pT balance, apply the uniform
       (delta_px, delta_py) correction across all valid particles so that the
       particle-cloud MET exactly matches the stage-1 predicted MET.

Order: cone X → cone Y → MET.

Usage:
    python3 apply_fix_cone_met.py \\
        --infer_dir .../infer_cfg_sweep/s1p5 \\
        --out_dir   .../infer_cfg_sweep/s1p5_fixcone \\
        --stats_path .../normalisation_stats_event_c_stage1_cfg.json

Output NPZs are identical to inputs except parts_gen is updated.
All other keys (mask_gen, parton_feat, jets_gen, etc.) are copied unchanged.
"""

import argparse, glob, json, os
import numpy as np

R_CONE = 1.0
IDX_CM_X = 4   # index in 7-dim event feature vector: log1p(cone_mass_X)
IDX_CM_Y = 6   # index in 7-dim event feature vector: log1p(cone_mass_Y)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--infer_dir',  required=True,
                   help='Input directory with bsm_*.npz files (e.g. s1p5/)')
    p.add_argument('--out_dir',    required=True,
                   help='Output directory for fixed NPZs')
    p.add_argument('--stats_path', required=True,
                   help='normalisation_stats_event_c_stage1_cfg.json')
    return p.parse_args()


args = _parse()
os.makedirs(args.out_dir, exist_ok=True)

with open(args.stats_path) as fh:
    stats = json.load(fh)

# Full 7-dim event feature normalisation
ev_mean = np.array(stats['jet_mean'][1:], dtype=np.float64)   # (7,)
ev_std  = np.array(stats['jet_std'][1:],  dtype=np.float64)   # (7,)

print(f'ev_mean = {ev_mean.round(3)}')
print(f'ev_std  = {ev_std.round(3)}')


# ── Cone mass helper ──────────────────────────────────────────────────────────

def _parton_direction(parton_feat, slot):
    """Return (eta_p, phi_p) for parton slot from (N, 4, 7) array."""
    pze   = np.clip(parton_feat[:, slot, 3], -1 + 1e-7, 1 - 1e-7)
    eta_p = 0.5 * np.log((1 + pze) / (1 - pze))           # (N,)
    phi_p = np.arctan2(parton_feat[:, slot, 1],
                       parton_feat[:, slot, 2])             # (N,)
    return eta_p, phi_p


def _cone_mask(parts, mask, eta_p, phi_p):
    """Return boolean (N, npart) — True for valid particles within dR<R_CONE."""
    eta   = parts[:, :, 0]                                 # (N, npart)
    phi   = np.arctan2(parts[:, :, 1], parts[:, :, 2])
    deta  = eta - eta_p[:, None]
    dphi  = (phi - phi_p[:, None] + np.pi) % (2 * np.pi) - np.pi
    dR    = np.sqrt(deta**2 + dphi**2)
    return (dR < R_CONE) & mask.astype(bool)               # (N, npart)


def _cone_mass(parts, in_cone):
    """Compute invariant mass of in-cone particles. Returns (N,) in GeV."""
    eta = parts[:, :, 0]
    pT  = np.exp(np.clip(parts[:, :, 3], -10, 10)) * in_cone
    E_c  = (pT * np.cosh(np.clip(eta, -8, 8))).sum(1)
    px_c = (pT * parts[:, :, 2]).sum(1)
    py_c = (pT * parts[:, :, 1]).sum(1)
    pz_c = (pT * np.sinh(np.clip(eta, -8, 8))).sum(1)
    m2   = np.maximum(E_c**2 - px_c**2 - py_c**2 - pz_c**2, 0.0)
    return np.sqrt(m2)


def fix_cone_mass(parts, mask, parton_feat, jets_gen, ev_mean, ev_std,
                  feat_idx, parton_slot, label):
    """Scale in-cone pT so reconstructed cone mass matches stage-1 prediction.

    Args:
        parts       : (N, npart, 6) float32 — physical generated particles
        mask        : (N, npart) float32 — 1 for valid particles
        parton_feat : (N, 4, 7) float32 — parton features (col 6 = mass/MASS_NORM)
        jets_gen    : (N, 8) float32 — normalised stage-1 output
        ev_mean/std : (7,) — event feature normalisation
        feat_idx    : index in 7-dim event feature vector for this cone mass
        parton_slot : 2 for X, 3 for Y
        label       : 'X' or 'Y' for logging

    Returns:
        Updated parts array (same shape, in-place safe — returns copy).
    """
    parts = parts.copy()

    # Decode target cone mass from normalised stage-1 output
    # jets_gen[:, feat_idx+1] because col 0 = log_npart
    log1p_target = (jets_gen[:, feat_idx + 1].astype(np.float64)
                    * float(ev_std[feat_idx]) + float(ev_mean[feat_idx]))
    m_target = np.expm1(np.clip(log1p_target, 0, 15))        # (N,) GeV

    # Current cone mass from particle cloud
    eta_p, phi_p = _parton_direction(parton_feat, parton_slot)
    in_cone = _cone_mask(parts, mask, eta_p, phi_p)           # (N, npart)
    m_curr  = _cone_mass(parts, in_cone)                      # (N,) GeV

    # Scale factor per event; skip events with empty or zero-mass cone
    can_fix = (m_curr > 1e-3) & (m_target > 1e-3)
    lam     = np.where(can_fix, m_target / m_curr, 1.0)       # (N,)
    # Cap extreme scale factors to avoid wildly unphysical events
    lam     = np.clip(lam, 0.1, 10.0)

    # Apply: log_pT += log(lam) for in-cone particles
    log_lam = np.log(np.maximum(lam, 1e-8))                   # (N,)
    parts[:, :, 3] += (log_lam[:, None] * in_cone.astype(np.float32))

    # Report
    m_after = _cone_mass(parts, in_cone)
    resid_before = np.abs(m_curr[can_fix]  - m_target[can_fix]).mean() if can_fix.any() else 0
    resid_after  = np.abs(m_after[can_fix] - m_target[can_fix]).mean() if can_fix.any() else 0
    n_skip = int((~can_fix).sum())
    print(f'  [cone {label}] mean |Δm| before={resid_before:.3f} GeV  '
          f'after={resid_after:.4f} GeV  '
          f'n_skip(empty/zero)={n_skip}  '
          f'mean λ={lam[can_fix].mean():.4f}' if can_fix.any() else '')
    return parts


def fix_met_out_of_cone(parts, mask, jets_gen, ev_mean, ev_std, in_cone_any):
    """MET correction applied only to out-of-cone particles.

    Distributes the (delta_px, delta_py) correction among particles that are
    outside BOTH cones, leaving in-cone particles (and thus cone masses)
    unchanged.  If all particles are in-cone for an event, falls back to the
    uniform (all-particle) correction for that event.

    Args:
        in_cone_any : (N, npart) bool — True if particle is in cone X OR cone Y
    """
    parts_out = parts.copy()
    m   = mask.astype(bool)
    ooc = m & ~in_cone_any                             # out-of-cone valid particles

    # Decode stage-1 predicted MET
    log1p_met = jets_gen[:, 1].astype(np.float64) * float(ev_std[0]) + float(ev_mean[0])
    met_mag   = np.expm1(np.clip(log1p_met, -10, 20))
    sp = jets_gen[:, 2].astype(np.float64) * float(ev_std[1]) + float(ev_mean[1])
    cp = jets_gen[:, 3].astype(np.float64) * float(ev_std[2]) + float(ev_mean[2])
    rn = np.sqrt(sp**2 + cp**2 + 1e-8)
    met_x_tgt = met_mag * cp / rn
    met_y_tgt = met_mag * sp / rn

    pT      = np.exp(np.clip(parts[:, :, 3], -10, 10)) * m
    met_x_c = (pT * parts[:, :, 2]).sum(1)
    met_y_c = (pT * parts[:, :, 1]).sum(1)

    # Use out-of-cone particles as receivers; fall back to all-valid if no OOC
    n_ooc    = ooc.sum(1).astype(float).clip(min=1)   # (N,)
    n_valid  = m.sum(1).astype(float).clip(min=1)
    has_ooc  = ooc.sum(1) > 0                          # (N,) bool

    recv = np.where(has_ooc[:, None], ooc, m)          # (N, npart)
    n_r  = np.where(has_ooc, n_ooc, n_valid)           # (N,)

    d_px = ((met_x_tgt - met_x_c) / n_r)[:, None] * recv
    d_py = ((met_y_tgt - met_y_c) / n_r)[:, None] * recv

    px_new = pT * parts[:, :, 2] + d_px
    py_new = pT * parts[:, :, 1] + d_py
    pT_new = np.sqrt(px_new**2 + py_new**2)

    safe = (pT_new > 1e-2) & recv
    pT_s = np.where(safe, pT_new, 1e-6)
    parts_out[:, :, 3] = np.where(safe, np.log(pT_s),    parts[:, :, 3])
    parts_out[:, :, 2] = np.where(safe, px_new / pT_s,   parts[:, :, 2])
    parts_out[:, :, 1] = np.where(safe, py_new / pT_s,   parts[:, :, 1])

    n_skip   = int((~safe & recv).sum())
    n_fb     = int((~has_ooc).sum())
    if n_skip: print(f'  [fix_met] {n_skip} particles skipped (pT < 10 MeV)')
    if n_fb:   print(f'  [fix_met] {n_fb} events used all-particle fallback (no OOC particles)')
    return parts_out


def met_residual(parts, mask, jets_gen, ev_mean, ev_std):
    m = mask.astype(bool)
    log1p_met = jets_gen[:, 1].astype(np.float64) * float(ev_std[0]) + float(ev_mean[0])
    met_mag   = np.expm1(np.clip(log1p_met, -10, 20))
    sp = jets_gen[:, 2].astype(np.float64) * float(ev_std[1]) + float(ev_mean[1])
    cp = jets_gen[:, 3].astype(np.float64) * float(ev_std[2]) + float(ev_mean[2])
    rn = np.sqrt(sp**2 + cp**2 + 1e-8)
    met_x_tgt = met_mag * cp / rn;  met_y_tgt = met_mag * sp / rn
    pT = np.exp(np.clip(parts[:, :, 3], -10, 10)) * m
    met_x_c = (pT * parts[:, :, 2]).sum(1)
    met_y_c = (pT * parts[:, :, 1]).sum(1)
    return float(np.sqrt((met_x_c - met_x_tgt)**2 + (met_y_c - met_y_tgt)**2).mean())


# ── Main loop ─────────────────────────────────────────────────────────────────

files = sorted(glob.glob(os.path.join(args.infer_dir, 'bsm_*.npz')))
assert files, f'No bsm_*.npz in {args.infer_dir}'

for fpath in files:
    name = os.path.basename(fpath)
    print(f'\n── {name}')
    d = dict(np.load(fpath))

    parts     = d['parts_gen'].astype(np.float64)
    mask      = d['mask_gen']
    pf        = d['parton_feat']
    jets_gen  = d['jets_gen']

    # Before: cone masses and MET residual
    eta_pX, phi_pX = _parton_direction(pf, 2)
    eta_pY, phi_pY = _parton_direction(pf, 3)
    in_X = _cone_mask(parts, mask, eta_pX, phi_pX)
    in_Y = _cone_mask(parts, mask, eta_pY, phi_pY)

    log1p_cmX_tgt = jets_gen[:, 5].astype(np.float64) * float(ev_std[4]) + float(ev_mean[4])
    log1p_cmY_tgt = jets_gen[:, 7].astype(np.float64) * float(ev_std[6]) + float(ev_mean[6])
    cm_X_tgt = np.expm1(np.clip(log1p_cmX_tgt, 0, 15))
    cm_Y_tgt = np.expm1(np.clip(log1p_cmY_tgt, 0, 15))
    cm_X_bef = _cone_mass(parts, in_X)
    cm_Y_bef = _cone_mass(parts, in_Y)
    met_bef  = met_residual(parts, mask, jets_gen, ev_mean, ev_std)

    print(f'  BEFORE: cone_mass_X |Δ|={np.abs(cm_X_bef - cm_X_tgt).mean():.3f} GeV  '
          f'cone_mass_Y |Δ|={np.abs(cm_Y_bef - cm_Y_tgt).mean():.3f} GeV  '
          f'MET residual={met_bef:.3f} GeV')

    # 1. Fix cone mass X
    parts = fix_cone_mass(parts, mask, pf, jets_gen, ev_mean, ev_std,
                          IDX_CM_X, parton_slot=2, label='X')
    # 2. Fix cone mass Y
    parts = fix_cone_mass(parts, mask, pf, jets_gen, ev_mean, ev_std,
                          IDX_CM_Y, parton_slot=3, label='Y')
    # 3. Fix global MET — applied only to out-of-cone particles to preserve cone masses.
    # Cone membership is dR-based (directions unchanged by pT scaling), so in_X/in_Y reuse is valid.
    in_cone_any = in_X | in_Y
    met_mid = met_residual(parts, mask, jets_gen, ev_mean, ev_std)
    print(f'  MET residual after cone fixes (before MET fix): {met_mid:.3f} GeV')
    parts = fix_met_out_of_cone(parts, mask, jets_gen, ev_mean, ev_std, in_cone_any)

    # After: report — use original cone masks (dR-based, unchanged by pT scaling).
    # The MET fix touches only OOC particles so in_X / in_Y particle identities are preserved.
    # Recomputing _cone_mask after the MET fix would pick up OOC particles whose directions
    # shifted under the large momentum kick, inflating the cone mass residual artificially.
    cm_X_aft = _cone_mass(parts, in_X)
    cm_Y_aft = _cone_mass(parts, in_Y)
    met_aft  = met_residual(parts, mask, jets_gen, ev_mean, ev_std)
    print(f'  AFTER:  cone_mass_X |Δ|={np.abs(cm_X_aft - cm_X_tgt).mean():.4f} GeV  '
          f'cone_mass_Y |Δ|={np.abs(cm_Y_aft - cm_Y_tgt).mean():.4f} GeV  '
          f'MET residual={met_aft:.6f} GeV  (cone masks fixed; MET fix on OOC only)')

    # Save: update parts_gen, keep everything else
    d['parts_gen'] = parts.astype(np.float32)
    out_path = os.path.join(args.out_dir, name)
    np.savez_compressed(out_path, **d)
    print(f'  Saved: {out_path}')

print('\nDone.')
