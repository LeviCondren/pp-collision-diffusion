#!/usr/bin/env python3
"""add_boson_parton.py

Append a hard-boson parton slot to each process HDF5 file and resize
parton_features to MAX_PARTONS_NEW=6 slots.

What each process gets:
  zjets  → Z (PDG 23) with kinematics reconstructed from the hardest OSSF
             lepton pair in particle_features (e+e- or mu+mu-)
  wjets  → W (PDG 24) with zero kinematics (W kinematics were filtered by
             _COLORED at generation time and are not recoverable)
  ttbar  → top/antitop PDG norms fixed (tops were in parton_features but
             mapped to 0 because PDG 6 was missing from _PDG_CLASS)
  dijet/wprime → no boson added; parton_features padded to 6 slots

Also:
  - Creates n_partons dataset in files that don't have it (dijet, zjets, wprime)
  - Rewrites normalisation_stats.json with new cond dimensions

Extended PDG norm encoding (÷16 instead of ÷10):
  class 0  → gluon (21) / unknown
  class 1  → d  (1)      class 2  → dbar (-1)
  class 3  → u  (2)      class 4  → ubar (-2)
  class 5  → s  (3)      class 6  → sbar (-3)
  class 7  → c  (4)      class 8  → cbar (-4)
  class 9  → b  (5)      class 10 → bbar (-5)
  class 11 → t  (6)      class 12 → tbar (-6)
  class 13 → Z (23)
  class 14 → W+ (24)     class 15 → W- (-24)

Usage (login node with module loaded):
    module load tensorflow/2.15.0
    python3 add_boson_parton.py [--data_dir DIR] [--dry_run]
"""

import argparse, json, shutil
import numpy as np
import h5py
from pathlib import Path

DATA_DIR        = Path('/pscratch/sd/l/lcondren/MCsim/full_event_mixed')
PROCESSES       = ['dijet', 'zjets', 'ttbar', 'wjets', 'wprime']
MAX_PARTONS_NEW = 6
N_PARTON_FEAT   = 6   # [log_E, sin_phi, cos_phi, pz/E, pdg_norm, is_valid]
CHUNK           = 10_000

# ── PDG encoding ──────────────────────────────────────────────────────────────

_N_PDG = 16.0

_PDG_CLASS = {
    21: 0,
    1: 1,   -1: 2,
    2: 3,   -2: 4,
    3: 5,   -3: 6,
    4: 7,   -4: 8,
    5: 9,   -5: 10,
    6: 11,  -6: 12,
    23: 13,
    24: 14, -24: 15,
}

def pdg_norm(pid):
    return float(_PDG_CLASS.get(pid, 0)) / _N_PDG

_PDG_NORM_Z = pdg_norm(23)   # 13/16 = 0.8125
_PDG_NORM_W = pdg_norm(24)   # 14/16 = 0.875
_PDG_NORM_T = pdg_norm(6)    # 11/16 = 0.6875
_PDG_NORM_TB= pdg_norm(-6)   # 12/16 = 0.75

# Mapping from old pdg_norm (÷10) to new (÷16) for existing quark/gluon slots
# Old class k → old norm k/10 → new norm k/16
# (for k=0..10; k=0 is gluon but also unknown, so stays 0)
_OLD_TO_NEW = {round(k / 10.0, 6): k / _N_PDG for k in range(11)}

def rescale_pdg_norm(old_val):
    """Convert old ÷10 pdg_norm to new ÷16 encoding (for quarks/gluons only)."""
    key = round(float(old_val), 6)
    return float(_OLD_TO_NEW.get(key, 0.0))


# ── Z reconstruction ──────────────────────────────────────────────────────────

def reconstruct_z_slot(particle_chunk):
    """Vectorised Z→ll reconstruction from hardest OSSF lepton pair.

    particle_chunk : (B, N_PART_MAX, 7)
        columns: [eta, sin_phi, cos_phi, log_pT, pid_cat, charge, is_valid]
        pid_cat: 3=electron, 4=muon

    Returns (B, 6) parton slot [log_E, sin_phi, cos_phi, pz/E, pdg_norm, is_valid]
    """
    B    = particle_chunk.shape[0]
    slot = np.zeros((B, N_PARTON_FEAT), dtype=np.float32)
    slot[:, 4] = _PDG_NORM_Z  # PDG pre-filled; is_valid set to 1 when found

    ptf      = particle_chunk
    is_valid = ptf[:, :, 6] > 0.5  # (B, P)
    log_pT   = ptf[:, :, 3]        # (B, P)

    for pid_cat in (3, 4):   # electrons first, then muons
        already = slot[:, 5] > 0.5
        if already.all():
            break

        flavor = (ptf[:, :, 4] == pid_cat) & is_valid
        pos    = flavor & (ptf[:, :, 5] > 0.5)
        neg    = flavor & (ptf[:, :, 5] < -0.5)

        has_pair = pos.any(axis=1) & neg.any(axis=1) & ~already
        if not has_pair.any():
            continue

        rows   = np.arange(B)
        l_pos  = ptf[rows, np.argmax(np.where(pos, log_pT, -999.0), axis=1)]
        l_neg  = ptf[rows, np.argmax(np.where(neg, log_pT, -999.0), axis=1)]

        def to_4mom(l):
            eta = l[:, 0]
            pT  = np.exp(np.clip(l[:, 3], -10, 10))
            E   = pT * np.cosh(eta)
            px  = pT * l[:, 2]   # cos_phi * pT
            py  = pT * l[:, 1]   # sin_phi * pT
            pz  = pT * np.sinh(eta)
            return E, px, py, pz

        E1, px1, py1, pz1 = to_4mom(l_pos)
        E2, px2, py2, pz2 = to_4mom(l_neg)

        E_Z  = E1 + E2
        px_Z = px1 + px2
        py_Z = py1 + py2
        pz_Z = pz1 + pz2
        pT_Z = np.sqrt(np.maximum(px_Z**2 + py_Z**2, 1e-12))

        m = has_pair
        slot[m, 0] = np.log(np.maximum(E_Z[m], 1e-3))
        slot[m, 1] = py_Z[m] / pT_Z[m]
        slot[m, 2] = px_Z[m] / pT_Z[m]
        slot[m, 3] = np.clip(pz_Z[m] / np.maximum(E_Z[m], 1e-6),
                             -1 + 1e-6, 1 - 1e-6)
        slot[m, 5] = 1.0

    return slot


# ── Per-file processing ───────────────────────────────────────────────────────

def process_file(path, proc, dry_run=False):
    tag = '[DRY-RUN] ' if dry_run else ''
    print(f"\n{tag}Processing {proc}: {path}", flush=True)

    with h5py.File(path, 'r') as f:
        N      = f['parton_features'].shape[0]
        P_old  = f['parton_features'].shape[1]
        has_np = 'n_partons' in f
    print(f"  N={N:,}  P_old={P_old}  has_n_partons={has_np}", flush=True)

    if dry_run:
        print("  [DRY-RUN] skipping writes.")
        return

    # Back up original file
    bak = path.with_suffix('.hdf5.bak')
    if not bak.exists():
        print(f"  Creating backup: {bak}", flush=True)
        shutil.copy2(path, bak)
    else:
        print(f"  Backup already exists: {bak}", flush=True)

    with h5py.File(bak, 'r') as src, h5py.File(path, 'a') as dst:
        old_pf = src['parton_features'][:]  # (N, P_old, 6)

        # Infer n_partons from is_valid if not stored
        if has_np:
            n_par = src['n_partons'][:].astype(np.int32)
        else:
            n_par = (old_pf[:, :, 5] > 0.5).sum(axis=1).astype(np.int32)

        # ── Re-encode existing PDG norms from ÷10 to ÷16 ─────────────────
        new_pf = np.zeros((N, MAX_PARTONS_NEW, N_PARTON_FEAT), dtype=np.float32)
        new_pf[:, :P_old, :] = old_pf

        # Rescale PDG column (index 4) for all existing slots
        pdg_col = new_pf[:, :P_old, 4]
        new_pdg = np.vectorize(rescale_pdg_norm)(pdg_col)
        new_pf[:, :P_old, 4] = new_pdg.astype(np.float32)

        # For ttbar: slots 2 and 3 with is_valid=1 are top/antitop
        # Their old pdg_norm=0 (missing from _PDG_CLASS); re-encode as top
        if proc == 'ttbar':
            for slot_idx, top_norm in [(2, _PDG_NORM_T), (3, _PDG_NORM_TB)]:
                valid_slot = (old_pf[:, slot_idx, 5] > 0.5) if P_old > slot_idx else np.zeros(N, bool)
                new_pf[valid_slot, slot_idx, 4] = top_norm
            print(f"  Fixed top-quark PDG norms in slots 2,3 "
                  f"({(old_pf[:,2,5]>0.5).sum():,} t, {(old_pf[:,3,5]>0.5).sum():,} tbar)",
                  flush=True)

        # ── Add boson slot ────────────────────────────────────────────────
        new_npar = n_par.copy()

        if proc == 'zjets':
            print("  Reconstructing Z kinematics from OSSF lepton pairs ...", flush=True)
            prt_f  = src['particle_features'][:]
            n_found = 0
            for s in range(0, N, CHUNK):
                e     = min(s + CHUNK, N)
                bslot = reconstruct_z_slot(prt_f[s:e])
                ins   = np.clip(n_par[s:e], 0, MAX_PARTONS_NEW - 1)
                new_pf[np.arange(s, e), ins, :] = bslot
                found = bslot[:, 5] > 0.5
                new_npar[s:e][found] += 1
                n_found += int(found.sum())
                if s % (CHUNK * 10) == 0:
                    print(f"    {e:,}/{N:,}  Z found so far: {n_found:,}", flush=True)
            print(f"  Z reconstructed: {n_found:,}/{N:,} ({100*n_found/N:.1f}%)", flush=True)

        elif proc == 'wjets':
            print("  Adding W slot (PDG=24, kinematics=0) ...", flush=True)
            for s in range(0, N, CHUNK):
                e   = min(s + CHUNK, N)
                B   = e - s
                ins = np.clip(n_par[s:e], 0, MAX_PARTONS_NEW - 1)
                bslot = np.zeros((B, N_PARTON_FEAT), dtype=np.float32)
                bslot[:, 4] = _PDG_NORM_W
                bslot[:, 5] = 1.0
                new_pf[np.arange(s, e), ins, :] = bslot
                new_npar[s:e] += 1
            print(f"  W slot added to all {N:,} events", flush=True)

        # ── Write datasets back ───────────────────────────────────────────
        print("  Writing parton_features ...", flush=True)
        del dst['parton_features']
        dst.create_dataset('parton_features', data=new_pf,
                           chunks=(min(1000, N), MAX_PARTONS_NEW, N_PARTON_FEAT),
                           compression='gzip')

        if 'n_partons' in dst:
            del dst['n_partons']
        dst.create_dataset('n_partons', data=new_npar.astype(np.int32),
                           compression='gzip')

    print(f"  Final shape: {new_pf.shape}", flush=True)
    print(f"  n_partons: min={new_npar.min()}  max={new_npar.max()}  "
          f"mean={new_npar.mean():.2f}", flush=True)


# ── Recompute normalisation stats ─────────────────────────────────────────────

def recompute_stats(data_dir, processes, sample_per_proc=5000, seed=42):
    print("\nRecomputing normalisation_stats.json ...", flush=True)
    rng = np.random.default_rng(seed)

    all_part, all_log_n, all_cond = [], [], []

    for proc in processes:
        path = data_dir / f'{proc}.hdf5'
        if not path.exists():
            continue
        with h5py.File(path, 'r') as f:
            N   = f['parton_features'].shape[0]
            idx = rng.choice(N, min(sample_per_proc, N), replace=False)
            idx.sort()

            pf_raw  = f['parton_features'][idx]            # (S, 6, 6)
            ptf_raw = f['particle_features'][idx]          # (S, P_part, 7)
            n_par   = f['n_partons'][idx].astype(np.int32)

            # Particle stats: first 6 features, valid particles only
            valid_p = ptf_raw[:, :, 6] > 0.5              # (S, P_part)
            all_part.append(ptf_raw[:, :, :6][valid_p])    # valid particles

            # Jet (log_npart) stats
            npart = valid_p.sum(axis=1)
            all_log_n.append(np.log(np.maximum(npart, 1)).astype(np.float32))

            # Conditioning stats: flat parton features
            cond = pf_raw.reshape(len(idx), -1).astype(np.float32)
            all_cond.append(cond)

    parts  = np.concatenate(all_part,  axis=0).astype(np.float32)
    log_ns = np.concatenate(all_log_n, axis=0).astype(np.float32)
    conds  = np.concatenate(all_cond,  axis=0).astype(np.float32)

    def safe_std(x):
        s = x.std(axis=0)
        return np.where(s > 1e-8, s, 1.0)

    stats = {
        'part_mean': parts.mean(axis=0).tolist(),
        'part_std':  safe_std(parts).tolist(),
        'jet_mean':  [float(log_ns.mean())],
        'jet_std':   [float(max(float(log_ns.std()), 1e-6))],
        'cond_mean': conds.mean(axis=0).tolist(),
        'cond_std':  safe_std(conds).tolist(),
    }

    out = data_dir / 'normalisation_stats.json'
    out.write_text(json.dumps(stats, indent=2))
    print(f"  Wrote {out}", flush=True)
    print(f"  cond dims: {len(stats['cond_mean'])} "
          f"(expected {MAX_PARTONS_NEW * N_PARTON_FEAT}={MAX_PARTONS_NEW}×{N_PARTON_FEAT})",
          flush=True)
    print(f"  part dims: {len(stats['part_mean'])}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir',  default=str(DATA_DIR))
    ap.add_argument('--processes', nargs='+', default=PROCESSES)
    ap.add_argument('--dry_run',   action='store_true')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    for proc in args.processes:
        path = data_dir / f'{proc}.hdf5'
        if not path.exists():
            print(f"Skipping {proc}: {path} not found")
            continue
        process_file(path, proc, dry_run=args.dry_run)

    if not args.dry_run:
        recompute_stats(data_dir, args.processes)
    print("\nAll done.", flush=True)


if __name__ == '__main__':
    main()
