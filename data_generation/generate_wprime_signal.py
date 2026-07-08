"""
generate_wprime_signal.py — W' → W(mX) Z(mY) → qq̄ qq̄ signal and QCD background
=====================================================================================
Pythia8-only (no MadGraph) generator producing the LHC Olympics-style
W' → XY parametric signal and QCD dijet background, stored in the pipeline
HDF5 format for use with per_parton_cond_train.py.

Signal: W' (3.5 TeV) → W(mX) + Z(mY) → (qq̄)(qq̄)
  Grid: mX, mY ∈ {50,100,...,600} GeV  →  12×12 = 144 signal points
  Pythia8 mapping: X = W boson (PDG 24) with mass mX
                   Y = Z boson (PDG 23) with mass mY
  Coupling: Wprime:coup2WZ=1, Wprime:vq≈0  (pure WZ coupling, no direct qq decay)

Background: QCD dijet (HardQCD:all) with pTHatMin = 1500 GeV
  (kinematic range matching the 3.5 TeV W' signal region)

HDF5 output (6-slot format, consistent with SM files in full_event_mixed/):
  particle_features : (N, 500, 7)  [η, sin_φ, cos_φ, log_pT, pid_cat, charge, occupancy]
  parton_features   : (N, 6, 6)   [log_E, sin_φ, cos_φ, pz/E, pdg_norm(÷16), occupancy]
    Slot 0: incoming parton (beam A side)
    Slot 1: incoming parton (beam B side)
    Slot 2: W boson (X, PDG ±24)  for signal  |  leading hard parton for background
    Slot 3: Z boson (Y, PDG  23)  for signal  |  subleading hard parton for background
    Slots 4-5: empty (zero-padded)
  n_partons         : (N,)  number of occupied parton slots (4 for signal/background)
  event_weights     : (N,)  all +1.0 (LO)
  attrs: mass_x, mass_y (0 for background)

Usage:
  # Signal at one mass point:
  python generate_wprime_signal.py --process signal --mass-x 500 --mass-y 100 \\
      --nevents 100000 --seed 42 --out /pscratch/sd/l/lcondren/MCsim/wprime_signal/

  # QCD background:
  python generate_wprime_signal.py --process background --nevents 100000 --seed 1 \\
      --out /pscratch/sd/l/lcondren/MCsim/wprime_signal/

  # Full grid via SLURM: bash submit_wprime_grid.slurm
"""

import argparse
import math
import sys
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import h5py
import pythia8


# ── Constants (matching existing pipeline) ────────────────────────────────────

N_PART_MAX    = 500
N_PART_FEAT   = 7     # [η, sin_φ, cos_φ, log_pT, pid_cat, charge, occupancy]
N_PARTONS     = 4     # 2 incoming + W(X) + Z(Y); hardcoded 4 to match training code
N_PARTON_FEAT = 6     # [log_E, sin_φ, cos_φ, pz/E, pdg_norm(÷16), occupancy]
PT_MIN        = 0.3   # GeV
ETA_MAX       = 5.0

WPRIME_MASS   = 3500.0  # GeV

# Signal grid: mX, mY ∈ {50, 100, ..., 600} GeV  →  144 points
MASS_GRID = list(range(50, 650, 50))   # [50, 100, ..., 600]


# ── PDG helpers ───────────────────────────────────────────────────────────────
# Class → index / 16.0  (consistent with SM generation scripts and SM HDF5 files)
# Classes 0-12: quarks/gluon; 13=Z, 14=W+, 15=W-

_PDG_CLASS = {
    21: 0,                          # gluon
    1: 1,  -1: 2,                  # d, dbar
    2: 3,  -2: 4,                  # u, ubar
    3: 5,  -3: 6,                  # s, sbar
    4: 7,  -4: 8,                  # c, cbar
    5: 9,  -5: 10,                 # b, bbar
    6: 11, -6: 12,                 # t, tbar
    23: 13,                        # Z
    24: 14, -24: 15,               # W+, W-
}

def _pdg_norm(pid: int) -> float:
    return float(_PDG_CLASS.get(pid, 0)) / 16.0

def _pid_cat(abs_pid: int, charge: float) -> int:
    if abs_pid == 22:     return 2   # photon
    if abs_pid == 11:     return 3   # electron
    if abs_pid == 13:     return 4   # muon
    if abs(charge) > 0.5: return 0   # charged hadron
    return 1                         # neutral hadron


# ── Parton-slot helper ────────────────────────────────────────────────────────

def _fill_parton_slot(arr: np.ndarray, slot: int, p) -> None:
    """Fill one row of a (N_PARTONS, N_PARTON_FEAT) array from a Pythia8 particle."""
    E = max(abs(p.e()), 1e-3)
    phi = math.atan2(p.py(), p.px())
    arr[slot, 0] = math.log(E)
    arr[slot, 1] = math.sin(phi)
    arr[slot, 2] = math.cos(phi)
    arr[slot, 3] = float(np.clip(p.pz() / E, -1.0 + 1e-6, 1.0 - 1e-6))
    arr[slot, 4] = _pdg_norm(p.id())
    arr[slot, 5] = 1.0   # occupancy flag


# ── Parton extraction — signal ────────────────────────────────────────────────

def _extract_partons_signal(event) -> np.ndarray:
    """
    Extract 4 parton slots from a W' → W(mX) Z(mY) Pythia8 event.

    W' undergoes many ISR recoil steps (status -44) before finally decaying
    at a high event-record index (status -62).  The W (PDG ±24) and Z (PDG 23)
    daughters of that final W' have status -22 and are the particles we want.

    We identify them robustly by scanning the full event record and checking
    that each W/Z candidate's direct mother has id=34 (W').

    Slots 0-1: incoming partons (status -21), sorted by pz descending
    Slot  2  : W boson (PDG ±24) from W' decay  →  X particle
    Slot  3  : Z boson (PDG  23) from W' decay  →  Y particle
    """
    arr = np.zeros((N_PARTONS, N_PARTON_FEAT), dtype=np.float32)

    incoming = []
    w_boson  = None
    z_boson  = None

    for i in range(event.size()):
        p   = event[i]
        pid = p.id()

        if p.status() == -21:
            incoming.append(p)
            continue

        # W boson: direct daughter of any W' (mother has |id|=34)
        if abs(pid) == 24 and w_boson is None:
            m1 = p.mother1()
            if 0 < m1 < event.size() and abs(event[m1].id()) == 34:
                w_boson = p

        # Z boson: direct daughter of any W'
        elif pid == 23 and z_boson is None:
            m1 = p.mother1()
            if 0 < m1 < event.size() and abs(event[m1].id()) == 34:
                z_boson = p

        # Stop once we have both (they always appear before FSR copies)
        if w_boson is not None and z_boson is not None:
            break

    # Sort incoming by pz descending (beam-A side first)
    incoming.sort(key=lambda q: q.pz(), reverse=True)

    for slot, p in enumerate(incoming[:2]):
        E = max(abs(p.e()), 1e-3)
        arr[slot, 0] = math.log(E)
        arr[slot, 1] = 0.0   # incoming partons: phi undefined, set to 0
        arr[slot, 2] = 0.0
        arr[slot, 3] = float(np.clip(p.pz() / E, -1.0 + 1e-6, 1.0 - 1e-6))
        arr[slot, 4] = _pdg_norm(p.id())
        arr[slot, 5] = 1.0

    if w_boson is not None:
        _fill_parton_slot(arr, 2, w_boson)
    if z_boson is not None:
        _fill_parton_slot(arr, 3, z_boson)

    return arr


# ── Parton extraction — background ────────────────────────────────────────────

def _extract_partons_background(event) -> np.ndarray:
    """
    Extract 4 parton slots from a QCD 2→2 Pythia8 event.

    Slots 0-1: incoming partons (status -21)
    Slots 2-3: outgoing hard-scatter colored partons (status -23),
               sorted by pT descending
    """
    arr = np.zeros((N_PARTONS, N_PARTON_FEAT), dtype=np.float32)

    incoming = []
    outgoing = []

    scan_limit = min(30, event.size())
    for i in range(scan_limit):
        p      = event[i]
        status = p.status()
        pid    = p.id()

        if status == -21:
            incoming.append(p)
        elif status == -23 and abs(pid) in {1, 2, 3, 4, 5, 6, 21}:
            outgoing.append(p)

    incoming.sort(key=lambda q: q.pz(), reverse=True)
    outgoing.sort(key=lambda q: q.pT(), reverse=True)

    for slot, p in enumerate(incoming[:2]):
        E = max(abs(p.e()), 1e-3)
        arr[slot, 0] = math.log(E)
        arr[slot, 1] = 0.0
        arr[slot, 2] = 0.0
        arr[slot, 3] = float(np.clip(p.pz() / E, -1.0 + 1e-6, 1.0 - 1e-6))
        arr[slot, 4] = _pdg_norm(p.id())
        arr[slot, 5] = 1.0

    for slot, p in enumerate(outgoing[:2]):
        _fill_parton_slot(arr, 2 + slot, p)

    return arr


# ── Particle extraction (identical to existing pipeline) ─────────────────────

def _extract_particles(event) -> Optional[np.ndarray]:
    """
    Extract particle features from all final-state Pythia8 particles.
    Returns (N_PART_MAX, 7) array or None if the event is empty after cuts.
    """
    etas, sin_phis, cos_phis, log_pts, pid_cats, charges = [], [], [], [], [], []

    for i in range(event.size()):
        p = event[i]
        if not p.isFinal():
            continue
        abs_pid = abs(p.id())
        if abs_pid in {12, 14, 16}:   # skip neutrinos
            continue
        pT = p.pT()
        if pT < PT_MIN:
            continue
        eta = p.eta()
        if abs(eta) > ETA_MAX:
            continue
        phi = p.phi()
        chg = p.charge()
        etas.append(eta)
        sin_phis.append(math.sin(phi))
        cos_phis.append(math.cos(phi))
        log_pts.append(math.log(pT))
        pid_cats.append(float(_pid_cat(abs_pid, chg)))
        charges.append(float(np.sign(chg)))

    if not etas:
        return None

    order  = np.argsort(log_pts)[::-1]   # sort by pT descending
    n_fill = min(len(etas), N_PART_MAX)
    arr    = np.zeros((N_PART_MAX, N_PART_FEAT), dtype=np.float32)

    arr[:n_fill, 0] = np.array(etas,     dtype=np.float32)[order[:n_fill]]
    arr[:n_fill, 1] = np.array(sin_phis, dtype=np.float32)[order[:n_fill]]
    arr[:n_fill, 2] = np.array(cos_phis, dtype=np.float32)[order[:n_fill]]
    arr[:n_fill, 3] = np.array(log_pts,  dtype=np.float32)[order[:n_fill]]
    arr[:n_fill, 4] = np.array(pid_cats, dtype=np.float32)[order[:n_fill]]
    arr[:n_fill, 5] = np.array(charges,  dtype=np.float32)[order[:n_fill]]
    arr[:n_fill, 6] = 1.0   # occupancy

    return arr


# ── Pythia8 initialisation ────────────────────────────────────────────────────

def _init_pythia_signal(mass_x: float, mass_y: float, seed: int) -> pythia8.Pythia:
    """
    Configure Pythia8 for W' → W(mX) Z(mY) → qq̄ qq̄.

    W  (PDG 24) = X particle, hadronic decays only (W+ → ud̄, cs̄)
    Z  (PDG 23) = Y particle, hadronic decays only (Z → uū, dd̄, ss̄, cc̄)
    W' (PDG 34) decays exclusively to WZ via Wprime:coup2WZ coupling.
    Direct qq̄ coupling of W' is suppressed to ≈0.

    MPI is ON (matches SM data generation; adds realistic underlying-event MET).
    ISR, FSR, and hadronization are ON.
    """
    py = pythia8.Pythia("", False)   # second arg suppresses banner
    py.readString("Print:quiet = on")

    # Beams
    py.readString("Beams:idA = 2212")
    py.readString("Beams:idB = 2212")
    py.readString("Beams:eCM = 13000.")

    # Hard process
    py.readString("NewGaugeBoson:ffbar2Wprime = on")

    # W' properties: mass 3.5 TeV, decays only to WZ
    py.readString(f"34:m0 = {WPRIME_MASS:.1f}")
    py.readString("Wprime:vq    = 0.0000001")   # suppress direct qq decay
    py.readString("Wprime:aq    = 0.")
    py.readString("Wprime:vl    = 0.")
    py.readString("Wprime:al    = 0.")
    py.readString("Wprime:coup2WZ = 1.")          # W' → WZ coupling ON

    # X = W boson (PDG 24) with mass mX, hadronic decays only
    py.readString(f"24:m0 = {mass_x:.1f}")
    py.readString("24:onMode = off")
    py.readString("24:onIfany = -1 2")    # W+ → u d̄
    py.readString("24:onIfany = -3 4")    # W+ → c s̄

    # Y = Z boson (PDG 23) with mass mY, hadronic decays only
    py.readString(f"23:m0 = {mass_y:.1f}")
    py.readString("23:onMode = off")
    py.readString("23:onIfany = -1 1")    # Z → d d̄
    py.readString("23:onIfany = -2 2")    # Z → u ū
    py.readString("23:onIfany = -3 3")    # Z → s s̄
    py.readString("23:onIfany = -4 4")    # Z → c c̄

    # MPI on: matches SM data generation and adds realistic underlying-event MET.
    py.readString("PartonLevel:MPI = on")

    # Reproducibility
    py.readString("Random:setSeed = on")
    py.readString(f"Random:seed = {seed}")

    py.init()
    return py


def _init_pythia_background(seed: int) -> pythia8.Pythia:
    """
    Configure Pythia8 for QCD dijet background in the W' kinematic region.

    pTHatMin = 1500 GeV efficiently populates the ~3.5 TeV dijet-mass region
    relevant to the W' signal. MPI on to match signal generation.
    """
    py = pythia8.Pythia("", False)
    py.readString("Print:quiet = on")

    # Beams
    py.readString("Beams:idA = 2212")
    py.readString("Beams:idB = 2212")
    py.readString("Beams:eCM = 13000.")

    # Hard process: QCD 2→2 (all sub-processes, 5-flavour)
    py.readString("HardQCD:all = on")
    py.readString("HardQCD:nQuarkNew = 5")
    py.readString("PhaseSpace:pTHatMin = 1500.")

    # MPI on: matches signal generation settings.
    py.readString("PartonLevel:MPI = on")

    # Reproducibility
    py.readString("Random:setSeed = on")
    py.readString(f"Random:seed = {seed}")

    py.init()
    return py


# ── Main generation loop ──────────────────────────────────────────────────────

def generate_events(process: str, mass_x: float, mass_y: float,
                    n_events: int, seed: int, out_path: Path) -> None:
    """
    Generate n_events and write to out_path in the pipeline HDF5 format.

    process : 'signal' or 'background'
    mass_x  : mX in GeV (signal only; ignored for background)
    mass_y  : mY in GeV (signal only; ignored for background)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if process == "signal":
        print(f"[signal]  mX={mass_x:.0f} mY={mass_y:.0f}  n={n_events}  seed={seed}",
              flush=True)
        py = _init_pythia_signal(mass_x, mass_y, seed)
        extract_partons = _extract_partons_signal
    else:
        print(f"[background]  pTHatMin=1500  n={n_events}  seed={seed}", flush=True)
        py = _init_pythia_background(seed)
        extract_partons = _extract_partons_background
        mass_x = mass_y = 0.0

    parts_list   = []
    partons_list = []
    n_saved      = 0
    n_tried      = 0
    t0           = time.time()

    while n_saved < n_events:
        n_tried += 1
        if not py.next():
            continue

        pf = _extract_particles(py.event)
        if pf is None:
            continue

        parts_list.append(pf)
        partons_list.append(extract_partons(py.event))
        n_saved += 1

        if n_saved % 10_000 == 0:
            elapsed = time.time() - t0
            rate    = n_saved / max(elapsed, 1e-9)
            eta     = (n_events - n_saved) / max(rate, 1e-9)
            print(f"  {n_saved:,}/{n_events:,}  ({n_tried:,} tried)  "
                  f"{rate:.0f} ev/s  ETA {eta/60:.1f} min", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n_saved:,} events in {elapsed:.1f}s "
          f"({n_saved/elapsed:.0f} ev/s)", flush=True)

    parts   = np.stack(parts_list,   axis=0).astype(np.float32)   # (N, 500, 7)
    partons = np.stack(partons_list, axis=0).astype(np.float32)   # (N, 4, 6)
    weights = np.ones(n_saved, dtype=np.float32)

    print(f"  Writing → {out_path}", flush=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("particle_features", data=parts,
                         compression="gzip", chunks=(min(1000, n_saved), N_PART_MAX, N_PART_FEAT))
        f.create_dataset("parton_features",   data=partons,
                         compression="gzip", chunks=(min(1000, n_saved), N_PARTONS, N_PARTON_FEAT))
        f.create_dataset("event_weights",     data=weights, compression="gzip")
        f.attrs["mass_x"]  = mass_x
        f.attrs["mass_y"]  = mass_y
        f.attrs["process"] = process
        f.attrs["wprime_mass"] = WPRIME_MASS if process == "signal" else 0.0
    print(f"  {n_saved:,} events written.", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate W' signal or QCD background in pipeline HDF5 format.")
    p.add_argument("--process",  default="signal", choices=["signal", "background"],
                   help="'signal': W'→WZ→4j  |  'background': QCD dijet")
    p.add_argument("--mass-x",   type=float, default=None,
                   help="mX (W mass) in GeV  [signal only; overridden by --task-id]")
    p.add_argument("--mass-y",   type=float, default=None,
                   help="mY (Z mass) in GeV  [signal only; overridden by --task-id]")
    p.add_argument("--task-id",  type=int,   default=None,
                   help="SLURM array task id 0-143; maps to mX=50+12*(id//12)*50, mY=50+(id%%12)*50")
    p.add_argument("--nevents",  type=int,   default=100_000)
    p.add_argument("--seed",     type=int,   default=None,
                   help="RNG seed (default: 1000 + task_id for grid tasks, 42 otherwise)")
    p.add_argument("--out",      type=Path,
                   default=Path("/pscratch/sd/l/lcondren/MCsim/wprime_signal"),
                   help="Output directory")
    args = p.parse_args()

    if args.task_id is not None:
        # Grid task: mX = 50, 100, ..., 600  (12 rows);  mY = 50, 100, ..., 600  (12 cols)
        args.process = "signal"
        args.mass_x  = 50.0 + (args.task_id // 12) * 50.0
        args.mass_y  = 50.0 + (args.task_id  % 12) * 50.0
        if args.seed is None:
            args.seed = 1000 + args.task_id
    else:
        if args.mass_x is None:
            args.mass_x = 500.0
        if args.mass_y is None:
            args.mass_y = 100.0
        if args.seed is None:
            args.seed = 42

    if args.process == "signal":
        fname = f"signal_mX{int(args.mass_x):04d}_mY{int(args.mass_y):04d}.hdf5"
    else:
        fname = "background.hdf5"

    out_path = args.out / fname
    generate_events(args.process, args.mass_x, args.mass_y,
                    args.nevents, args.seed, out_path)


if __name__ == "__main__":
    main()
