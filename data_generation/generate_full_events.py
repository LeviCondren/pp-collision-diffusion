"""
generate_full_events.py  (v2 — MG5+Pythia8 matched pairs)
==========================================================
Generates matched (MG5 hard-scatter parton kinematics, Pythia8 shower) pairs
for GSGM training.

Workflow per chunk:
  1. MG5 generates LHE events (hard-scatter partons, shower=OFF).
  2. Pythia8 showers each LHE event sequentially (frameType=4).
  3. MG5 parton kinematics → event_features (conditioning, known before shower).
     Pythia8 final-state particles → particle_features (target).

Output HDF5 format:
  particle_features : (N, 500, 7)
    [eta, sin_phi, cos_phi, log_pT, pid_category, charge, mask]
    pid_category:  0=charged hadron, 1=neutral hadron, 2=photon, 3=electron, 4=muon

  event_features : (N, 3)   — all knowable from MG5 LHE BEFORE showering
    [log_pThat_mg5, proc_id, log_shat_mg5]
    log_pThat_mg5 : log(max pT of colored LHE final-state partons)
    proc_id       : 0=dijet, 1=zjets
    log_shat_mg5  : log(invariant mass of full LHE hard system)

Processes:
  dijet  : p p > j j        pTj > 100 GeV
  zjets  : p p > e+ e- j  + p p > mu+ mu- j   mll > 10 GeV, pTj > 10 GeV

Usage:
  # Step 1 — one-time process setup (fast, ~10 min total):
  python generate_full_events.py --process dijet --setup-only
  python generate_full_events.py --process zjets --setup-only

  # Step 2 — parallel generation (SLURM array, 10 workers × 50k events = 500k):
  python generate_full_events.py --process dijet --nevents 50000 --chunk-id $SLURM_ARRAY_TASK_ID
  python generate_full_events.py --process zjets --nevents 50000 --chunk-id $SLURM_ARRAY_TASK_ID

  # Step 3 — merge:
  python generate_full_events.py --process dijet --merge
  python generate_full_events.py --process zjets --merge
  # Also delete old normalisation_stats.json so training script recomputes:
  rm -f /pscratch/sd/l/lcondren/MCsim/full_event_fpcd/normalisation_stats.json

Parton feature format (N, 4, 6) stored as parton_features:
  Slots: [init_beam+, init_beam-, final_colored_1, final_colored_2]
  Features per slot: [log_E, sin_phi, cos_phi, pz_over_E, pdg_norm, is_valid]
  - log_E    : log(parton energy in GeV)
  - sin_phi  : sin(phi); 0.0 for beam-collinear initial partons
  - cos_phi  : cos(phi); 0.0 for beam-collinear initial partons
  - pz_over_E: pz/E = tanh(eta) for massless particles; ≈ ±1 for initial partons
  - pdg_norm : PDG class index / 16.0  (see _PDG_CLASS below)
  - is_valid : 1.0 if slot is filled, 0.0 if padding
"""

import sys, os, math, time, argparse, gzip, shutil, subprocess, re
from pathlib import Path
import numpy as np
import h5py
import pythia8

# ── bsm_pipeline for LHE parsing ──────────────────────────────────────────────
BSM_PIPELINE = "/global/u2/l/lcondren/ContinuousParamFit/bsm_pipeline"
sys.path.insert(0, BSM_PIPELINE)
from bsm_pipeline.mg5_runner import parse_lhe, _get_mg5_env, MG5_BIN, PYTHON3

# ── Constants ──────────────────────────────────────────────────────────────────
N_PART_MAX    = 500
N_PART_FEAT   = 7     # eta, sin_phi, cos_phi, log_pT, pid_category, charge, mask
MAX_PARTONS   = 6     # 2 initial + up to 3 final colored + 1 hard boson slot
N_PARTON_FEAT = 6     # log_E, sin_phi, cos_phi, pz_over_E, pdg_norm, is_valid
ETA_MAX       = 5.0
PT_MIN        = 0.3   # GeV

OUT_DIR  = Path("/pscratch/sd/l/lcondren/MCsim/full_event_fpcd")
MG5_WORK = Path("/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/mg5_work")

# Colored PDG IDs (quarks + gluon) — hard-scatter initiators and final-state jets
_COLORED = {1, 2, 3, 4, 5, 6, 21}

# PDG class index ÷16: g=0, d/dbar=1/2, u/ubar=3/4, s/sbar=5/6, c/cbar=7/8,
#   b/bbar=9/10, t/tbar=11/12, Z=13, W+=14, W-=15
_PDG_CLASS = {21: 0, 1: 1, -1: 2, 2: 3, -2: 4, 3: 5, -3: 6, 4: 7, -4: 8, 5: 9, -5: 10,
              6: 11, -6: 12, 23: 13, 24: 14, -24: 15}

def _pdg_norm(pid: int) -> float:
    return float(_PDG_CLASS.get(pid, 0)) / 16.0


# ── PID categorisation ─────────────────────────────────────────────────────────
def _pid_cat(abs_pid: int, charge: float) -> int:
    if abs_pid == 22:       return 2   # photon
    if abs_pid == 11:       return 3   # electron
    if abs_pid == 13:       return 4   # muon
    if abs(charge) > 0.5:   return 0   # charged hadron
    return 1                           # neutral hadron


# ── MG5 SM process setup ───────────────────────────────────────────────────────

def _configure_run_card(proc_dir: Path, process: str) -> None:
    rc = proc_dir / "Cards" / "run_card.dat"
    if not rc.exists():
        return
    txt = rc.read_text()
    # Disable systematics (slow and not needed)
    txt = re.sub(r'True\s*=\s*use_syst\b', 'False = use_syst', txt, flags=re.IGNORECASE)
    if process == "dijet":
        # Hard pT cut on jets
        txt = re.sub(r'[\d.]+\s*=\s*ptj\b', '100.0 = ptj', txt, flags=re.IGNORECASE)
        if not re.search(r'ptj\b', txt, re.IGNORECASE):
            txt += '\n100.0 = ptj\n'
    elif process == "zjets":
        # Dilepton invariant mass cut (avoids photon-pole divergence)
        txt = re.sub(r'[\d.e+\-]+\s*=\s*mmll\b', '10.0 = mmll', txt, flags=re.IGNORECASE)
        if not re.search(r'\bmmll\b', txt, re.IGNORECASE):
            txt += '\n10.0 = mmll\n'
        # Minimum pT on the recoiling parton
        txt = re.sub(r'[\d.]+\s*=\s*ptj\b', '10.0 = ptj', txt, flags=re.IGNORECASE)
        if not re.search(r'ptj\b', txt, re.IGNORECASE):
            txt += '\n10.0 = ptj\n'
    rc.write_text(txt)


def setup_mg5_proc(process: str) -> Path:
    """
    One-time creation of MG5 process directory for a SM process.
    Returns the process directory. Idempotent — returns existing dir if present.
    """
    MG5_WORK.mkdir(parents=True, exist_ok=True)
    proc_dir = MG5_WORK / f"{process}_proc"
    if proc_dir.exists():
        print(f"[MG5] Process dir already exists: {proc_dir}")
        return proc_dir

    # 5-flavour scheme: include b quarks in proton and jet definitions
    _5f = "define p = g u c d s b u~ c~ d~ s~ b~\ndefine j = g u c d s b u~ c~ d~ s~ b~\n"

    if process == "dijet":
        gen_lines = _5f + "generate p p > j j"
    elif process == "zjets":
        # Z/gamma* + 1 parton via Drell-Yan + jet; combine e and mu channels
        gen_lines = _5f + "generate p p > e+ e- j\nadd process p p > mu+ mu- j"
    else:
        raise ValueError(f"Unknown process: {process}")

    script = MG5_WORK / f"setup_{process}.mg5"
    script.write_text(f"{gen_lines}\noutput {proc_dir}\n")

    print(f"[MG5] Setting up {process} process (this takes ~5 min)...")
    r = subprocess.run(
        [PYTHON3, str(MG5_BIN), str(script)],
        capture_output=True, text=True,
        env=_get_mg5_env(), timeout=900,
    )
    script.unlink(missing_ok=True)

    if not proc_dir.exists():
        raise RuntimeError(
            f"MG5 process setup failed for {process}:\n"
            f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-2000:]}"
        )

    _configure_run_card(proc_dir, process)
    print(f"[MG5] Process dir created: {proc_dir}")
    return proc_dir


# ── MG5 LHE generation ─────────────────────────────────────────────────────────

def run_mg5_lhe(process: str, n_events: int, seed: int) -> Path:
    """
    Run MG5 (shower=OFF) to produce an LHE file.
    Returns path to the uncompressed LHE file (caller is responsible for deletion).

    Each call copies the shared template proc_dir to a per-seed working copy so
    that concurrent SLURM array tasks never share a proc directory.
    """
    template_dir = setup_mg5_proc(process)

    # Per-seed working copy — isolates concurrent array tasks from each other.
    proc_dir = MG5_WORK / f"{process}_proc_{seed}"
    if proc_dir.exists():
        shutil.rmtree(proc_dir)
    shutil.copytree(template_dir, proc_dir, symlinks=True)

    try:
        # Update nevents and seed in run card
        rc = proc_dir / "Cards" / "run_card.dat"
        if rc.exists():
            txt = rc.read_text()
            txt = re.sub(r'\d+\s*=\s*nevents\b', f'{n_events} = nevents', txt)
            txt = re.sub(r'\d+\s*=\s*iseed\b',   f'{seed} = iseed',       txt)
            rc.write_text(txt)

        # Fresh copy has no Events directory; no need to clear.
        ev_dir = proc_dir / "Events"
        ev_dir.mkdir(exist_ok=True)

        # MG5 launch script — '0' accepts defaults (no shower configured → LHE only)
        launch = MG5_WORK / f"launch_{process}_{seed}.mg5"
        launch.write_text(
            f"launch {proc_dir}\n"
            f"0\n"
        )
        t0 = time.time()
        r = subprocess.run(
            [PYTHON3, str(MG5_BIN), str(launch)],
            capture_output=True, text=True,
            env=_get_mg5_env(), timeout=7200,
        )
        launch.unlink(missing_ok=True)
        print(f"[MG5] {process} n={n_events} seed={seed}  rc={r.returncode}  "
              f"{(time.time()-t0)/60:.1f} min")

        # Locate the LHE (gz) file
        runs = sorted(
            (d for d in ev_dir.iterdir() if d.is_dir() and d.name.startswith("run_")),
            key=lambda d: d.name,
        )
        lhe_gz = None
        for run_d in reversed(runs):
            lhe_gz = next(
                (p for p in run_d.iterdir() if "unweighted_events" in p.name), None
            )
            if lhe_gz:
                break

        if lhe_gz is None:
            raise RuntimeError(
                f"[MG5] No LHE file found for {process} seed={seed}.\n"
                f"stdout:\n{r.stdout[-500:]}\nstderr:\n{r.stderr[-1000:]}"
            )

        # Decompress .lhe.gz → .lhe if needed (Pythia8 Python bindings may lack zlib)
        if str(lhe_gz).endswith(".gz"):
            lhe_plain = MG5_WORK / f"{process}_{seed}.lhe"
            with gzip.open(lhe_gz, "rb") as fi, open(lhe_plain, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            return lhe_plain
        return lhe_gz

    finally:
        # Always remove the per-seed working copy to keep disk usage bounded.
        shutil.rmtree(proc_dir, ignore_errors=True)


# ── LHE parton feature extraction ─────────────────────────────────────────────

def _parton_features(ev, process: str):
    """
    Extract (MAX_PARTONS, N_PARTON_FEAT) and n_partons from one LHEEvent.

    Slot ordering:
      0   — initial-state parton from beam+ (pz > 0)
      1   — initial-state parton from beam- (pz < 0)
      2–4 — up to 3 hardest final-state colored partons (zero-padded if absent)
      5   — hard boson slot: reconstructed Z for zjets (zero-padded for dijet)

    zjets: Z kinematics reconstructed from the hardest OSSF lepton pair (e+e-
    or μ+μ-) in the LHE final state.  The Z PDG norm is 13/16 = 0.8125.

    Features per slot: [log_E, sin_phi, cos_phi, pz_over_E, pdg_norm, is_valid]
    """
    arr = np.zeros((MAX_PARTONS, N_PARTON_FEAT), dtype=np.float32)

    # Initial-state partons: status=-1, excluding beam protons (PDG 2212)
    init_partons = [p for p in ev.particles if p.status == -1 and abs(p.pid) != 2212]
    init_partons.sort(key=lambda p: p.pz, reverse=True)   # beam+ first

    for i, p in enumerate(init_partons[:2]):
        E = max(abs(p.E), 1e-3)
        arr[i, 0] = math.log(E)
        arr[i, 1] = 0.0                               # pT≈0, phi undefined
        arr[i, 2] = 0.0
        arr[i, 3] = float(np.clip(p.pz / E, -1 + 1e-6, 1 - 1e-6))
        arr[i, 4] = _pdg_norm(p.pid)
        arr[i, 5] = 1.0

    # Final-state colored partons: up to 3 slots (leave slot 5 for boson)
    final_colored = [p for p in ev.final_state if abs(p.pid) in _COLORED]
    final_colored.sort(key=lambda p: p.pT, reverse=True)  # hardest first

    n_final = min(len(final_colored), MAX_PARTONS - 3)
    for j, p in enumerate(final_colored[:n_final]):
        E = max(abs(p.E), 1e-3)
        phi = math.atan2(p.py, p.px)
        arr[2 + j, 0] = math.log(E)
        arr[2 + j, 1] = math.sin(phi)
        arr[2 + j, 2] = math.cos(phi)
        arr[2 + j, 3] = float(np.clip(p.pz / E, -1 + 1e-6, 1 - 1e-6))
        arr[2 + j, 4] = _pdg_norm(p.pid)
        arr[2 + j, 5] = 1.0

    n_partons = 2 + n_final

    # zjets: reconstruct Z from hardest OSSF lepton pair in LHE final state
    if process == 'zjets':
        fs = ev.final_state
        elec  = [p for p in fs if p.pid == 11]
        posit = [p for p in fs if p.pid == -11]
        muon  = [p for p in fs if p.pid == 13]
        amuon = [p for p in fs if p.pid == -13]
        pair = None
        if elec and posit:
            pair = (max(elec,  key=lambda p: p.pT), max(posit, key=lambda p: p.pT))
        elif muon and amuon:
            pair = (max(muon,  key=lambda p: p.pT), max(amuon, key=lambda p: p.pT))
        if pair is not None and n_partons < MAX_PARTONS:
            p1, p2 = pair
            E_Z  = max(p1.E + p2.E, 1e-3)
            px_Z = p1.px + p2.px
            py_Z = p1.py + p2.py
            pz_Z = p1.pz + p2.pz
            pT_Z = math.sqrt(max(px_Z**2 + py_Z**2, 1e-12))
            s = n_partons
            arr[s, 0] = math.log(E_Z)
            arr[s, 1] = py_Z / pT_Z
            arr[s, 2] = px_Z / pT_Z
            arr[s, 3] = float(np.clip(pz_Z / E_Z, -1 + 1e-6, 1 - 1e-6))
            arr[s, 4] = _pdg_norm(23)
            arr[s, 5] = 1.0
            n_partons += 1

    return arr, n_partons


# ── Pythia8 particle-cloud extraction ──────────────────────────────────────────

def _parse_pythia_event(pythia) -> np.ndarray:
    """
    Extract final-state particle cloud from the current Pythia8 event.
    Returns (N_PART_MAX, 7) float32 or None if no particles pass cuts.
    """
    event = pythia.event
    etas, sin_phis, cos_phis, log_pts, pid_cats, charges = [], [], [], [], [], []

    for i in range(event.size()):
        p = event[i]
        if not p.isFinal():
            continue
        abs_pid = abs(p.id())
        if abs_pid in {12, 14, 16}:   # neutrinos invisible
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
        pid_cats.append(_pid_cat(abs_pid, chg))
        charges.append(float(np.sign(chg)))

    if not etas:
        return None

    order    = np.argsort(log_pts)[::-1]
    etas     = np.array(etas,     dtype=np.float32)[order]
    sin_phis = np.array(sin_phis, dtype=np.float32)[order]
    cos_phis = np.array(cos_phis, dtype=np.float32)[order]
    log_pts  = np.array(log_pts,  dtype=np.float32)[order]
    pid_cats = np.array(pid_cats, dtype=np.float32)[order]
    charges  = np.array(charges,  dtype=np.float32)[order]

    n_fill = min(len(etas), N_PART_MAX)
    arr = np.zeros((N_PART_MAX, N_PART_FEAT), dtype=np.float32)
    arr[:n_fill, 0] = etas[:n_fill]
    arr[:n_fill, 1] = sin_phis[:n_fill]
    arr[:n_fill, 2] = cos_phis[:n_fill]
    arr[:n_fill, 3] = log_pts[:n_fill]
    arr[:n_fill, 4] = pid_cats[:n_fill]
    arr[:n_fill, 5] = charges[:n_fill]
    arr[:n_fill, 6] = 1.0   # mask
    return arr


# ── LHE shower loop ────────────────────────────────────────────────────────────

def shower_lhe(lhe_path: Path, seed: int, process: str):
    """
    Run Pythia8 over every event in the LHE file.
    Returns (parts, partons, npartons) arrays, one row per accepted event.

    Parton features are from MG5 LHE (lhe_events[i]).
    Particle cloud is from Pythia8 (event record after pythia.next()).
    The i-th call to pythia.next() corresponds to lhe_events[i] exactly.
    """
    lhe_events = parse_lhe(lhe_path)
    n_lhe = len(lhe_events)
    print(f"  Parsed {n_lhe} LHE events from {lhe_path.name}")

    # Pythia8 in LHE shower mode
    py = pythia8.Pythia("", False)
    for s in [
        "Print:quiet = on",
        f"Beams:frameType = 4",
        f"Beams:LHEF = {lhe_path}",
        "PartonLevel:ISR = on",
        "PartonLevel:MPI = on",
        "PartonLevel:FSR = on",
        "HadronLevel:all = on",
        "Next:numberShowEvent = 0",
        f"Random:setSeed = on",
        f"Random:seed = {seed % 900_000_000}",
    ]:
        py.readString(s)
    py.init()

    parts_list    = []
    partons_list  = []
    npartons_list = []
    t0 = time.time()

    for evt_idx in range(n_lhe):
        if not py.next():
            print(f"  pythia.next() returned False at LHE event {evt_idx}; stopping.")
            break

        arr = _parse_pythia_event(py)
        if arr is None:
            continue   # no particles pass acceptance; skip this matched pair

        parton_arr, n_par = _parton_features(lhe_events[evt_idx], process)
        parts_list.append(arr)
        partons_list.append(parton_arr)
        npartons_list.append(n_par)

        if len(parts_list) % 5000 == 0:
            elapsed = time.time() - t0
            rate    = len(parts_list) / elapsed
            eta_min = (n_lhe - evt_idx) / rate / 60.0
            print(f"  {len(parts_list)}/{n_lhe}  ({rate:.0f} evt/s  ETA {eta_min:.1f} min)",
                  flush=True)

    py.stat()
    parts    = (np.stack(parts_list,   axis=0) if parts_list
                else np.zeros((0, N_PART_MAX, N_PART_FEAT), dtype=np.float32))
    partons  = (np.stack(partons_list, axis=0) if partons_list
                else np.zeros((0, MAX_PARTONS, N_PARTON_FEAT), dtype=np.float32))
    npartons = (np.array(npartons_list, dtype=np.int32) if npartons_list
                else np.zeros(0, dtype=np.int32))
    return parts, partons, npartons


# ── Main generation ────────────────────────────────────────────────────────────

def generate(process: str, n_events: int, seed: int, chunk_id: int) -> Path:
    print(f"\n[{process}] chunk={chunk_id}  n_events={n_events}  seed={seed}")

    print(f"[{process}] chunk={chunk_id}: Running MG5...")
    lhe_path = run_mg5_lhe(process, n_events, seed)

    print(f"[{process}] chunk={chunk_id}: Showering with Pythia8...")
    parts, partons, npartons = shower_lhe(lhe_path, seed, process)

    lhe_path.unlink(missing_ok=True)   # clean up temp LHE

    n_saved = len(parts)
    fname = OUT_DIR / f"{process}_chunk{chunk_id:04d}.hdf5"
    with h5py.File(fname, "w") as f:
        f.create_dataset("particle_features", data=parts,    compression="gzip")
        f.create_dataset("parton_features",   data=partons,  compression="gzip")
        f.create_dataset("n_partons",         data=npartons, compression="gzip")
    print(f"[{process}] chunk={chunk_id}: Saved {n_saved} events → {fname}")
    return fname


# ── Merge chunks ───────────────────────────────────────────────────────────────

def merge_chunks(process: str) -> None:
    files = sorted(OUT_DIR.glob(f"{process}_chunk*.hdf5"))
    if not files:
        print(f"No chunk files found for {process}")
        return

    all_parts, all_partons, all_npartons = [], [], []

    # Append to existing merged file if present — never overwrite accumulated data.
    out = OUT_DIR / f"{process}.hdf5"
    if out.exists():
        with h5py.File(out, "r") as h:
            all_parts.append(h["particle_features"][:])
            all_partons.append(h["parton_features"][:])
            if "n_partons" in h:
                all_npartons.append(h["n_partons"][:])
        print(f"Loaded existing {out.name}: {all_parts[0].shape[0]:,} events")

    for fpath in files:
        with h5py.File(fpath, "r") as h:
            all_parts.append(h["particle_features"][:])
            all_partons.append(h["parton_features"][:])
            if "n_partons" in h:
                all_npartons.append(h["n_partons"][:])

    parts   = np.concatenate(all_parts,   axis=0)
    partons = np.concatenate(all_partons, axis=0)

    with h5py.File(out, "w") as f:
        f.create_dataset("particle_features", data=parts,   compression="gzip")
        f.create_dataset("parton_features",   data=partons, compression="gzip")
        if all_npartons:
            npartons = np.concatenate(all_npartons, axis=0)
            f.create_dataset("n_partons", data=npartons, compression="gzip")
    print(f"Merged {len(files)} new chunks → {out}  ({parts.shape[0]} events total)")

    for fpath in files:
        fpath.unlink()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MG5+Pythia8 matched pairs")
    parser.add_argument("--process",    choices=["dijet", "zjets"], required=True)
    parser.add_argument("--nevents",    type=int, default=50_000,
                        help="Events per chunk (default 50k)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--chunk-id",   type=int, default=0,
                        help="Chunk index for SLURM array jobs")
    parser.add_argument("--setup-only", action="store_true",
                        help="Only create the MG5 process directory, then exit")
    parser.add_argument("--merge",      action="store_true",
                        help="Merge all chunk files into {process}.hdf5 and exit")
    parser.add_argument("--out-dir",    type=Path, default=None,
                        help="Override output directory (default: OUT_DIR constant)")
    parser.add_argument("--mg5-work",   type=Path, default=None,
                        help="Override MG5 working directory (default: MG5_WORK constant)")
    args = parser.parse_args()

    # Allow CLI overrides of the module-level path constants so smoke tests
    # can run in an isolated directory without touching the production paths.
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    if args.mg5_work is not None:
        MG5_WORK = args.mg5_work

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MG5_WORK.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_chunks(args.process)
    elif args.setup_only:
        setup_mg5_proc(args.process)
        print(f"Setup complete for {args.process}.")
    else:
        generate(
            args.process,
            args.nevents,
            seed=args.seed + args.chunk_id * 997,
            chunk_id=args.chunk_id,
        )
