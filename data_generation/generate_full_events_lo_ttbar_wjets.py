"""
generate_full_events_lo_ttbar_wjets.py — LO MG5+Pythia8 matched pairs for ttbar and wjets.

Generates 500k events per process at leading order.  Output HDF5 schema is
identical to the NLO files (event_weights, n_partons fields included) so that
LO and NLO data can be combined or compared directly.

Parton feature layout (N, 6, 6) — MAX_PARTONS=6 (2 initial + 3 final + 1 boson).
At LO, ttbar has n_partons=4 (t/tbar with correct PDG norms 11/16, 12/16) and
wjets has n_partons=4 (1 jet + W boson slot with full LHE kinematics).

Outputs:
  /pscratch/sd/l/lcondren/MCsim/full_event_fpcd_more/ttbar.hdf5
  /pscratch/sd/l/lcondren/MCsim/full_event_fpcd_more/wjets.hdf5

Usage (same pattern as other generation scripts):
  python generate_full_events_lo_ttbar_wjets.py --process ttbar --setup-only
  python generate_full_events_lo_ttbar_wjets.py --process wjets --setup-only

  python generate_full_events_lo_ttbar_wjets.py \\
      --process ttbar --nevents 50000 --chunk-id $SLURM_ARRAY_TASK_ID

  python generate_full_events_lo_ttbar_wjets.py --process ttbar --merge
  python generate_full_events_lo_ttbar_wjets.py --process wjets --merge
"""

import sys, os, math, time, argparse, gzip, shutil, subprocess, re
from pathlib import Path
import numpy as np
import h5py
import pythia8

BSM_PIPELINE = "/global/u2/l/lcondren/ContinuousParamFit/bsm_pipeline"
sys.path.insert(0, BSM_PIPELINE)
from bsm_pipeline.mg5_runner import parse_lhe, _get_mg5_env, MG5_BIN, PYTHON3

# ── Constants ──────────────────────────────────────────────────────────────────
N_PART_MAX    = 500
N_PART_FEAT   = 7
MAX_PARTONS   = 6     # 2 initial + up to 3 final colored + 1 hard boson slot
N_PARTON_FEAT = 6
ETA_MAX       = 5.0
PT_MIN        = 0.3

OUT_DIR  = Path("/pscratch/sd/l/lcondren/MCsim/full_event_fpcd_more")
MG5_WORK = Path("/pscratch/sd/l/lcondren/MCsim/full_event_fpcd_more/mg5_work")

_COLORED   = {1, 2, 3, 4, 5, 6, 21}
# PDG class index ÷16: g=0, d/dbar=1/2, u/ubar=3/4, s/sbar=5/6, c/cbar=7/8,
#   b/bbar=9/10, t/tbar=11/12, Z=13, W+=14, W-=15
_PDG_CLASS = {21: 0, 1: 1, -1: 2, 2: 3, -2: 4, 3: 5, -3: 6, 4: 7, -4: 8, 5: 9, -5: 10,
              6: 11, -6: 12, 23: 13, 24: 14, -24: 15}

def _pdg_norm(pid):
    return float(_PDG_CLASS.get(pid, 0)) / 16.0

def _pid_cat(abs_pid, charge):
    if abs_pid == 22:     return 2
    if abs_pid == 11:     return 3
    if abs_pid == 13:     return 4
    if abs(charge) > 0.5: return 0
    return 1


# ── MG5 LO process setup ───────────────────────────────────────────────────────

def _configure_run_card(proc_dir, process):
    rc = proc_dir / "Cards" / "run_card.dat"
    if not rc.exists():
        return
    txt = rc.read_text()
    txt = re.sub(r'True\s*=\s*use_syst\b', 'False = use_syst', txt, flags=re.IGNORECASE)
    if process == "wjets":
        txt = re.sub(r'[\d.]+\s*=\s*ptj\b', '10.0 = ptj', txt, flags=re.IGNORECASE)
        if not re.search(r'\bptj\b', txt, re.IGNORECASE):
            txt += '\n10.0 = ptj\n'
    rc.write_text(txt)


def setup_mg5_proc(process):
    MG5_WORK.mkdir(parents=True, exist_ok=True)
    proc_dir = MG5_WORK / f"{process}_lo_proc"
    if proc_dir.exists():
        print(f"[MG5-LO] Process dir already exists: {proc_dir}")
        return proc_dir

    _5f = ("define p = g u c d s b u~ c~ d~ s~ b~\n"
           "define j = g u c d s b u~ c~ d~ s~ b~\n")

    if process == "ttbar":
        gen_lines = _5f + "generate p p > t t~"
    elif process == "wjets":
        gen_lines = _5f + "generate p p > w+ j\nadd process p p > w- j"
    else:
        raise ValueError(f"Unknown process: {process}")

    script = MG5_WORK / f"setup_lo_{process}.mg5"
    script.write_text(f"{gen_lines}\noutput {proc_dir}\n")

    print(f"[MG5-LO] Setting up {process} (~5 min)...")
    r = subprocess.run(
        [PYTHON3, str(MG5_BIN), str(script)],
        capture_output=True, text=True,
        env=_get_mg5_env(), timeout=900,
    )
    script.unlink(missing_ok=True)

    if not proc_dir.exists():
        raise RuntimeError(
            f"MG5 LO setup failed for {process}:\n"
            f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-2000:]}"
        )

    _configure_run_card(proc_dir, process)
    print(f"[MG5-LO] Process dir created: {proc_dir}")
    return proc_dir


# ── MG5 LHE generation ─────────────────────────────────────────────────────────

def run_mg5_lhe(process, n_events, seed):
    template_dir = setup_mg5_proc(process)

    proc_dir = MG5_WORK / f"{process}_lo_proc_{seed}"
    if proc_dir.exists():
        shutil.rmtree(proc_dir)
    shutil.copytree(template_dir, proc_dir, symlinks=True)

    try:
        rc = proc_dir / "Cards" / "run_card.dat"
        if rc.exists():
            txt = rc.read_text()
            txt = re.sub(r'\d+\s*=\s*nevents\b', f'{n_events} = nevents', txt)
            txt = re.sub(r'\d+\s*=\s*iseed\b',   f'{seed} = iseed',       txt)
            rc.write_text(txt)

        ev_dir = proc_dir / "Events"
        ev_dir.mkdir(exist_ok=True)

        launch = MG5_WORK / f"launch_lo_{process}_{seed}.mg5"
        launch.write_text(f"launch {proc_dir}\n0\n")
        t0 = time.time()
        r = subprocess.run(
            [PYTHON3, str(MG5_BIN), str(launch)],
            capture_output=True, text=True,
            env=_get_mg5_env(), timeout=7200,
        )
        launch.unlink(missing_ok=True)
        print(f"[MG5-LO] {process} n={n_events} seed={seed}  rc={r.returncode}  "
              f"{(time.time()-t0)/60:.1f} min")

        runs = sorted(
            (d for d in ev_dir.iterdir() if d.is_dir() and d.name.startswith("run_")),
            key=lambda d: d.name,
        )
        lhe_gz = None
        for run_d in reversed(runs):
            lhe_gz = next(
                (p for p in run_d.iterdir() if "unweighted_events" in p.name), None)
            if lhe_gz:
                break

        if lhe_gz is None:
            raise RuntimeError(
                f"[MG5-LO] No LHE file found for {process} seed={seed}.\n"
                f"stdout:\n{r.stdout[-500:]}\nstderr:\n{r.stderr[-1000:]}"
            )

        if str(lhe_gz).endswith(".gz"):
            lhe_plain = MG5_WORK / f"{process}_lo_{seed}.lhe"
            with gzip.open(lhe_gz, "rb") as fi, open(lhe_plain, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            return lhe_plain
        return lhe_gz

    finally:
        shutil.rmtree(proc_dir, ignore_errors=True)


# ── Parton feature extraction ──────────────────────────────────────────────────

def _parton_features(ev, process: str):
    """
    Extract (MAX_PARTONS, N_PARTON_FEAT) and n_partons from one LHE event.

    Slot ordering:
      0   — initial-state parton from beam+ (pz > 0)
      1   — initial-state parton from beam- (pz < 0)
      2–4 — up to 3 hardest final-state colored partons
      5   — hard boson slot: W+/W- for wjets (with real kinematics from LHE);
            zero-padded for ttbar

    At LO:
      ttbar : n_partons = 4  (2 initial + t + t~; correct PDG norms 11/16, 12/16)
      wjets : n_partons = 4  (2 initial + 1 jet + W boson with kinematics)
    """
    arr = np.zeros((MAX_PARTONS, N_PARTON_FEAT), dtype=np.float32)

    init_partons = [p for p in ev.particles if p.status == -1 and abs(p.pid) != 2212]
    init_partons.sort(key=lambda p: p.pz, reverse=True)

    for i, p in enumerate(init_partons[:2]):
        E = max(abs(p.E), 1e-3)
        arr[i, 0] = math.log(E)
        arr[i, 1] = 0.0
        arr[i, 2] = 0.0
        arr[i, 3] = float(np.clip(p.pz / E, -1 + 1e-6, 1 - 1e-6))
        arr[i, 4] = _pdg_norm(p.pid)
        arr[i, 5] = 1.0

    final_colored = [p for p in ev.final_state if abs(p.pid) in _COLORED]
    final_colored.sort(key=lambda p: p.pT, reverse=True)

    # Leave slot 5 for the boson; allow up to 3 colored slots (2–4)
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

    # wjets: W+/W- is status=1 in the LHE with real kinematics
    if process == 'wjets':
        w_particles = [p for p in ev.final_state if abs(p.pid) == 24]
        if w_particles and n_partons < MAX_PARTONS:
            w = w_particles[0]
            E_W  = max(abs(w.E), 1e-3)
            pT_W = math.sqrt(max(w.px**2 + w.py**2, 1e-12))
            s = n_partons
            arr[s, 0] = math.log(E_W)
            arr[s, 1] = w.py / pT_W
            arr[s, 2] = w.px / pT_W
            arr[s, 3] = float(np.clip(w.pz / E_W, -1 + 1e-6, 1 - 1e-6))
            arr[s, 4] = _pdg_norm(w.pid)
            arr[s, 5] = 1.0
            n_partons += 1

    return arr, n_partons


# ── Pythia8 particle extraction ─────────────────────────────────────────────────

def _parse_pythia_event(pythia):
    event = pythia.event
    etas, sin_phis, cos_phis, log_pts, pid_cats, charges = [], [], [], [], [], []

    for i in range(event.size()):
        p = event[i]
        if not p.isFinal():
            continue
        abs_pid = abs(p.id())
        if abs_pid in {12, 14, 16}:
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
    arr[:n_fill, 6] = 1.0
    return arr


# ── LHE shower loop ─────────────────────────────────────────────────────────────

def shower_lhe(lhe_path, seed, process):
    lhe_events = parse_lhe(lhe_path)
    n_lhe = len(lhe_events)
    print(f"  Parsed {n_lhe} LHE events from {lhe_path.name}")

    py = pythia8.Pythia("", False)
    for s in [
        "Print:quiet = on",
        "Beams:frameType = 4",
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
            continue

        parton_arr, n_par = _parton_features(lhe_events[evt_idx], process)
        parts_list.append(arr)
        partons_list.append(parton_arr)
        npartons_list.append(n_par)

        if len(parts_list) % 5000 == 0:
            elapsed = time.time() - t0
            rate    = len(parts_list) / max(elapsed, 1e-9)
            eta_min = (n_lhe - evt_idx) / max(rate, 1e-9) / 60.0
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


# ── Main generation ─────────────────────────────────────────────────────────────

def generate(process, n_events, seed, chunk_id):
    print(f"\n[{process}-LO] chunk={chunk_id}  n_events={n_events}  seed={seed}")

    print(f"[{process}-LO] Running MG5...")
    lhe_path = run_mg5_lhe(process, n_events, seed)

    print(f"[{process}-LO] Showering with Pythia8...")
    parts, partons, npartons = shower_lhe(lhe_path, seed, process)

    lhe_path.unlink(missing_ok=True)

    n_saved  = len(parts)
    weights  = np.ones(n_saved, dtype=np.float32)   # LO: all weights +1
    fname    = OUT_DIR / f"{process}_lo_chunk{chunk_id:04d}.hdf5"
    with h5py.File(fname, "w") as f:
        f.create_dataset("particle_features", data=parts,    compression="gzip")
        f.create_dataset("parton_features",   data=partons,  compression="gzip")
        f.create_dataset("n_partons",         data=npartons, compression="gzip")
        f.create_dataset("event_weights",     data=weights,  compression="gzip")
    print(f"[{process}-LO] chunk={chunk_id}: Saved {n_saved} events → {fname}")

    npar_unique, npar_counts = np.unique(npartons, return_counts=True)
    for v, c in zip(npar_unique, npar_counts):
        print(f"  n_partons={v}: {c:,} events ({100*c/max(n_saved,1):.1f}%)")
    return fname


# ── Merge chunks ─────────────────────────────────────────────────────────────────

def merge_chunks(process):
    files = sorted(OUT_DIR.glob(f"{process}_lo_chunk*.hdf5"))
    if not files:
        print(f"No LO chunk files found for {process}")
        return

    all_parts, all_partons, all_npartons, all_weights = [], [], [], []

    for fpath in files:
        with h5py.File(fpath, "r") as h:
            all_parts.append(h["particle_features"][:])
            all_partons.append(h["parton_features"][:])
            if "n_partons" in h:
                all_npartons.append(h["n_partons"][:])
            if "event_weights" in h:
                all_weights.append(h["event_weights"][:])

    parts   = np.concatenate(all_parts,   axis=0)
    partons = np.concatenate(all_partons, axis=0)

    out = OUT_DIR / f"{process}.hdf5"
    with h5py.File(out, "w") as f:
        f.create_dataset("particle_features", data=parts,   compression="gzip")
        f.create_dataset("parton_features",   data=partons, compression="gzip")
        if all_npartons:
            npartons = np.concatenate(all_npartons, axis=0)
            f.create_dataset("n_partons",     data=npartons, compression="gzip")
        weights = (np.concatenate(all_weights, axis=0) if all_weights
                   else np.ones(len(parts), dtype=np.float32))
        f.create_dataset("event_weights",     data=weights,  compression="gzip")

    print(f"Merged {len(files)} chunks → {out}  ({parts.shape[0]:,} events)")
    for fpath in files:
        fpath.unlink()


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LO MG5+Pythia8 matched pairs: ttbar, wjets")
    parser.add_argument("--process",    choices=["ttbar", "wjets"], required=True)
    parser.add_argument("--nevents",    type=int, default=50_000)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--chunk-id",   type=int, default=0)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--merge",      action="store_true")
    parser.add_argument("--out-dir",    type=Path, default=None)
    parser.add_argument("--mg5-work",   type=Path, default=None)
    args = parser.parse_args()

    if args.out_dir  is not None: OUT_DIR  = args.out_dir
    if args.mg5_work is not None: MG5_WORK = args.mg5_work

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MG5_WORK.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_chunks(args.process)
    elif args.setup_only:
        setup_mg5_proc(args.process)
        print(f"LO setup complete for {args.process}.")
    else:
        generate(
            args.process,
            args.nevents,
            seed=args.seed + args.chunk_id * 997,
            chunk_id=args.chunk_id,
        )
