"""
Shower interface: LHE event → showered particle-level event.

Two backends:
  Pythia8Backend  — exact physics via MG5aMC_PY8_interface (works today)
  FPCDBackend     — GSGM Stage 2 via pre-trained JetNet checkpoints

All backends implement ShowerBackend.shower(lhe_path) → Path
and the higher-level shower_events(lhe_events) → List[ParticleEvent].
"""
from __future__ import annotations
import gzip, math, os, subprocess, tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import numpy as np

from .event_formats import LHEEvent, ParticleEvent, Particle
from .mg5_runner import _get_mg5_env, HTOOLS, parse_lhe

PY8_IF = f"{HTOOLS}/bin/MG5aMC_PY8_interface"


class ShowerBackend(ABC):
    @abstractmethod
    def shower(self, lhe_path: Path, seed: int = 42) -> Path:
        """Shower the LHE file. Returns path to output (HepMC or .fpcd.npz)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can run in the current environment."""

    def shower_events(self, lhe_events: List["LHEEvent"],
                      seed: int = 42) -> List["ParticleEvent"]:
        """Higher-level: shower a list of LHEEvent objects in memory.
        Default implementation writes a temp LHE file and calls shower().
        FPCDBackend overrides this for direct batch inference.
        """
        import tempfile
        from .mg5_runner import _write_lhe
        with tempfile.NamedTemporaryFile(suffix=".lhe", delete=False) as f:
            tmp = Path(f.name)
        try:
            _write_lhe(lhe_events, tmp)
            out = self.shower(tmp, seed=seed)
            return _read_shower_output(out)
        finally:
            tmp.unlink(missing_ok=True)


# ── Pythia8 backend ────────────────────────────────────────────────────────────
class Pythia8Backend(ShowerBackend):
    """
    Exact Pythia8 showering via MG5aMC_PY8_interface.
    MLM matching disabled for simple 2→2 processes; enable for MLM samples.
    """
    def __init__(self, isr: bool = True, mpi: bool = True, fsr: bool = True,
                 mlm_matching: bool = False, n_jet_max: int = 1):
        self.isr          = isr
        self.mpi          = mpi
        self.fsr          = fsr
        self.mlm_matching = mlm_matching
        self.n_jet_max    = n_jet_max

    def is_available(self) -> bool:
        return Path(PY8_IF).exists()

    def shower(self, lhe_path: Path, seed: int = 42) -> Path:
        # MG5aMC_PY8_interface command-file requirements (discovered empirically):
        #   - LHEFInputs:nSubruns must be set (default 0 → loop runs 0 times)
        #   - HEPMCoutput:file takes a raw path (hepmc@ is MG5 Python-level syntax)
        #   - Main:subrun=0 block must contain Beams:frameType + Beams:LHEF
        #   - Main:numberOfEvents must be ≥ 1 (-1 is out-of-range in Pythia8 8.3)
        #   - LHE must be uncompressed (Pythia8 may lack zlib on this installation)
        import gzip as _gz, shutil as _sh

        # Decompress LHE if needed
        if str(lhe_path).endswith('.gz'):
            lhe_plain = lhe_path.with_suffix('')
            with _gz.open(lhe_path, 'rb') as fi, open(lhe_plain, 'wb') as fo:
                _sh.copyfileobj(fi, fo)
        else:
            lhe_plain = lhe_path

        hepmc_path = lhe_path.with_suffix('').with_suffix('.hepmc')
        cmnd_path  = lhe_path.parent / "py8_shower.dat"

        matching_lines = []
        if self.mlm_matching:
            matching_lines = [
                "JetMatching:merge = on",
                "JetMatching:setMad = on",
                f"JetMatching:nJetMax = {self.n_jet_max}",
                "JetMatching:doShowerKt = off",
            ]

        lines = (
            [
                "LHEFInputs:nSubruns = 1",
                "Main:numberOfEvents = 100000",
                f"HEPMCoutput:file = {hepmc_path}",
                "Check:epTolErr = 1.0e-02",
                f"PartonLevel:ISR = {'on' if self.isr else 'off'}",
                f"PartonLevel:MPI = {'on' if self.mpi else 'off'}",
                f"PartonLevel:FSR = {'on' if self.fsr else 'off'}",
                f"Random:setSeed = on",
                f"Random:seed = {seed % 900000000}",
            ]
            + matching_lines
            + [
                "Main:subrun=0",
                "Beams:frameType=4",
                f"Beams:LHEF={lhe_plain}",
            ]
        )
        cmnd_path.write_text("\n".join(lines) + "\n")

        r = subprocess.run([PY8_IF, str(cmnd_path)],
                           capture_output=True, text=True,
                           env=_get_mg5_env(), timeout=7200)
        cmnd_path.unlink(missing_ok=True)

        if not hepmc_path.exists() or hepmc_path.stat().st_size < 1024:
            raise RuntimeError(
                f"Pythia8 shower failed (no/empty HepMC output):\n{r.stderr[-500:]}"
            )
        return hepmc_path


# ── FPCD backend ───────────────────────────────────────────────────────────────
# PDG ID → GSGM class label (matches JetNet training classes)
_PDG_TO_CLASS = {
    21: 0,                          # gluon
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1,  # light quarks (u/d/s/c/b)
    6: 2,                           # top
    24: 3,                          # W
    23: 4,                          # Z
}

# Default: treat unknown heavy particles as light-quark-like showers
def _pdg_to_class(pdg: int) -> int:
    return _PDG_TO_CLASS.get(abs(pdg), 1)


def _lhe_events_to_parton_array(lhe_events: List["LHEEvent"]) -> np.ndarray:
    """
    Convert a list of LHE events to a flat (N_partons, 7) array:
      [pT, eta, phi, mass, npart_est(=0), class_id, event_idx]

    Only hadronic final-state partons (quarks, gluons, W, Z, top) are included.
    Leptons/photons are skipped — they do not undergo QCD shower.
    event_idx is the index into lhe_events for reconstructing per-event structure.
    """
    _HADRONIC_PDGS = {1, 2, 3, 4, 5, 6, 21, 23, 24}
    rows = []
    for i, ev in enumerate(lhe_events):
        for p in ev.final_state:
            if abs(p.pid) not in _HADRONIC_PDGS:
                continue
            pT   = p.pT
            if pT < 1e-6:
                continue
            eta  = math.asinh(p.pz / pT) if pT > 0 else 0.
            phi  = math.atan2(p.py, p.px)
            mass = p.mass
            rows.append([pT, eta, phi, mass, 0., _pdg_to_class(p.pid), i])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 7), np.float32)


class FPCDBackend(ShowerBackend):
    """
    Fast shower surrogate: GSGM Stage 2 conditioned on LHE parton kinematics.

    Uses the pre-trained JetNet checkpoints from arXiv:2304.01266 (no retraining).
    Stage 1 (jet kinematic generation) is bypassed entirely; LHE parton 4-momenta
    are injected directly as jet-level conditioning for the particle diffusion model.

    Requires the vmikuni/tensorflow:ngc-22.08-tf2-v0 Shifter image on Perlmutter,
    or a local TF environment that matches the GSGM dependencies.

    Parameters
    ----------
    gsgm_dir    : path to the cloned GSGM repo (contains scripts/ and checkpoints_*)
    npart       : 30 or 150 (JetNet30 or JetNet150 model)
    distill     : use distilled 1-step model (faster, slightly lower quality)
    shifter_img : Shifter image string; set to None to run without Shifter (needs local TF)
    """

    SHIFTER_IMG  = "vmikuni/tensorflow:ngc-22.08-tf2-v0"
    INFER_SCRIPT = Path("/pscratch/sd/l/lcondren/MCsim/GSGM/scripts/infer_from_lhe.py")

    def __init__(self,
                 gsgm_dir: str = "/pscratch/sd/l/lcondren/MCsim/GSGM",
                 npart: int = 30,
                 distill: bool = True,
                 shifter_img: Optional[str] = SHIFTER_IMG):
        self.gsgm_dir   = Path(gsgm_dir)
        self.npart      = npart
        self.distill    = distill
        self.shifter_img = shifter_img

    def is_available(self) -> bool:
        if not self.INFER_SCRIPT.exists():
            return False
        ckpt_name = f"checkpoints_GSGM_v4{'_big' if self.npart == 150 else ''}"
        if self.distill:
            ckpt_name += "_d512"
        return (self.gsgm_dir / ckpt_name / "checkpoint").exists()

    def _build_cmd(self, input_path: Path, output_path: Path) -> List[str]:
        py_cmd = [
            "python", str(self.INFER_SCRIPT),
            "--input",          str(input_path),
            "--output",         str(output_path),
            "--npart",          str(self.npart),
            "--checkpoint-dir", str(self.gsgm_dir),
        ]
        if self.distill:
            py_cmd += ["--distill", "--factor", "512"]

        if self.shifter_img:
            return ["shifter",
                    f"--image={self.shifter_img}",
                    "--module=gpu"] + py_cmd
        return py_cmd

    def shower(self, lhe_path: Path, seed: int = 42) -> Path:
        """Run FPCD inference on all partons in the LHE file."""
        lhe_events = parse_lhe(lhe_path)
        return self._run_on_events(lhe_events, lhe_path.with_suffix('.fpcd.npz'))

    def shower_events(self, lhe_events: List["LHEEvent"],
                      seed: int = 42) -> List["ParticleEvent"]:
        """Direct batch inference on in-memory LHE events (no temp LHE file needed)."""
        with tempfile.NamedTemporaryFile(suffix=".fpcd.npz", delete=False) as f:
            out_path = Path(f.name)
        try:
            self._run_on_events(lhe_events, out_path)
            return parse_fpcd_npz(out_path)
        finally:
            out_path.unlink(missing_ok=True)

    def _run_on_events(self, lhe_events: List["LHEEvent"],
                       out_path: Path) -> Path:
        """Core: parton array → infer_from_lhe.py → per-event npz."""
        parton_arr = _lhe_events_to_parton_array(lhe_events)

        if len(parton_arr) == 0:
            # No hadronic partons (e.g. pure dilepton event) — return empty
            np.savez(out_path,
                     particles=np.zeros((len(lhe_events), 0, 4), np.float32),
                     event_idx=np.zeros(0, np.int32),
                     n_events=np.array([len(lhe_events)]))
            return out_path

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            in_tmp = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            out_tmp = Path(f.name)

        try:
            # Save parton array (drop event_idx column before passing to script)
            np.save(in_tmp, parton_arr[:, :6])

            cmd = self._build_cmd(in_tmp, out_tmp)
            r = subprocess.run(cmd, capture_output=False, text=True, timeout=3600)
            if r.returncode != 0:
                raise RuntimeError(f"infer_from_lhe.py failed (rc={r.returncode})")
            if not out_tmp.exists():
                raise RuntimeError("infer_from_lhe.py produced no output file")

            particles = np.load(out_tmp)   # (N_partons, npart, 4)
            event_idx = parton_arr[:, 6].astype(np.int32)

            np.savez(out_path,
                     particles=particles,
                     event_idx=event_idx,
                     n_events=np.array([len(lhe_events)]))
        finally:
            in_tmp.unlink(missing_ok=True)
            out_tmp.unlink(missing_ok=True)

        return out_path


# ── Output parsers ─────────────────────────────────────────────────────────────

def _read_shower_output(path: Path) -> List[ParticleEvent]:
    """Dispatch to the right parser based on file extension."""
    if str(path).endswith('.fpcd.npz'):
        return parse_fpcd_npz(path)
    return parse_hepmc(path)


def parse_fpcd_npz(path: Path) -> List[ParticleEvent]:
    """
    Parse a .fpcd.npz file (written by FPCDBackend) into a list of ParticleEvent.

    The npz contains:
      particles  : (N_partons, n_part, 4) float32  [px, py, pz, E] per constituent
      event_idx  : (N_partons,) int32   which LHE event each parton belongs to
      n_events   : scalar int            total number of LHE events

    All constituents from all partons in the same LHE event are merged into one
    ParticleEvent (i.e., the full showered final state per event).
    """
    data      = np.load(path)
    particles = data["particles"]   # (N_partons, n_part, 4)
    evt_idx   = data["event_idx"]   # (N_partons,)
    n_events  = int(data["n_events"][0])

    events: List[ParticleEvent] = [ParticleEvent() for _ in range(n_events)]

    for parton_i, ev_i in enumerate(evt_idx):
        cloud = particles[parton_i]   # (n_part, 4)
        for constituent in cloud:
            px, py, pz, E = float(constituent[0]), float(constituent[1]), \
                            float(constituent[2]), float(constituent[3])
            if E < 1e-6:
                continue   # zero-padded slot
            events[ev_i].particles.append(
                Particle(pid=211, status=1,  # generic charged hadron
                         px=px, py=py, pz=pz, E=E))

    return events


# ── HepMC parser ───────────────────────────────────────────────────────────────
def parse_hepmc(hepmc_path: Path,
                pid_filter: Optional[List[int]] = None) -> List[ParticleEvent]:
    """
    Parse a HepMC file into a list of ParticleEvent objects.

    Parameters
    ----------
    hepmc_path  : path to .hepmc or .hepmc.gz
    pid_filter  : if given, keep only particles with these PIDs (final state only)
    """
    events: List[ParticleEvent] = []
    opener = gzip.open if str(hepmc_path).endswith('.gz') else open

    with opener(hepmc_path, 'rt') as fh:
        current: Optional[List[Particle]] = None
        for raw in fh:
            line = raw.strip()
            if line.startswith('E '):
                if current is not None:
                    events.append(ParticleEvent(particles=current))
                current = []
            elif line.startswith('P ') and current is not None:
                parts = line.split()
                if len(parts) < 9:
                    continue
                pid    = int(parts[2])
                status = int(parts[8]) if len(parts) > 8 else 1
                if status != 1:
                    continue
                if pid_filter and pid not in pid_filter:
                    continue
                current.append(Particle(
                    pid=pid, status=1,
                    px=float(parts[3]), py=float(parts[4]),
                    pz=float(parts[5]), E=float(parts[6])
                ))
    if current is not None:
        events.append(ParticleEvent(particles=current))
    return events


# ── Dilepton kinematics extractor ──────────────────────────────────────────────
def extract_dilepton_kinematics(events: List[ParticleEvent],
                                pid_pos: int, pid_neg: int) -> np.ndarray:
    """
    Extract 5D dilepton kinematics [log_m, cosθ_CS, φ, log(pT+1), y]
    from a list of ParticleEvent objects.  Uses the correct Collins-Soper formula.
    """
    rows = []
    sqr2 = math.sqrt(2.)
    for ev in events:
        pos = next((p for p in ev.particles if p.pid == pid_pos), None)
        neg = next((p for p in ev.particles if p.pid == pid_neg), None)
        if pos is None or neg is None:
            continue
        E  = pos.E + neg.E;   px = pos.px + neg.px
        py = pos.py + neg.py; pz = pos.pz + neg.pz
        m2 = E**2 - px**2 - py**2 - pz**2
        if m2 <= 0:
            continue
        m  = math.sqrt(m2); pT = math.sqrt(px**2 + py**2)
        y  = 0.5*math.log((E+pz)/(E-pz)) if (E+pz)>0 and (E-pz)>0 else 0.
        # Collins-Soper cosθ*
        p_plus_pos  = (pos.E + pos.pz) / sqr2
        p_minus_pos = (pos.E - pos.pz) / sqr2
        p_plus_neg  = (neg.E + neg.pz) / sqr2
        p_minus_neg = (neg.E - neg.pz) / sqr2
        pT_sq  = px**2 + py**2
        cs_num = 2. * (p_plus_pos * p_minus_neg - p_minus_pos * p_plus_neg)
        cs_den = m * math.sqrt(m**2 + pT_sq)
        sign_pz = 1. if pz >= 0. else -1.
        costh  = max(-1., min(1., sign_pz * cs_num / cs_den)) if cs_den > 0. else 0.
        phi    = math.atan2(pos.py, pos.px)
        rows.append([math.log(max(m, 1e-8)), costh, phi, math.log(pT+1.), y])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def make_shower_backend(backend: str = "pythia8", **kwargs) -> ShowerBackend:
    """Factory: return the requested shower backend."""
    if backend == "pythia8":
        return Pythia8Backend(**kwargs)
    elif backend == "fpcd":
        return FPCDBackend(**kwargs)
    else:
        raise ValueError(f"Unknown shower backend: {backend!r}. "
                         f"Choose 'pythia8' or 'fpcd'.")
