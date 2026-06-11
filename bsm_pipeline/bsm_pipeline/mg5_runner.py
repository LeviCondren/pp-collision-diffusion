"""
MadGraph5 interface for arbitrary BSM models.

Accepts:
  - UFO model directory
  - Process definition string (MG5 syntax)
  - Parameter dictionary {name: value}
  - Number of events, energy cuts

Outputs LHE events (gz) at a specified path.
"""
from __future__ import annotations
import gzip, math, os, re, shutil, subprocess, tarfile, tempfile, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .event_formats import LHEEvent, Particle


# ── Paths (matching existing infrastructure) ──────────────────────────────────
MG5_BIN    = Path("/pscratch/sd/l/lcondren/MCsim/MG5_aMC_v3_6_7/bin/mg5_aMC")
HTOOLS     = "/pscratch/sd/l/lcondren/MCsim/MG5_aMC_v3_6_7/HEPTools"
PYTHON3    = "/pscratch/sd/l/lcondren/.conda/envs/mg5_new/bin/python3"
CONDA_BIN  = "/pscratch/sd/l/lcondren/.conda/envs/mg5_new/bin"
PYTHON_SHIM= "/pscratch/sd/l/lcondren/MCsim/python_shim"
PY8_IF     = f"{HTOOLS}/bin/MG5aMC_PY8_interface"

MG5_ENV = None   # initialised lazily


def _get_mg5_env() -> dict:
    global MG5_ENV
    if MG5_ENV is None:
        env = {k: v for k, v in os.environ.items() if k != 'PYTHIA8DATA'}
        env.update({
            'PATH': f"{PYTHON_SHIM}:{CONDA_BIN}:{os.environ.get('PATH','')}",
            'LD_LIBRARY_PATH': (f"{HTOOLS}/pythia8/lib:{HTOOLS}/lib:"
                                f"{os.environ.get('LD_LIBRARY_PATH','')}"),
            'PYTHONPATH': '',
            'PYTHIA8DATA': f"{HTOOLS}/pythia8/share/Pythia8/xmldoc",
            # Use system GCC-12 for HEPTools compilation (Ninja, CutTools, etc.).
            # conda gcc-14 produces a linker error (undefined __TMC_END__) when
            # building shared objects via libtool, breaking the Ninja install.
            'CC':  '/usr/bin/gcc-12',
            'CXX': '/usr/bin/g++-12',
            'FC':  '/usr/bin/gfortran-12',
        })
        MG5_ENV = env
    return MG5_ENV


class MG5Runner:
    """
    Run MadGraph5 for an arbitrary BSM process.

    Parameters
    ----------
    ufo_path : Path
        Directory containing the UFO model files (e.g. DarkPhoton_UFO/).
    process  : str
        MG5 process string, e.g. "p p > zp > e+ e-".
    work_dir : Path
        Scratch directory for MG5 output. Created if absent.
    """

    def __init__(self, ufo_path: Path, process: str, work_dir: Path):
        self.ufo_path = Path(ufo_path)
        self.process  = process
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._proc_dir: Optional[Path] = None

    # ── One-time process setup ─────────────────────────────────────────────────
    def setup_process(self, energy_gev: float = 13000., mmll_min: float = 4.,
                      mmll_max: float = 120., nb_core: int = 64,
                      force: bool = False) -> Path:
        """
        Generate the MG5 process directory (slow; only needed once per process).
        Returns the process directory path.
        """
        proc_dir = self.work_dir / "mg5_proc"
        if proc_dir.exists() and not force:
            self._proc_dir = proc_dir
            return proc_dir

        if proc_dir.exists():
            shutil.rmtree(proc_dir)

        model_name = self.ufo_path.name
        mg5_script = self.work_dir / "setup_proc.mg5"
        mg5_script.write_text(
            f"import model {self.ufo_path}\n"
            f"generate {self.process}\n"
            f"output {proc_dir}\n"
        )
        print(f"[MG5] Setting up process: {self.process}")
        r = subprocess.run(
            [PYTHON3, str(MG5_BIN), str(mg5_script)],
            capture_output=True, text=True, env=_get_mg5_env()
        )
        mg5_script.unlink(missing_ok=True)
        if not proc_dir.exists():
            raise RuntimeError(f"MG5 process setup failed:\n{r.stderr[-2000:]}")

        # Configure run card
        self._configure_run_card(proc_dir, n_events=1000, nb_core=nb_core,
                                 mmll_min=mmll_min, mmll_max=mmll_max,
                                 energy_gev=energy_gev)
        self._proc_dir = proc_dir
        print(f"[MG5] Process directory: {proc_dir}")
        return proc_dir

    # ── Per-run generation ─────────────────────────────────────────────────────
    def generate(self, params: Dict[str, float], n_events: int = 5000,
                 seed: int = 42, nb_core: int = 64,
                 run_tag: str = "run") -> Path:
        """
        Generate n_events LHE events for the given parameter point.

        Parameters
        ----------
        params : dict
            Parameter values to write into param_card.dat.
            Keys must match param card block entries.
        n_events : int
        seed : int
        nb_core : int
        run_tag : str
            Label for this run (used in filenames).

        Returns
        -------
        Path to the .lhe.gz file.
        """
        if self._proc_dir is None:
            self.setup_process(nb_core=nb_core)

        # Update param card
        self._write_params(self._proc_dir, params)

        # Update run card with n_events
        self._configure_run_card(self._proc_dir, n_events=n_events,
                                 nb_core=nb_core)

        # Clear old event directories
        ev_dir = self._proc_dir / "Events"
        ev_dir.mkdir(exist_ok=True)
        for d in ev_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                shutil.rmtree(d, ignore_errors=True)

        # Launch script
        script = self.work_dir / f"launch_{run_tag}.mg5"
        script.write_text(
            f"set nb_core {nb_core}\n"
            f"set iseed {seed}\n"
            f"launch {self._proc_dir}\n"
            f"shower=Pythia8\n"
            f"0\n"
        )
        print(f"[MG5] Generating {n_events} events  seed={seed}  params={params}")
        t0 = time.time()
        r = subprocess.run(
            [PYTHON3, str(MG5_BIN), str(script)],
            capture_output=True, text=True, env=_get_mg5_env()
        )
        script.unlink(missing_ok=True)
        print(f"[MG5] Done in {(time.time()-t0)/60:.1f} min  rc={r.returncode}")

        # Find output
        runs = sorted(p.name for p in ev_dir.iterdir()
                      if p.is_dir() and p.name.startswith("run_"))
        if not runs:
            raise RuntimeError(f"MG5 produced no output:\n{r.stderr[-2000:]}")

        rd = ev_dir / runs[0]
        lhe = next((p for p in rd.iterdir()
                    if 'unweighted_events' in p.name
                    and p.suffix in ('.gz', '.lhe')), None)
        if lhe is None:
            raise RuntimeError(f"LHE file not found in {rd}")
        return lhe

    # ── Gridpack-based generation (faster for repeated calls) ─────────────────
    def build_gridpack(self, base_params: Dict[str, float],
                       nb_core: int = 64) -> Path:
        """
        Build a gridpack (tarball) for fast repeated generation at varied params.
        Returns path to madevent.tar.gz.
        """
        if self._proc_dir is None:
            self.setup_process(nb_core=nb_core)
        self._write_params(self._proc_dir, base_params)
        script = self.work_dir / "build_gridpack.mg5"
        script.write_text(
            f"set nb_core {nb_core}\n"
            f"launch {self._proc_dir} -p\n"
            f"0\n"
        )
        r = subprocess.run(
            [PYTHON3, str(MG5_BIN), str(script)],
            capture_output=True, text=True, env=_get_mg5_env()
        )
        script.unlink(missing_ok=True)
        gp = self._proc_dir / "madevent.tar.gz"
        if not gp.exists():
            raise RuntimeError(f"Gridpack build failed:\n{r.stderr[-2000:]}")
        return gp

    def generate_from_gridpack(self, gridpack: Path, params: Dict[str, float],
                                n_events: int = 5000, seed: int = 42,
                                nb_core: int = 64) -> Path:
        """Extract gridpack, update param card, run. Returns LHE path."""
        work = Path(tempfile.mkdtemp(prefix="mg5gp_"))
        try:
            madevent_dir = work / "madevent"
            madevent_dir.mkdir()
            with tarfile.open(gridpack, "r:gz") as tf:
                tf.extractall(madevent_dir)

            # Update param card inside the extracted gridpack
            pc_candidates = list(madevent_dir.rglob("param_card.dat"))
            if pc_candidates:
                self._write_params_to_file(pc_candidates[0], params)

            run_sh = gridpack.parent / "bin" / "internal" / "Gridpack" / "run.sh"
            if not run_sh.exists():
                run_sh = next(madevent_dir.rglob("run.sh"), None)

            if run_sh is None:
                raise FileNotFoundError("run.sh not found in gridpack")

            r = subprocess.run(
                ["bash", str(run_sh), "-p", str(nb_core), str(n_events), str(seed)],
                cwd=work, capture_output=True, text=True,
                env=_get_mg5_env(), timeout=3600
            )
            lhe = work / "events.lhe.gz"
            if not lhe.exists():
                raise RuntimeError(f"Gridpack run failed:\n{r.stderr[-500:]}")
            # Move to persistent location before work dir cleanup
            dest = gridpack.parent / f"events_{seed}.lhe.gz"
            shutil.copy(lhe, dest)
            return dest
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # ── Internal helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _configure_run_card(proc_dir: Path, n_events: int = 5000,
                             nb_core: int = 64, mmll_min: float = 4.,
                             mmll_max: float = 120., energy_gev: float = 13000.):
        rc = proc_dir / "Cards" / "run_card.dat"
        if not rc.exists():
            return
        txt = rc.read_text()
        txt = re.sub(r'\d+\s*=\s*nevents\b', f'{n_events} = nevents', txt)
        txt = re.sub(r'\d+\s*=\s*nb_core\b',  f'{nb_core} = nb_core',  txt)
        txt = re.sub(r'True\s*=\s*use_syst\b', 'False = use_syst', txt,
                     flags=re.IGNORECASE)
        txt = re.sub(r'[\d.e+\-]+\s*=\s*mmll\b',    f'{mmll_min} = mmll',
                     txt, flags=re.IGNORECASE)
        txt = re.sub(r'[\d.e+\-]+\s*=\s*mmllmax\b', f'{mmll_max} = mmllmax',
                     txt, flags=re.IGNORECASE)
        if 'mmll' not in txt.lower():
            txt += f'\n{mmll_min} = mmll\n{mmll_max} = mmllmax\n'
        rc.write_text(txt)

        py8 = proc_dir / "Cards" / "pythia8_card.dat"
        if py8.exists():
            t = py8.read_text()
            t = re.sub(r'!?\s*partonlevel:mpi\s*=\s*\w+',
                       'PartonLevel:MPI = off', t, flags=re.IGNORECASE)
            if 'PartonLevel:MPI' not in t:
                t += '\nPartonLevel:MPI = off\n'
            py8.write_text(t)

    @staticmethod
    def _write_params(proc_dir: Path, params: Dict[str, float]):
        """Write parameter values into param_card.dat using regex substitution."""
        pc = proc_dir / "Cards" / "param_card.dat"
        if not pc.exists():
            raise FileNotFoundError(f"param_card.dat not found in {proc_dir / 'Cards'}")
        MG5Runner._write_params_to_file(pc, params)

    @staticmethod
    def _write_params_to_file(pc: Path, params: Dict[str, float]):
        txt = pc.read_text()
        for name, val in params.items():
            # Try block entry: "   <code>   <old_val>   # <name>"
            pattern = rf'(^\s*\d+\s+)[\d.e+\-]+(\s+#\s*{re.escape(name)}\b)'
            txt = re.sub(pattern, lambda m: m.group(1) + f'{val:.6e}' + m.group(2),
                         txt, flags=re.MULTILINE | re.IGNORECASE)
            # Try DECAY line: "DECAY <name> <old_val>"
            pattern2 = rf'(^DECAY\s+\S+\s+)[\d.e+\-]+(\s+#\s*{re.escape(name)}\b)'
            txt = re.sub(pattern2, lambda m: m.group(1) + f'{val:.6e}' + m.group(2),
                         txt, flags=re.MULTILINE | re.IGNORECASE)
        pc.write_text(txt)


# ── LHE parser ─────────────────────────────────────────────────────────────────
def parse_lhe(lhe_path: Path) -> List[LHEEvent]:
    """Parse a .lhe or .lhe.gz file into a list of LHEEvent objects.

    raw_text on each LHEEvent stores the verbatim lines between <event>…</event>
    so that _write_lhe can reproduce the exact color connections and spin info.
    """
    events = []
    opener = gzip.open if str(lhe_path).endswith('.gz') else open
    with opener(lhe_path, 'rt') as fh:
        in_event = False; header_done = False
        particles: List[Particle] = []; process_id = 0; weight = 1.0
        raw_lines: List[str] = []
        for raw in fh:
            line = raw.strip()
            if line == '<event>':
                in_event = True; header_done = False
                particles = []; process_id = 0; weight = 1.0
                raw_lines = []
            elif line == '</event>':
                events.append(LHEEvent(particles=particles,
                                       weight=weight, process_id=process_id,
                                       raw_text='\n'.join(raw_lines)))
                in_event = False
            elif in_event:
                raw_lines.append(raw.rstrip('\n'))
                if line.startswith('<') or line.startswith('#'):
                    continue
                parts = line.split()
                if not header_done:
                    if len(parts) >= 6:
                        try: weight = float(parts[2])
                        except ValueError: pass
                    header_done = True
                elif len(parts) >= 10:
                    try:
                        particles.append(Particle(
                            pid=int(parts[0]), status=int(parts[1]),
                            px=float(parts[6]), py=float(parts[7]),
                            pz=float(parts[8]), E=float(parts[9])
                        ))
                    except (ValueError, IndexError):
                        pass
    return events


def _write_lhe(events: List[LHEEvent], path: Path) -> None:
    """Write a list of LHEEvent objects to an uncompressed LHE file.

    Uses raw_text (captured by parse_lhe) when available so that color
    connections, spin correlations, and weights are preserved exactly.
    Falls back to approximate reconstruction for events without raw_text.
    """
    lines: List[str] = [
        '<LesHouchesEvents version="3.0">',
        '<header>',
        '<!-- Written by bsm_pipeline _write_lhe -->',
        '</header>',
        '<init>',
        # 13 TeV pp, NNPDF23 (id 247000), LO (weight strategy -4), 1 process
        '  2212  2212  6.5e+03  6.5e+03  0  0  247000  247000  -4  1',
        '  1.0  0.0  1.0  1',
        '</init>',
    ]
    for ev in events:
        lines.append('<event>')
        if ev.raw_text is not None:
            lines.append(ev.raw_text)
        else:
            # Approximate reconstruction — color connections may differ from original
            nparts = len(ev.particles)
            lines.append(
                f'  {nparts}   0  {ev.weight:.8e}  1.000e+03  7.546e-03  1.176e-01'
            )
            col = [501]
            def _nc() -> int:
                c = col[0]; col[0] += 1; return c
            for p in ev.particles:
                pid = p.pid
                if abs(pid) == 21:
                    c1, c2 = _nc(), _nc()
                elif pid > 0 and abs(pid) in {1, 2, 3, 4, 5, 6}:
                    c1, c2 = _nc(), 0
                elif pid < 0 and abs(pid) in {1, 2, 3, 4, 5, 6}:
                    c1, c2 = 0, _nc()
                else:
                    c1, c2 = 0, 0
                m1, m2 = (0, 0) if p.status != 1 else (1, 2)
                lines.append(
                    f'  {p.pid:5d}  {p.status:2d}  {m1:4d}  {m2:4d}'
                    f'  {c1:5d}  {c2:5d}'
                    f'  {p.px:+.10e}  {p.py:+.10e}  {p.pz:+.10e}'
                    f'  {p.E:.10e}  {p.mass:.5e}  0.  9.'
                )
        lines.append('</event>')
    lines.append('</LesHouchesEvents>')
    path.write_text('\n'.join(lines) + '\n')
