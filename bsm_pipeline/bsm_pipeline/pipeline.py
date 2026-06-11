"""
End-to-end BSM simulation pipeline.

Chains:
  MG5Runner → ShowerBackend → DetectorBackend

Accepts a PipelineConfig and produces numpy arrays + optional ROOT files.
"""
from __future__ import annotations
import json, shutil, time
from itertools import product
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import numpy as np

from .config import PipelineConfig
from .mg5_runner import MG5Runner, parse_lhe
from .shower import (make_shower_backend, parse_hepmc, parse_fpcd_npz,
                     _read_shower_output, extract_dilepton_kinematics)
from .detector import make_detector_backend


class SimulationResult:
    """Holds all outputs from one pipeline run (one parameter point)."""

    def __init__(self, params: dict, tag: str, out_dir: Path):
        self.params   = params
        self.tag      = tag
        self.out_dir  = out_dir
        self.lhe_path: Optional[Path]   = None
        self.hepmc_path: Optional[Path] = None
        self.root_path: Optional[Path]  = None
        # Kinematics as numpy arrays
        self.z_lhe:   Optional[np.ndarray] = None  # (N, 5) parton-level
        self.z_truth: Optional[np.ndarray] = None  # (N, 5) showered
        self.reco_events = None                     # list of RecoEvent

    def save_numpy(self):
        """Save z_lhe and z_truth arrays to out_dir."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.z_lhe is not None:
            np.save(self.out_dir / f"z_lhe_{self.tag}.npy", self.z_lhe)
        if self.z_truth is not None:
            np.save(self.out_dir / f"z_truth_{self.tag}.npy", self.z_truth)
        np.save(self.out_dir / f"theta_{self.tag}.npy",
                np.array(list(self.params.values()), dtype=np.float32))
        meta = {"params": self.params, "tag": self.tag,
                "n_lhe": int(len(self.z_lhe)) if self.z_lhe is not None else 0,
                "n_truth": int(len(self.z_truth)) if self.z_truth is not None else 0}
        (self.out_dir / f"meta_{self.tag}.json").write_text(
            json.dumps(meta, indent=2))

    def summary(self) -> str:
        lines = [f"[{self.tag}] params={self.params}"]
        if self.z_lhe is not None:
            lines.append(f"  z_lhe:   {self.z_lhe.shape}")
        if self.z_truth is not None:
            lines.append(f"  z_truth: {self.z_truth.shape}")
        return "\n".join(lines)


class BSMPipeline:
    """
    Full BSM simulation pipeline.

    Usage
    -----
    pipeline = BSMPipeline.from_config("my_config.yaml")
    results  = pipeline.run()           # single parameter point
    # or
    results  = list(pipeline.scan())    # parameter scan
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.work_dir.mkdir(parents=True, exist_ok=True)

        self.mg5 = MG5Runner(
            ufo_path=cfg.ufo_path,
            process=cfg.process,
            work_dir=cfg.work_dir,
        )
        self.shower_backend   = make_shower_backend(cfg.shower_backend,
                                                    **cfg.shower_kwargs)
        self.detector_backend = (make_detector_backend(cfg.detector_backend,
                                                        **cfg.detector_kwargs)
                                 if cfg.run_detector else None)

        # Check backends
        if not self.shower_backend.is_available():
            raise RuntimeError(
                f"Shower backend '{cfg.shower_backend}' is not available. "
                "Check that MG5aMC_PY8_interface exists or provide a trained FPCD model."
            )
        if cfg.run_detector and self.detector_backend and \
                not self.detector_backend.is_available():
            print(f"[WARNING] Detector backend '{cfg.detector_backend}' is not available. "
                  "Detector simulation will be skipped.")
            self.detector_backend = None

        # Dilepton channel PIDs (inferred from process string or set manually)
        self._pid_pos, self._pid_neg = self._infer_pids(cfg.process)

    # ── Public API ─────────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, path) -> "BSMPipeline":
        return cls(PipelineConfig.from_yaml(path))

    def run(self, params: Optional[Dict[str, float]] = None,
            tag: str = "run", seed: Optional[int] = None) -> SimulationResult:
        """
        Run the full pipeline for a single parameter point.

        Parameters
        ----------
        params : dict, optional
            Override the parameters from config.
        tag    : str
            Label for output files.
        seed   : int, optional
            RNG seed (defaults to config value).
        """
        cfg    = self.cfg
        params = params or cfg.parameters
        seed   = seed   or cfg.seed

        result = SimulationResult(params, tag, cfg.output_dir / tag)
        t_total = time.time()

        # ── Step 1: MG5 generation ────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"[Pipeline] Step 1: MG5 generation  tag={tag}")
        print(f"  process : {cfg.process}")
        print(f"  params  : {params}")
        print(f"  n_events: {cfg.n_events}")

        if cfg.use_gridpack:
            gp = cfg.work_dir / "gridpack" / "madevent.tar.gz"
            if not gp.exists():
                print("[Pipeline]   Building gridpack (one-time) ...")
                gp = self.mg5.build_gridpack(params, nb_core=cfg.nb_core)
            lhe_path = self.mg5.generate_from_gridpack(
                gp, params, n_events=cfg.n_events, seed=seed, nb_core=cfg.nb_core)
        else:
            lhe_path = self.mg5.generate(
                params, n_events=cfg.n_events, seed=seed,
                nb_core=cfg.nb_core, run_tag=tag)

        result.lhe_path = lhe_path

        # Parse LHE for z_lhe
        lhe_events = parse_lhe(lhe_path)
        from .shower import extract_dilepton_kinematics as _ek
        # Build shower-level ParticleEvent stubs from LHE (for LHE kinematics)
        from .event_formats import ParticleEvent
        lhe_as_pe = [ParticleEvent(particles=ev.final_state) for ev in lhe_events]
        result.z_lhe = _ek(lhe_as_pe, self._pid_pos, self._pid_neg)
        print(f"  z_lhe: {result.z_lhe.shape}")

        # Save LHE if requested
        if cfg.save_lhe:
            dest = cfg.output_dir / tag / lhe_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(lhe_path, dest)
            result.lhe_path = dest

        # ── Step 2: Shower ────────────────────────────────────────────────────
        print(f"\n[Pipeline] Step 2: Shower  ({cfg.shower_backend})")
        shower_out = self.shower_backend.shower(lhe_path, seed=seed)
        result.hepmc_path = shower_out

        particle_events = _read_shower_output(shower_out)
        result.z_truth = _ek(particle_events, self._pid_pos, self._pid_neg)
        print(f"  z_truth: {result.z_truth.shape}")

        if not cfg.save_hepmc:
            shower_out.unlink(missing_ok=True)

        # ── Step 3: Detector (optional) ───────────────────────────────────────
        if self.detector_backend is not None:
            print(f"\n[Pipeline] Step 3: Detector  ({cfg.detector_backend})")
            try:
                # Preferred path: pass in-memory particle events directly
                result.reco_events = self.detector_backend.simulate_from_particles(
                    particle_events)
                print(f"  reco events: {len(result.reco_events)}")
            except NotImplementedError:
                # Fallback: file-based interface (Delphes)
                det_out = self.detector_backend.simulate(shower_out)
                result.root_path = det_out
                if hasattr(self.detector_backend, "read_root_to_reco"):
                    result.reco_events = self.detector_backend.read_root_to_reco(det_out)
                    print(f"  reco events: {len(result.reco_events)}")

        # ── Save ──────────────────────────────────────────────────────────────
        if cfg.save_numpy:
            result.save_numpy()

        print(f"\n[Pipeline] Done in {(time.time()-t_total)/60:.1f} min")
        print(result.summary())
        return result

    def scan(self) -> Iterator[SimulationResult]:
        """
        Run the pipeline over a parameter grid defined in config.parameter_scan.

        Yields one SimulationResult per grid point.

        Example config:
          parameter_scan:
            MAp: [10, 20, 50, 100]
            gV:  [0.01, 0.038, 0.1]
        """
        scan = self.cfg.parameter_scan
        if scan is None:
            yield self.run()
            return

        base_params = dict(self.cfg.parameters)
        names  = list(scan.keys())
        values = [scan[k] for k in names]

        for i, combo in enumerate(product(*values)):
            params = dict(base_params)
            params.update(dict(zip(names, combo)))
            tag  = "scan_" + "_".join(f"{n}{v:.3g}" for n, v in zip(names, combo))
            seed = self.cfg.seed + i
            yield self.run(params=params, tag=tag, seed=seed)

    # ── Internal ───────────────────────────────────────────────────────────────
    @staticmethod
    def _infer_pids(process: str) -> Tuple[int, int]:
        """
        Guess the lepton pair PIDs from the process string.
        Returns (pid_positive_lepton, pid_negative_lepton).
        """
        s = process.lower()
        if "mu+" in s or "mu-" in s or "mu+ mu-" in s:
            return 13, -13
        elif "e+" in s or "e-" in s or "e+ e-" in s:
            return -11, 11
        elif "ta+" in s or "ta-" in s:
            return -15, 15
        else:
            # Default to electrons; can be overridden manually
            print("[Pipeline] WARNING: could not infer lepton PIDs from process string. "
                  "Defaulting to e+e-. Set pid_pos / pid_neg manually if wrong.")
            return -11, 11
