"""
Detector simulation interface: particle-level → detector-level observables.

Three backends:
  DelphesBackend    — exact fast detector via Delphes
  ParnassusBackend  — pre-trained Parnassus flow-matching surrogate
  MLDetectorBackend — stub for future custom trained models
"""
from __future__ import annotations
import os, subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import numpy as np

from .event_formats import ParticleEvent, RecoEvent, Particle
from .mg5_runner import _get_mg5_env, HTOOLS


DELPHES_BIN = Path("/pscratch/sd/l/lcondren/MCsim/MG5_aMC_v3_6_7/Delphes/DelphesHepMC2")
DELPHES_CARD_DEFAULT = Path(
    "/pscratch/sd/l/lcondren/MCsim/MG5_aMC_v3_6_7/Delphes/cards/delphes_card_ATLAS.tcl"
)

_PARNASSUS_DIR = Path("/pscratch/sd/l/lcondren/MCsim/Parnassus")
_PARNASSUS_PYTHON = Path(
    "/pscratch/sd/l/lcondren/.conda/envs/pipeline_copy-gpu2/bin/python3"
)


class DetectorBackend(ABC):
    @abstractmethod
    def simulate(self, shower_out: Path) -> Path:
        """Run detector simulation on a shower output file. Returns output path."""

    @abstractmethod
    def is_available(self) -> bool:
        pass

    def simulate_from_particles(
        self, particle_events: List[ParticleEvent]
    ) -> List[RecoEvent]:
        """Higher-level: simulate directly from in-memory particle events.
        Default writes a temp file and calls simulate(); subclasses may override.
        """
        raise NotImplementedError


# ── Delphes backend ────────────────────────────────────────────────────────────
class DelphesBackend(DetectorBackend):
    """
    Fast parametric detector simulation using Delphes.
    Requires a Delphes installation and a detector card.
    """
    def __init__(self, delphes_bin: Path = DELPHES_BIN,
                 detector_card: Path = DELPHES_CARD_DEFAULT):
        self.delphes_bin  = Path(delphes_bin)
        self.detector_card = Path(detector_card)

    def is_available(self) -> bool:
        return self.delphes_bin.exists() and self.detector_card.exists()

    def simulate(self, hepmc_path: Path) -> Path:
        """Run Delphes on a HepMC file. Returns path to .root output."""
        out_root = hepmc_path.with_suffix('.root')
        r = subprocess.run(
            [str(self.delphes_bin), str(self.detector_card),
             str(out_root), str(hepmc_path)],
            capture_output=True, text=True,
            env=_get_mg5_env(), timeout=3600
        )
        if not out_root.exists():
            raise RuntimeError(f"Delphes failed:\n{r.stderr[-1000:]}")
        return out_root

    def read_root_to_reco(self, root_path: Path) -> List[RecoEvent]:
        """
        Read Delphes ROOT output into RecoEvent objects.
        Requires uproot or PyROOT.
        """
        try:
            import uproot
        except ImportError:
            raise ImportError("uproot not installed. Run: pip install uproot awkward")

        events = []
        with uproot.open(root_path) as f:
            tree = f["Delphes"]
            jets = tree["Jet"].arrays(["PT", "Eta", "Phi", "Mass"], library="np")
            eles = tree["Electron"].arrays(["PT", "Eta", "Phi"], library="np")
            mus  = tree["Muon"].arrays(["PT", "Eta", "Phi"], library="np")
            mets = tree["MissingET"].arrays(["MET", "Phi"], library="np")

            n_ev = len(jets["PT"])
            for i in range(n_ev):
                ev = RecoEvent()
                for j in range(len(jets["PT"][i])):
                    pT = jets["PT"][i][j]
                    eta = jets["Eta"][i][j]
                    phi = jets["Phi"][i][j]
                    m   = jets["Mass"][i][j]
                    px = pT * np.cos(phi); py = pT * np.sin(phi)
                    pz = pT * np.sinh(eta)
                    E  = np.sqrt(px**2 + py**2 + pz**2 + m**2)
                    ev.jets.append(Particle(pid=0, status=1, px=px, py=py, pz=pz, E=E))

                for j in range(len(eles["PT"][i])):
                    pT = eles["PT"][i][j]; eta = eles["Eta"][i][j]; phi = eles["Phi"][i][j]
                    px = pT*np.cos(phi); py = pT*np.sin(phi); pz = pT*np.sinh(eta)
                    E  = np.sqrt(px**2+py**2+pz**2)
                    ev.electrons.append(Particle(pid=11, status=1, px=px, py=py, pz=pz, E=E))

                for j in range(len(mus["PT"][i])):
                    pT = mus["PT"][i][j]; eta = mus["Eta"][i][j]; phi = mus["Phi"][i][j]
                    px = pT*np.cos(phi); py = pT*np.sin(phi); pz = pT*np.sinh(eta)
                    E  = np.sqrt(px**2+py**2+pz**2)
                    ev.muons.append(Particle(pid=13, status=1, px=px, py=py, pz=pz, E=E))

                if len(mets["MET"][i]) > 0:
                    met_mag = mets["MET"][i][0]; met_phi = mets["Phi"][i][0]
                    ev.met_x = met_mag * np.cos(met_phi)
                    ev.met_y = met_mag * np.sin(met_phi)

                events.append(ev)
        return events

    def read_eflow_to_reco(self, root_path: Path) -> List[RecoEvent]:
        """
        Read Delphes EFlowTrack / EFlowPhoton / EFlowNeutralHadron into RecoEvent.
        All EFlow candidates are merged into ev.jets to match Parnassus convention
        (which also stores all PF candidates there regardless of type).
        """
        try:
            import uproot
        except ImportError:
            raise ImportError("uproot not installed. Run: pip install uproot awkward")

        events = []
        with uproot.open(root_path) as f:
            tree = f["Delphes"]
            # Uproot 5: sub-branches are accessed as "Collection/Collection.Field"
            trk = tree["EFlowTrack"].arrays(
                ["EFlowTrack.PT", "EFlowTrack.Eta", "EFlowTrack.Phi"], library="np")
            pho = tree["EFlowPhoton"].arrays(
                ["EFlowPhoton.ET", "EFlowPhoton.Eta", "EFlowPhoton.Phi"], library="np")
            neu = tree["EFlowNeutralHadron"].arrays(
                ["EFlowNeutralHadron.ET", "EFlowNeutralHadron.Eta",
                 "EFlowNeutralHadron.Phi"], library="np")
            n_ev = len(trk["EFlowTrack.PT"])
            for i in range(n_ev):
                ev = RecoEvent()
                for arr, pt_key, eta_key, phi_key in [
                    (trk, "EFlowTrack.PT",          "EFlowTrack.Eta",          "EFlowTrack.Phi"),
                    (pho, "EFlowPhoton.ET",          "EFlowPhoton.Eta",         "EFlowPhoton.Phi"),
                    (neu, "EFlowNeutralHadron.ET",   "EFlowNeutralHadron.Eta",  "EFlowNeutralHadron.Phi"),
                ]:
                    pts = arr[pt_key][i]
                    etas = arr[eta_key][i]
                    phis = arr[phi_key][i]
                    for j in range(len(pts)):
                        pT, eta, phi = float(pts[j]), float(etas[j]), float(phis[j])
                        px = pT * np.cos(phi); py = pT * np.sin(phi)
                        pz = pT * np.sinh(eta)
                        E  = np.sqrt(px**2 + py**2 + pz**2)
                        ev.jets.append(Particle(pid=0, status=1,
                                                px=px, py=py, pz=pz, E=E))
                events.append(ev)
        return events


# ── ML detector backend (stub) ─────────────────────────────────────────────────
class MLDetectorBackend(DetectorBackend):
    """
    Trained ML detector surrogate.

    Takes a particle-level point cloud and predicts reconstructed objects.
    Should be trained on (particle_cloud, Delphes_output) pairs from
    diverse SM and BSM processes.

    Architecture options:
      - Flow matching conditioned on particle-level point cloud encoding
      - Transformer encoder over particle cloud → decoder for reco objects
      - Point cloud diffusion (similar to FPCD but for detector effects)

    Training data:
      Generate using: python training/generate_detector_data.py
      Train using:    python training/train_detector.py
    """
    def __init__(self, checkpoint: Optional[Path] = None, device: str = "cpu"):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    def is_available(self) -> bool:
        return self.checkpoint is not None and Path(self.checkpoint).exists()

    def simulate(self, hepmc_path: Path) -> Path:
        raise NotImplementedError(
            "ML detector model not yet implemented. "
            "Use DelphesBackend for now, generate training data, "
            "then train and plug in an MLDetectorBackend."
        )


# ── Parnassus backend ──────────────────────────────────────────────────────────
class ParnassusBackend(DetectorBackend):
    """
    Detector surrogate using the pre-trained Parnassus CMS flow-matching model.

    Input  : List[ParticleEvent] with particles stored as (px, py, pz, E)
    Output : List[RecoEvent] where each PF candidate is stored in ev.jets

    The backend:
      1. Converts each ParticleEvent to (pT, eta, phi, charge_class=1) numpy format
      2. Calls infer_particles.py as a subprocess in pipeline_copy-gpu2 env
      3. Parses the output (N, max_pf, 3) array back into RecoEvent objects

    charge_class is set to 1 (charged) for all particles because FPCD outputs
    a generic point cloud with no PID information.
    """

    INFER_SCRIPT = _PARNASSUS_DIR / "infer_particles.py"

    def __init__(
        self,
        checkpoint: Path = _PARNASSUS_DIR / "trained_models" / "fm_cms_J800_1000_epoch=49.ckpt",
        config: Path = _PARNASSUS_DIR / "configs" / "fm_cms_J800_1000.json",
        n_steps: int = 25,
        batch_size: int = 256,
        device: str = "cuda",
        python: Path = _PARNASSUS_PYTHON,
        work_dir: Path = Path("/pscratch/sd/l/lcondren/MCsim/parnassus_work"),
    ):
        self.checkpoint = Path(checkpoint)
        self.config     = Path(config)
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.device     = device
        self.python     = Path(python)
        self.work_dir   = Path(work_dir)

    def is_available(self) -> bool:
        return (self.checkpoint.exists()
                and self.config.exists()
                and self.INFER_SCRIPT.exists()
                and self.python.exists())

    def simulate(self, shower_out: Path) -> Path:
        """Read shower output file and run Parnassus. Returns .parnassus.npz path."""
        from .shower import _read_shower_output
        particle_events = _read_shower_output(shower_out)
        reco_events = self.simulate_from_particles(particle_events)
        # Save PF candidates alongside the shower output for auditability
        out_path = shower_out.with_suffix("").with_suffix(".parnassus.npz")
        self._save_reco_npz(reco_events, out_path)
        return out_path

    def simulate_from_particles(
        self, particle_events: List[ParticleEvent]
    ) -> List[RecoEvent]:
        """Core inference path: List[ParticleEvent] → List[RecoEvent]."""
        import tempfile

        if not particle_events:
            return []

        truth_np, truth_mask_np = self._events_to_numpy(particle_events)

        self.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(self.work_dir)) as tmpdir:
            tmpdir = Path(tmpdir)
            truth_path    = tmpdir / "truth.npy"
            mask_path     = tmpdir / "truth_mask.npy"
            out_path      = tmpdir / "pf_candidates.npy"
            out_mask_path = tmpdir / "pf_mask.npy"

            np.save(truth_path, truth_np)
            np.save(mask_path,  truth_mask_np)

            cmd = [
                str(self.python), str(self.INFER_SCRIPT),
                "--input",       str(truth_path),
                "--input-mask",  str(mask_path),
                "--output",      str(out_path),
                "--output-mask", str(out_mask_path),
                "--checkpoint",  str(self.checkpoint),
                "--config",      str(self.config),
                "--n-steps",     str(self.n_steps),
                "--batch-size",  str(self.batch_size),
                "--device",      self.device,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if r.returncode != 0:
                raise RuntimeError(f"Parnassus inference failed:\n{r.stderr[-2000:]}")

            pf_out  = np.load(out_path)       # (N, max_pf, 3) [pT, eta, phi]
            pf_mask = np.load(out_mask_path)  # (N, max_pf) bool

        return self._numpy_to_reco_events(pf_out, pf_mask)

    @staticmethod
    def _events_to_numpy(events: List[ParticleEvent]):
        """Convert particle events to Parnassus input format.

        Returns
        -------
        truth_np   : (N, max_part, 4)  float32  [pT_GeV, eta, phi, charge_class]
        truth_mask : (N, max_part)     bool
        """
        N        = len(events)
        max_part = max((len(e.particles) for e in events), default=1)
        max_part = max(max_part, 1)

        truth_np   = np.zeros((N, max_part, 4), dtype=np.float32)
        truth_mask = np.zeros((N, max_part),    dtype=bool)

        for i, ev in enumerate(events):
            for j, p in enumerate(ev.particles):
                pT  = float(np.sqrt(p.px**2 + p.py**2))
                mag = float(np.sqrt(p.px**2 + p.py**2 + p.pz**2))
                eta = float(np.arctanh(np.clip(p.pz / max(mag, 1e-9), -0.9999, 0.9999)))
                phi = float(np.arctan2(p.py, p.px))
                truth_np[i, j]   = [pT, eta, phi, 1.0]  # charge_class=1 (charged)
                truth_mask[i, j] = True

        return truth_np, truth_mask

    @staticmethod
    def _numpy_to_reco_events(pf_out: np.ndarray, pf_mask: np.ndarray) -> List[RecoEvent]:
        """Convert Parnassus output arrays to RecoEvent objects.

        PF candidates have no class information, so all are stored in ev.jets
        as massless particles in (px, py, pz, E) coordinates.
        """
        events: List[RecoEvent] = []
        for i in range(pf_out.shape[0]):
            ev = RecoEvent()
            for j in np.where(pf_mask[i])[0]:
                pT  = float(pf_out[i, j, 0])
                eta = float(pf_out[i, j, 1])
                phi = float(pf_out[i, j, 2])
                px  = pT * np.cos(phi)
                py  = pT * np.sin(phi)
                pz  = pT * np.sinh(eta)
                E   = pT * np.cosh(eta)
                ev.jets.append(Particle(pid=0, status=1, px=px, py=py, pz=pz, E=E))
            events.append(ev)
        return events

    @staticmethod
    def _save_reco_npz(reco_events: List[RecoEvent], path: Path):
        """Save reco events as a .npz for audit/debug purposes."""
        N = len(reco_events)
        max_pf = max((len(e.jets) for e in reco_events), default=0)
        pf_out  = np.zeros((N, max_pf, 3), dtype=np.float32)
        pf_mask = np.zeros((N, max_pf),    dtype=bool)
        for i, ev in enumerate(reco_events):
            for j, p in enumerate(ev.jets):
                pT  = p.pT
                eta = float(np.arctanh(np.clip(p.pz / max(p.E, 1e-9), -0.9999, 0.9999)))
                phi = float(np.arctan2(p.py, p.px))
                pf_out[i, j]  = [pT, eta, phi]
                pf_mask[i, j] = True
        np.savez(path, pf_candidates=pf_out, pf_mask=pf_mask)


def make_detector_backend(backend: str = "delphes", **kwargs) -> DetectorBackend:
    """Factory: return the requested detector backend."""
    if backend == "delphes":
        return DelphesBackend(**kwargs)
    elif backend == "parnassus":
        return ParnassusBackend(**kwargs)
    elif backend == "ml":
        return MLDetectorBackend(**kwargs)
    else:
        raise ValueError(f"Unknown detector backend: {backend!r}. "
                         f"Choose 'delphes', 'parnassus', or 'ml'.")
