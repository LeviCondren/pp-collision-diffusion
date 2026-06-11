# -*- coding: utf-8 -*-
"""
bsm_pipeline: one-command BSM fast simulation pipeline.

  MG5 (theta -> LHE)  ->  Shower (LHE -> particles)  ->  Detector (particles -> reco)

Backends:
  Shower  : Pythia8 (today) | FPCD (when trained)
  Detector: Delphes (today) | ML model (when trained)
"""
from .config   import PipelineConfig
from .pipeline import BSMPipeline, SimulationResult
from .mg5_runner import MG5Runner, parse_lhe
from .shower   import (Pythia8Backend, FPCDBackend, make_shower_backend,
                       parse_hepmc, extract_dilepton_kinematics)
from .detector import DelphesBackend, MLDetectorBackend, make_detector_backend
from .event_formats import Particle, LHEEvent, ParticleEvent, RecoEvent

__all__ = [
    "BSMPipeline", "PipelineConfig", "SimulationResult",
    "MG5Runner", "parse_lhe",
    "Pythia8Backend", "FPCDBackend", "make_shower_backend",
    "parse_hepmc", "extract_dilepton_kinematics",
    "DelphesBackend", "MLDetectorBackend", "make_detector_backend",
    "Particle", "LHEEvent", "ParticleEvent", "RecoEvent",
]
