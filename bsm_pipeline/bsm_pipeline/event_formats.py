"""
Event data structures shared across pipeline stages.

LHEEvent   : hard-process particles from MadGraph5
ParticleEvent : showered/hadronized final-state particles
RecoEvent  : reconstructed detector-level objects
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Particle:
    pid: int          # PDG particle ID
    status: int       # LHE status (+1 = final state)
    px: float
    py: float
    pz: float
    E: float

    @property
    def pT(self) -> float:
        return float(np.sqrt(self.px**2 + self.py**2))

    @property
    def mass(self) -> float:
        m2 = self.E**2 - self.px**2 - self.py**2 - self.pz**2
        return float(np.sqrt(max(m2, 0.)))

    def four_vector(self) -> np.ndarray:
        return np.array([self.E, self.px, self.py, self.pz], dtype=np.float32)


@dataclass
class LHEEvent:
    """Single hard-process event as output by MadGraph5."""
    particles: List[Particle] = field(default_factory=list)
    weight: float = 1.0
    process_id: int = 0
    raw_text: Optional[str] = None  # verbatim lines between <event>…</event> for round-trip LHE writing

    @property
    def final_state(self) -> List[Particle]:
        return [p for p in self.particles if p.status == 1]

    def as_point_cloud(self, include_pid: bool = True) -> np.ndarray:
        """
        Return final-state particles as (N, 5) array: [pid, E, px, py, pz]
        or (N, 4) if include_pid=False.
        Used as input to the shower model.
        """
        fs = self.final_state
        if not fs:
            return np.zeros((0, 5 if include_pid else 4), dtype=np.float32)
        rows = []
        for p in fs:
            if include_pid:
                rows.append([p.pid, p.E, p.px, p.py, p.pz])
            else:
                rows.append([p.E, p.px, p.py, p.pz])
        return np.array(rows, dtype=np.float32)


@dataclass
class ParticleEvent:
    """Showered/hadronized event: variable-length list of final-state particles."""
    particles: List[Particle] = field(default_factory=list)
    lhe_event: Optional[LHEEvent] = None   # paired LHE event for training

    def as_point_cloud(self) -> np.ndarray:
        """(N, 5) array: [pid, E, px, py, pz] for all final-state particles."""
        if not self.particles:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array([[p.pid, p.E, p.px, p.py, p.pz]
                         for p in self.particles], dtype=np.float32)


@dataclass
class RecoEvent:
    """Detector-level reconstructed objects."""
    jets: List[Particle] = field(default_factory=list)       # anti-kT R=0.4 jets
    electrons: List[Particle] = field(default_factory=list)
    muons: List[Particle] = field(default_factory=list)
    photons: List[Particle] = field(default_factory=list)
    met_x: float = 0.0
    met_y: float = 0.0

    @property
    def MET(self) -> float:
        return float(np.sqrt(self.met_x**2 + self.met_y**2))

    def as_feature_dict(self) -> dict:
        """Flat dict of scalar observables, for analysis."""
        d = {"MET": self.MET, "n_jets": len(self.jets),
             "n_electrons": len(self.electrons), "n_muons": len(self.muons)}
        for tag, col in [("jet", self.jets), ("ele", self.electrons),
                         ("mu", self.muons)]:
            for i, p in enumerate(col[:4]):
                d[f"{tag}{i+1}_pT"] = p.pT
                d[f"{tag}{i+1}_eta"] = float(np.arctanh(np.clip(
                    p.pz / max(p.E, 1e-9), -0.9999, 0.9999)))
                d[f"{tag}{i+1}_phi"] = float(np.arctan2(p.py, p.px))
        return d
