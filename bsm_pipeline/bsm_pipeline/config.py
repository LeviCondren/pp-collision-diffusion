"""
Pipeline configuration: load and validate a YAML config file.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


class PipelineConfig:
    """Loaded and validated pipeline configuration."""

    def __init__(self, cfg: dict):
        self._cfg = cfg

    # ── Model / process ────────────────────────────────────────────────────────
    @property
    def ufo_path(self) -> Path:
        return Path(self._cfg["model"]["ufo_path"])

    @property
    def process(self) -> str:
        return self._cfg["model"]["process"]

    @property
    def model_name(self) -> str:
        return self._cfg["model"].get("name", self.ufo_path.name)

    # ── Parameters ────────────────────────────────────────────────────────────
    @property
    def parameters(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self._cfg.get("parameters", {}).items()}

    @property
    def parameter_scan(self) -> Optional[Dict[str, List[float]]]:
        """If set, defines a grid scan over parameters."""
        return self._cfg.get("parameter_scan", None)

    # ── Generation ────────────────────────────────────────────────────────────
    @property
    def n_events(self) -> int:
        return int(self._cfg.get("generation", {}).get("n_events", 5000))

    @property
    def nb_core(self) -> int:
        return int(self._cfg.get("generation", {}).get("nb_core", 64))

    @property
    def energy_gev(self) -> float:
        return float(self._cfg.get("generation", {}).get("energy_gev", 13000.))

    @property
    def mmll_min(self) -> float:
        return float(self._cfg.get("generation", {}).get("mmll_min", 4.))

    @property
    def mmll_max(self) -> float:
        return float(self._cfg.get("generation", {}).get("mmll_max", 120.))

    @property
    def seed(self) -> int:
        return int(self._cfg.get("generation", {}).get("seed", 42))

    @property
    def use_gridpack(self) -> bool:
        return bool(self._cfg.get("generation", {}).get("use_gridpack", False))

    # ── Shower ────────────────────────────────────────────────────────────────
    @property
    def shower_backend(self) -> str:
        return self._cfg.get("shower", {}).get("backend", "pythia8")

    @property
    def shower_kwargs(self) -> dict:
        cfg = dict(self._cfg.get("shower", {}))
        cfg.pop("backend", None)
        return cfg

    # ── Detector ──────────────────────────────────────────────────────────────
    @property
    def detector_backend(self) -> str:
        return self._cfg.get("detector", {}).get("backend", "delphes")

    @property
    def detector_kwargs(self) -> dict:
        cfg = dict(self._cfg.get("detector", {}))
        cfg.pop("backend", None)
        cfg.pop("enabled", None)
        return cfg

    @property
    def run_detector(self) -> bool:
        return bool(self._cfg.get("detector", {}).get("enabled", True))

    # ── Output ────────────────────────────────────────────────────────────────
    @property
    def output_dir(self) -> Path:
        return Path(self._cfg.get("output", {}).get("dir", "./pipeline_output"))

    @property
    def save_lhe(self) -> bool:
        return bool(self._cfg.get("output", {}).get("save_lhe", True))

    @property
    def save_hepmc(self) -> bool:
        return bool(self._cfg.get("output", {}).get("save_hepmc", False))

    @property
    def save_numpy(self) -> bool:
        return bool(self._cfg.get("output", {}).get("save_numpy", True))

    @property
    def work_dir(self) -> Path:
        return Path(self._cfg.get("output", {}).get("work_dir",
                    str(self.output_dir / "_work")))

    def validate(self):
        """Raise ValueError if required fields are missing."""
        if not self.ufo_path.exists():
            raise ValueError(f"UFO model path not found: {self.ufo_path}")
        if not self.process.strip():
            raise ValueError("'model.process' must be a non-empty MG5 process string.")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        inst = cls(cfg)
        inst.validate()
        return inst

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        inst = cls(d)
        inst.validate()
        return inst
