#!/usr/bin/env python3
"""
run_pipeline.py — single-button BSM simulation pipeline.

Usage
-----
# Single parameter point (from config file):
python run_pipeline.py --config my_model.yaml

# Override a parameter on the command line:
python run_pipeline.py --config my_model.yaml --param MAp=30 --param gV_0=0.05

# Parameter scan (defined in config under 'parameter_scan'):
python run_pipeline.py --config my_model.yaml --scan

# Quick test with N events:
python run_pipeline.py --config my_model.yaml --n-events 100

Output
------
  <output.dir>/<tag>/z_lhe_<tag>.npy    (N, 5) parton-level kinematics
  <output.dir>/<tag>/z_truth_<tag>.npy  (N, 5) showered kinematics
  <output.dir>/<tag>/theta_<tag>.npy    (P,)   parameter vector
  <output.dir>/<tag>/meta_<tag>.json    run metadata
"""

import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bsm_pipeline import BSMPipeline, PipelineConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="BSM fast simulation: MG5 → Pythia8 → Delphes (or ML backends)")
    p.add_argument("--config", required=True,
                   help="Path to YAML pipeline configuration file.")
    p.add_argument("--param", action="append", default=[],
                   metavar="NAME=VALUE",
                   help="Override a parameter (repeatable). E.g. --param MAp=30")
    p.add_argument("--scan", action="store_true",
                   help="Run over parameter_scan grid defined in config.")
    p.add_argument("--n-events", type=int, default=None,
                   help="Override n_events from config.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override random seed.")
    p.add_argument("--tag", type=str, default="run",
                   help="Output tag (used in filenames).")
    p.add_argument("--setup-only", action="store_true",
                   help="Only set up the MG5 process directory, do not generate events.")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # ── Load config ────────────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"ERROR: config file not found: {cfg_path}")

    import yaml
    with open(cfg_path) as f:
        cfg_dict = yaml.safe_load(f)

    # Apply command-line overrides
    if args.n_events:
        cfg_dict.setdefault("generation", {})["n_events"] = args.n_events
    if args.seed:
        cfg_dict.setdefault("generation", {})["seed"] = args.seed

    param_overrides = {}
    for kv in args.param:
        if "=" not in kv:
            sys.exit(f"ERROR: --param must be NAME=VALUE, got: {kv!r}")
        k, v = kv.split("=", 1)
        param_overrides[k.strip()] = float(v.strip())
    if param_overrides:
        cfg_dict.setdefault("parameters", {}).update(param_overrides)

    cfg = PipelineConfig.from_dict(cfg_dict)
    pipeline = BSMPipeline(cfg)

    # ── Run ────────────────────────────────────────────────────────────────────
    if args.setup_only:
        print("[run_pipeline] Setting up MG5 process (--setup-only) ...")
        pipeline.mg5.setup_process(nb_core=cfg.nb_core, force=False)
        print(f"[run_pipeline] Process ready at {pipeline.mg5._proc_dir}")
        return

    if args.scan:
        if cfg.parameter_scan is None:
            sys.exit("ERROR: --scan requires 'parameter_scan' section in config.")
        results = []
        for result in pipeline.scan():
            results.append(result)
            print(f"  Completed: {result.tag}")
        print(f"\n[run_pipeline] Scan complete: {len(results)} points "
              f"in {(time.time()-t0)/60:.1f} min")
        print(f"Output: {cfg.output_dir}")
    else:
        result = pipeline.run(tag=args.tag)
        print(f"\n[run_pipeline] Complete in {(time.time()-t0)/60:.1f} min")
        print(f"Output: {cfg.output_dir / args.tag}")


if __name__ == "__main__":
    main()
