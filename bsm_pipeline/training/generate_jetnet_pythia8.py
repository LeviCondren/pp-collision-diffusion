"""
generate_jetnet_pythia8.py — Reproduce JetNet training data with Pythia8.

Matches the exact generation settings from FPCD (arXiv:2304.01266) / JetNet:
  - Pythia8, pp @ 13 TeV
  - QCD hard scatter → anti-kT R=0.8 jet clustering
  - Jet pT ∈ [0.8, 1.6] TeV, |η_jet| < 2.5
  - 5 jet types: gluon, light-quark (u/d/s), top, W, Z
  - Top 30 (or 150) highest-pT constituents per jet
  - Per-particle features: (η_rel, φ_rel, pT_rel, mask)

Requires: Pythia8 Python bindings (pythia8.so) — available via the mg5_new env
or via pip install pythia8 / conda install pythia8.

Alternatively uses the fastjet Python bindings for jet clustering.

Usage:
  python generate_jetnet_pythia8.py --jet-type g --n-jets 200000 \
      --out-dir /pscratch/sd/l/lcondren/MCsim/jetnet_generated --n-particles 30

  # All types in parallel (submit as SLURM array):
  for t in g q t w z; do
      sbatch submit_jetnet_gen.sh $t
  done
"""

import argparse, math, sys, time
from pathlib import Path
import numpy as np

# ── Jet clustering via fastjet ─────────────────────────────────────────────────
def _require_fastjet():
    try:
        import fastjet as fj
        return fj
    except ImportError:
        sys.exit(
            "fastjet Python bindings not found.\n"
            "Install via: pip install fastjet\n"
            "or: conda install -c conda-forge fastjet"
        )


# ── Pythia8 setup ─────────────────────────────────────────────────────────────
def _require_pythia8():
    # Try the standard pythia8 Python binding
    for mod_name in ("pythia8", "Pythia8"):
        try:
            return __import__(mod_name)
        except ImportError:
            pass
    # Try the mg5 env path
    sys.path.insert(0, "/pscratch/sd/l/lcondren/MCsim/MG5_aMC_v3_6_7"
                       "/HEPTools/pythia8/lib")
    try:
        import pythia8
        return pythia8
    except ImportError:
        sys.exit(
            "Pythia8 Python bindings not found.\n"
            "Install via: pip install pythia8\n"
            "or use the standalone generation script (generate_jetnet_standalone.sh)"
        )


# ── Jet type → Pythia8 process settings ───────────────────────────────────────
JET_PROCESSES = {
    # (description, Pythia8 settings list)
    "g": ("gluon jets",
          ["HardQCD:gg2gg = on",
           "HardQCD:gg2ccbar = off",
           "HardQCD:gg2bbbar = off",
           "HardQCD:qg2qg = off",
           "HardQCD:qq2qq = off",
           "HardQCD:qqbar2gg = off"]),

    "q": ("light-quark jets (u/d/s)",
          ["HardQCD:qq2qq = on",
           "HardQCD:qqbar2gg = off",
           "HardQCD:gg2gg = off",
           "HardQCD:qg2qg = off"]),

    "t": ("top-quark jets",
          ["Top:gg2ttbar = on",
           "Top:qqbar2ttbar = on",
           "24:onMode = off",       # W decays → hadronic (top jet context)
           "24:onIfAny = 1 2 3 4"]),

    "w": ("W-boson jets",
          ["WeakSingleBoson:ffbar2W = on",
           "24:onMode = off",
           "24:onIfAny = 1 2 3 4 5"]),   # hadronic W decays

    "z": ("Z-boson jets",
          ["WeakSingleBoson:ffbar2Z = on",
           "23:onMode = off",
           "23:onIfAny = 1 2 3 4 5"]),   # hadronic Z decays
}

# pT hat bins (GeV) — center near 1 TeV
PT_HAT_MIN = 800.    # GeV
PT_HAT_MAX = 1600.   # GeV
JET_PT_MIN = 800.
JET_PT_MAX = 1600.
JET_ETA_MAX = 2.5
R_JET = 0.8


def generate_jets(jet_type: str, n_jets: int, n_particles: int,
                  out_dir: Path, seed: int = 42):
    fj = _require_fastjet()
    pythia8 = _require_pythia8()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file     = out_dir / f"{jet_type}_jets.npy"
    jet_feat_file = out_dir / f"{jet_type}_jet_features.npy"

    if out_file.exists():
        print(f"[{jet_type}] Already exists: {out_file}, skipping.")
        return

    desc, proc_settings = JET_PROCESSES[jet_type]
    print(f"[{jet_type}] Generating {n_jets} {desc} ...")

    # ── Pythia8 initialisation ─────────────────────────────────────────────────
    pythia = pythia8.Pythia("", False)
    pythia.readString("Beams:idA = 2212")
    pythia.readString("Beams:idB = 2212")
    pythia.readString("Beams:eCM = 13000.")
    pythia.readString(f"PhaseSpace:pTHatMin = {PT_HAT_MIN}")
    pythia.readString(f"PhaseSpace:pTHatMax = {PT_HAT_MAX}")
    pythia.readString("PartonLevel:MPI = off")
    pythia.readString("PartonLevel:ISR = on")
    pythia.readString("PartonLevel:FSR = on")
    pythia.readString("HadronLevel:Hadronize = on")
    pythia.readString(f"Random:seed = {seed}")
    pythia.readString("Random:setSeed = on")
    pythia.readString("Print:quiet = on")
    for s in proc_settings:
        pythia.readString(s)
    pythia.init()

    # ── fastjet setup ──────────────────────────────────────────────────────────
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R_JET)

    particles_list = []
    jet_features_list = []
    n_generated = 0
    t0 = time.time()

    while len(particles_list) < n_jets:
        if not pythia.next():
            continue
        n_generated += 1

        # Build fastjet input: all final-state particles
        fj_particles = []
        for i in range(pythia.event.size()):
            p = pythia.event[i]
            if not p.isFinal():
                continue
            fj_particles.append(fj.PseudoJet(p.px(), p.py(), p.pz(), p.e()))

        if not fj_particles:
            continue

        # Cluster
        cs   = fj.ClusterSequence(fj_particles, jet_def)
        jets = fj.sorted_by_pt(cs.inclusive_jets())

        # Select leading jet in acceptance
        lead = None
        for j in jets:
            if (JET_PT_MIN < j.pt() < JET_PT_MAX and abs(j.rapidity()) < JET_ETA_MAX):
                lead = j
                break
        if lead is None:
            continue

        # Get constituents, sort by pT descending
        constituents = sorted(lead.constituents(), key=lambda p: p.pt(), reverse=True)

        # Build (n_particles, 4) array: [η_rel, φ_rel, pT_rel, mask]
        jet_eta = lead.eta(); jet_phi = lead.phi(); jet_pt = lead.pt()
        row = np.zeros((n_particles, 4), dtype=np.float32)
        for k, c in enumerate(constituents[:n_particles]):
            dphi = c.phi() - jet_phi
            # Wrap to [-π, π]
            while dphi >  math.pi: dphi -= 2*math.pi
            while dphi < -math.pi: dphi += 2*math.pi
            row[k] = [c.eta() - jet_eta, dphi, c.pt() / jet_pt, 1.0]

        particles_list.append(row)
        jet_features_list.append([jet_pt, jet_eta, lead.m()])

        if len(particles_list) % 10000 == 0:
            elapsed = time.time() - t0
            rate = len(particles_list) / elapsed
            eta  = (n_jets - len(particles_list)) / rate
            print(f"  [{jet_type}] {len(particles_list):6d}/{n_jets}  "
                  f"{rate:.0f} jets/s  ETA {eta/60:.1f} min")

    pythia.stat()

    data     = np.stack(particles_list,    axis=0)  # (N, n_particles, 4)
    jet_feat = np.array(jet_features_list, dtype=np.float32)  # (N, 3)

    np.save(out_file,      data)
    np.save(jet_feat_file, jet_feat)

    elapsed = time.time() - t0
    print(f"[{jet_type}] Saved {data.shape} in {elapsed/60:.1f} min")
    print(f"  Mean constituents: {data[:,:,3].sum(1).mean():.1f}")
    print(f"  Jet pT range: [{jet_feat[:,0].min():.0f}, {jet_feat[:,0].max():.0f}] GeV")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jet-type", choices=list(JET_PROCESSES.keys()) + ["all"],
                   required=True, help="Jet type to generate")
    p.add_argument("--n-jets", type=int, default=200000,
                   help="Number of jets (paper uses 200k per type)")
    p.add_argument("--n-particles", type=int, default=30, choices=[30, 150],
                   help="Max particles per jet (30 or 150, matching JetNet versions)")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for generated jets")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir) / f"jetnet{args.n_particles}_generated"
    types   = list(JET_PROCESSES.keys()) if args.jet_type == "all" else [args.jet_type]

    for jtype in types:
        generate_jets(jtype, args.n_jets, args.n_particles, out_dir, args.seed)

    print(f"\nDone. Data at: {out_dir}")
    print("Next step: python training/prepare_fpcd_training.py "
          f"--jetnet-dir {out_dir.parent}")


if __name__ == "__main__":
    main()
