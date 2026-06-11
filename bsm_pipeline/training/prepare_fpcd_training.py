"""
prepare_fpcd_training.py — Convert JetNet data into the exact HDF5 format
expected by the FPCD training code (github.com/ViniciusMikuni/GSGM).

Input : downloaded or generated JetNet numpy arrays
Output: train.h5 and test.h5 in FPCD format

FPCD HDF5 format:
  data/  : (N, n_particles, 3)  float32  [eta_rel, phi_rel, pt_rel]
           (mask is stored separately, not as a feature channel)
  mask/  : (N, n_particles)     float32  1=real, 0=padding
  jet/   : (N, 3)               float32  [pt, eta, mass]

Train/test split: 80/20 (matching FPCD paper)

Usage:
  python prepare_fpcd_training.py \
      --jetnet-dir /pscratch/sd/l/lcondren/MCsim/jetnet_data \
      --out-dir    /pscratch/sd/l/lcondren/MCsim/fpcd_training \
      --version    30
"""

import argparse, sys
from pathlib import Path
import numpy as np

JET_NAMES  = ["gluon", "quark", "top", "w_boson", "z_boson"]
JET_LABELS = {"gluon": 0, "quark": 1, "top": 2, "w_boson": 3, "z_boson": 4}


def load_jetnet_type(jetnet_dir: Path, name: str, n_particles: int):
    """Load one jet type. Returns (particles, jet_features)."""
    f = jetnet_dir / f"jetnet{n_particles}" / f"{name}.npy"
    jf = jetnet_dir / f"jetnet{n_particles}" / f"jet_features_{name}.npy"

    if not f.exists():
        # Try generated path
        f  = jetnet_dir / f"jetnet{n_particles}_generated" / f"{name[0]}_jets.npy"
        jf = jetnet_dir / f"jetnet{n_particles}_generated" / f"{name[0]}_jet_features.npy"

    if not f.exists():
        print(f"  WARNING: {name} not found at {f}, skipping")
        return None, None

    particles = np.load(f)     # (N, n_part, 4): [eta_rel, phi_rel, pt_rel, mask]
    jet_feat  = np.load(jf) if jf.exists() else np.zeros((len(particles), 3), np.float32)
    return particles, jet_feat


def prepare(jetnet_dir: Path, out_dir: Path, n_particles: int,
            train_frac: float = 0.8):
    try:
        import h5py
    except ImportError:
        sys.exit("h5py required: pip install h5py")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_particles = []
    all_jets      = []
    all_labels    = []

    for name in JET_NAMES:
        particles, jet_feat = load_jetnet_type(jetnet_dir, name, n_particles)
        if particles is None:
            continue
        print(f"  {name}: {particles.shape}")
        all_particles.append(particles)
        all_jets.append(jet_feat)
        all_labels.append(np.full(len(particles), JET_LABELS[name], dtype=np.int32))

    if not all_particles:
        sys.exit("No jet data found. Run download_jetnet.py first.")

    particles = np.concatenate(all_particles, axis=0)  # (N_total, n_part, 4)
    jets      = np.concatenate(all_jets,      axis=0)  # (N_total, 3)
    labels    = np.concatenate(all_labels,    axis=0)  # (N_total,)

    # Shuffle
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(particles))
    particles = particles[perm]; jets = jets[perm]; labels = labels[perm]

    # Split features and mask
    features = particles[:, :, :3].astype(np.float32)   # (N, n_part, 3)
    mask     = particles[:, :, 3].astype(np.float32)     # (N, n_part)

    # Train/test split
    n_train = int(train_frac * len(features))
    splits  = {"train": slice(None, n_train), "test": slice(n_train, None)}

    for split_name, sl in splits.items():
        out_path = out_dir / f"{split_name}_jetnet{n_particles}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("data",   data=features[sl], compression="gzip")
            f.create_dataset("mask",   data=mask[sl],     compression="gzip")
            f.create_dataset("jet",    data=jets[sl],     compression="gzip")
            f.create_dataset("labels", data=labels[sl],   compression="gzip")

        n = features[sl].shape[0]
        print(f"  Saved {split_name}: {n:,} jets → {out_path}")

    # Print summary matching paper numbers
    print(f"\nSummary (JetNet{n_particles}):")
    print(f"  Total jets : {len(features):,}")
    print(f"  Train      : {n_train:,}")
    print(f"  Test       : {len(features) - n_train:,}")
    print(f"  Features   : (N, {n_particles}, 3) = [eta_rel, phi_rel, pt_rel]")
    print(f"  Mask       : (N, {n_particles})")
    for name in JET_NAMES:
        n = (labels == JET_LABELS[name]).sum()
        if n > 0:
            print(f"  {name:10s}: {n:,} jets")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jetnet-dir", required=True,
                   help="Directory containing jetnet30/ and jetnet150/ subdirs")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for FPCD training HDF5 files")
    p.add_argument("--version", choices=["30", "150", "both"], default="both")
    args = p.parse_args()

    jetnet_dir = Path(args.jetnet_dir)
    out_dir    = Path(args.out_dir)
    versions   = [30, 150] if args.version == "both" else [int(args.version)]

    for n_part in versions:
        print(f"\n{'='*60}")
        print(f"Preparing JetNet{n_part} for FPCD training")
        print(f"{'='*60}")
        prepare(jetnet_dir, out_dir, n_part)

    print(f"\nDone. Training data at: {out_dir}")
    print("Clone FPCD and run:")
    print("  git clone https://github.com/ViniciusMikuni/GSGM")
    print(f"  python GSGM/train.py --data-path {out_dir} --n-part 30")


if __name__ == "__main__":
    main()
