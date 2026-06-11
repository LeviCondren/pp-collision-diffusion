"""
download_jetnet.py — Download JetNet30 and JetNet150 datasets from Zenodo.

These are the exact datasets used in FPCD (arXiv:2304.01266).

JetNet contents:
  5 jet types: gluon (g), light quark (q), top (t), W boson (w), Z boson (z)
  JetNet30:  top-30 highest-pT particles per jet, ~177k jets per type
  JetNet150: top-150 highest-pT particles per jet, ~177k jets per type

  Particle features (4 per particle):
    eta_rel  : pseudorapidity relative to jet axis
    phi_rel  : azimuthal angle relative to jet axis
    pt_rel   : transverse momentum fraction (pT_particle / pT_jet)
    mask     : 1 if real particle, 0 if zero-padding

  Generation: Pythia8.212, pp @ 13 TeV, anti-kT R=0.8,
              jet pT ∈ [0.8, 1.6] TeV, |eta_jet| < 2.5

Output:
  <out_dir>/jetnet30/
    gluon.npy   (177252, 30, 4)
    quark.npy
    top.npy
    w.npy
    z.npy
    jet_features_gluon.npy   (177252, 3)  [pt, eta, mass]
  <out_dir>/jetnet150/   (same structure, 150 particles per jet)

Usage:
  python download_jetnet.py --out-dir /pscratch/sd/l/lcondren/MCsim/jetnet_data
  python download_jetnet.py --out-dir ... --version 30    # JetNet30 only
  python download_jetnet.py --out-dir ... --version 150   # JetNet150 only
"""

import argparse, sys, time
from pathlib import Path
import numpy as np

# Individual HDF5 files per jet type (tarballs were removed from Zenodo)
ZENODO_FILES_30 = {
    jt: f"https://zenodo.org/records/6975118/files/{jt}.hdf5?download=1"
    for jt in ["g", "q", "t", "w", "z"]
}
ZENODO_FILES_150 = {
    jt: f"https://zenodo.org/records/6975117/files/{jt}150.hdf5?download=1"
    for jt in ["g", "q", "t", "w", "z"]
}

JET_TYPES  = ["g", "q", "t", "w", "z"]
JET_NAMES  = {"g": "gluon", "q": "quark", "t": "top", "w": "w_boson", "z": "z_boson"}


def download_via_jetnet_package(out_dir: Path, n_particles: int):
    """Use the jetnet Python package (cleanest method)."""
    try:
        import jetnet
    except ImportError:
        return False

    print(f"[jetnet] Downloading JetNet{n_particles} via jetnet package ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    for jtype in JET_TYPES:
        name = JET_NAMES[jtype]
        out_file = out_dir / f"{name}.npy"
        jet_file = out_dir / f"jet_features_{name}.npy"

        if out_file.exists() and jet_file.exists():
            print(f"  {name}: already downloaded, skipping")
            continue

        print(f"  Downloading {name} jets ...")
        t0 = time.time()
        try:
            particle_data, jet_data = jetnet.datasets.JetNet.getData(
                jet_type=jtype,
                data_dir=str(out_dir / "_raw"),
                num_particles=n_particles,
                split="all",
                download=True,
            )
            # particle_data: (N, n_particles, 4) [eta_rel, phi_rel, pt_rel, mask]
            # jet_data:      (N, 3)              [pt, eta, mass]
            np.save(out_file,  particle_data.astype(np.float32))
            np.save(jet_file,  jet_data.astype(np.float32))
            print(f"  {name}: {particle_data.shape}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            return False

    return True


def download_via_zenodo(out_dir: Path, n_particles: int):
    """Download individual HDF5 files from Zenodo (tarballs were removed)."""
    import urllib.request

    try:
        import h5py
    except ImportError:
        print("ERROR: h5py not installed. Run: pip install h5py")
        return False

    url_map = ZENODO_FILES_30 if n_particles == 30 else ZENODO_FILES_150
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)

    for jtype in JET_TYPES:
        name     = JET_NAMES[jtype]
        out_file = out_dir / f"{name}.npy"
        jet_file = out_dir / f"jet_features_{name}.npy"

        if out_file.exists() and jet_file.exists():
            print(f"  {name}: already downloaded, skipping")
            continue

        suffix = "" if n_particles == 30 else "150"
        h5_path = raw_dir / f"{jtype}{suffix}.hdf5"

        if not h5_path.exists():
            url = url_map[jtype]
            print(f"  Downloading {name} from Zenodo ...")
            t0 = time.time()
            try:
                urllib.request.urlretrieve(url, h5_path,
                    reporthook=lambda b, bs, total:
                        print(f"\r    {b*bs/1e6:.0f}/{total/1e6:.0f} MB",
                              end="", flush=True) if b % 100 == 0 else None)
                print(f"\n    Downloaded in {(time.time()-t0):.0f}s")
            except Exception as e:
                print(f"  {name}: download FAILED — {e}")
                return False

        with h5py.File(h5_path, "r") as f:
            if "particle_features" in f:
                particles = f["particle_features"][:]
                jets      = f["jet_features"][:]
            else:
                key = list(f.keys())[0]
                particles = f[key][:]
                jets      = np.zeros((len(particles), 3), dtype=np.float32)

        np.save(out_file, particles.astype(np.float32))
        np.save(jet_file, jets.astype(np.float32))
        print(f"  {name}: {particles.shape}")

    return True


def verify_and_print_stats(out_dir: Path, n_particles: int):
    """Print basic statistics to verify the data looks right."""
    print(f"\n{'='*60}")
    print(f"JetNet{n_particles} dataset statistics:")
    print(f"{'='*60}")
    for jtype in JET_TYPES:
        name = JET_NAMES[jtype]
        f = out_dir / f"{name}.npy"
        if not f.exists():
            print(f"  {name}: MISSING")
            continue
        data = np.load(f)  # (N, n_part, 4): [eta_rel, phi_rel, pt_rel, mask]
        masks = data[:, :, 3]
        n_real = masks.sum(axis=1).mean()
        print(f"  {name:10s}: {data.shape}  "
              f"mean_npart={n_real:.1f}  "
              f"pt_rel_max={data[:,:,2].max():.4f}  "
              f"eta_range=[{data[:,:,0].min():.2f}, {data[:,:,0].max():.2f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True,
                   help="Output directory for JetNet data")
    p.add_argument("--version", choices=["30", "150", "both"], default="both",
                   help="Which JetNet version to download (default: both)")
    p.add_argument("--method", choices=["auto", "jetnet", "zenodo"], default="auto",
                   help="Download method")
    args = p.parse_args()

    out_base = Path(args.out_dir)
    versions = ([30, 150] if args.version == "both" else
                [30] if args.version == "30" else [150])

    for n_part in versions:
        out_dir = out_base / f"jetnet{n_part}"
        print(f"\n{'='*60}")
        print(f"Downloading JetNet{n_part} → {out_dir}")
        print(f"{'='*60}")

        ok = False
        if args.method in ("auto", "jetnet"):
            ok = download_via_jetnet_package(out_dir, n_part)
        if not ok and args.method in ("auto", "zenodo"):
            ok = download_via_zenodo(out_dir, n_part)
        if not ok:
            print(f"ERROR: failed to download JetNet{n_part}")
            sys.exit(1)

        verify_and_print_stats(out_dir, n_part)

    print(f"\nDone. Data at: {out_base}")
    print("Next step: python training/prepare_fpcd_training.py "
          f"--jetnet-dir {out_base}")


if __name__ == "__main__":
    main()
