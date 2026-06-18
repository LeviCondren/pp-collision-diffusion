#!/usr/bin/env python3
"""
parnassus_wrapper.py — BSM diffusion output → Parnassus full-event detector surrogate.

IMPORTANT: run in pipeline_copy-gpu2 conda environment:
    conda activate pipeline_copy-gpu2
    python3 parnassus_wrapper.py --input_file <path-to-NPZ> [options]

Input NPZ (produced by infer_bsm_grid.py):
    parts_gen  (N, 500, 6)  physical particle features [eta, sin_phi, cos_phi, log_pT, pdg_norm, charge]
    mask_gen   (N, 500)     float32 mask (1.0 = valid particle)
    mass_x, mass_y          scalar mass metadata (GeV)

Output HDF5 at {output_dir}/recoparticles.hdf5:
    reco_particles   (N, 600, 3)  [pT_GeV, eta, phi]  Parnassus detector output
    reco_mask        (N, 600)     bool
    hadron_particles (N, 500, 3)  [pT_GeV, eta, phi]  hadron-level input (for comparison)
    hadron_mask      (N, 500)     bool
    Attributes: m_X, m_Y, event_count
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import h5py

# ── sys.path setup for Parnassus model imports ─────────────────────────────────

_THIS_DIR       = Path(__file__).resolve().parent
_DARKPHOTON_DIR = _THIS_DIR.parent.parent / "SM_vs_Darkphoton"
_PARNASSUS_DIR  = Path("/pscratch/sd/l/lcondren/MCsim/Parnassus")

for _d in [str(_DARKPHOTON_DIR), str(_PARNASSUS_DIR), str(_PARNASSUS_DIR / "models")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch
from train_full_event_detector import FullEventFlowLightning, DEFAULT_CONFIG

# ── Constants ──────────────────────────────────────────────────────────────────

_FULL_EVENT_CKPT   = Path(
    "/pscratch/sd/l/lcondren/MCsim/full_event_detector_data/checkpoints/"
    "fm_full_event-epoch=034-val_loss=1.0545.ckpt"
)
_DEFAULT_OUT_ROOT  = Path("/pscratch/sd/l/lcondren/MCsim/parnassus_output")
_MAX_PARTICLES     = DEFAULT_CONFIG["max_particles"]   # 600


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Chain BSM diffusion output through Parnassus full-event detector surrogate"
    )
    p.add_argument("--input_file",  required=True,
                   help="NPZ produced by infer_bsm_grid.py")
    p.add_argument("--output_dir",  default=None,
                   help="HDF5 output directory "
                        "(default: /pscratch/.../parnassus_output/{run_name}/{mX}_{mY}/)")
    p.add_argument("--batch_size",  type=int, default=64,
                   help="Events per Parnassus forward pass (default: 64)")
    p.add_argument("--device",      default="cuda",
                   help="cuda or cpu (default: cuda)")
    p.add_argument("--max_events",  type=int, default=None,
                   help="Cap on events to process (default: all)")
    p.add_argument("--n_steps",     type=int, default=25,
                   help="ODE integration steps (default: 25)")
    return p.parse_args()


# ── Preprocessing ──────────────────────────────────────────────────────────────

def _normalize_phi(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi))


def preprocess_diffusion_to_parnassus(
    parts_gen: np.ndarray,   # (N, 500, 6) physical
    mask_gen:  np.ndarray,   # (N, 500)    float32
):
    """
    Convert infer_bsm_grid.py output to FullEventDataset-compatible tensors.

    Input feature order (physical, already denormalized by infer_bsm_grid.py):
        [eta, sin_phi, cos_phi, log_pT_GeV, pdg_norm, charge]

    Preprocessing exactly follows FullEventDataset.__init__ in train_full_event_detector.py:
        pT  → log(pT_GeV * 1000) then per-event z-score
        eta → per-event z-score
        phi → atan2(sin, cos) then per-event z-score
        charge_class → 1 if |charge| > 0.5 else 0

    Returns
    -------
    truth_arr    : (N, 600, 4) float32  [pT_norm, eta_norm, phi_norm, charge_class]
    mask_arr     : (N, 600, 2) bool     [:,:,0]=truth_mask, [:,:,1]=pflow_mask
    scale_arr    : (N, 6)      float32  [pt_mean, pt_std, eta_mean, eta_std, phi_mean, phi_std]
    hadron_phys  : (N, 500, 3) float32  [pT_GeV, eta, phi] for audit output
    mask_bool    : (N, 500)    bool
    """
    N  = parts_gen.shape[0]
    MP = _MAX_PARTICLES

    # Recover physical quantities from diffusion features
    eta         = parts_gen[..., 0].copy()                            # (N, 500)
    log_pT_GeV  = parts_gen[..., 3].copy()                           # (N, 500) base-e
    sin_phi     = parts_gen[..., 1]
    cos_phi     = parts_gen[..., 2]
    charge      = parts_gen[..., 5]

    pT_GeV      = np.exp(log_pT_GeV).astype(np.float32)              # (N, 500)
    phi         = _normalize_phi(np.arctan2(sin_phi, cos_phi))       # (N, 500)
    charge_cls  = (np.abs(charge) > 0.5).astype(np.float32)          # (N, 500)
    log_pT_MeV  = (log_pT_GeV + np.log(1000.)).astype(np.float32)    # log(pT_GeV * 1000)
    mask_bool   = (mask_gen > 0.5)                                    # (N, 500) bool

    # Per-event normalization stats from masked truth particles
    # (replicates FullEventDataset._masked_mean / _masked_std)
    tm = mask_bool.astype(np.float32)  # (N, 500)

    def masked_mean(x, m):
        return (x * m).sum(1) / np.clip(m.sum(1), 1., None)         # (N,)

    def masked_std(x, m, mu):
        return np.sqrt(
            ((x - mu[:, None]) ** 2 * m).sum(1)
            / np.clip(m.sum(1) - 1., 1., None)
        )                                                             # (N,)

    def safe_std(v):
        return np.where(v < 1e-6, 1., v)

    pt_mean  = masked_mean(log_pT_MeV, tm)
    pt_std   = safe_std(masked_std(log_pT_MeV, tm, pt_mean))
    eta_mean = masked_mean(eta,         tm)
    eta_std  = safe_std(masked_std(eta,         tm, eta_mean))
    phi_mean = masked_mean(phi,         tm)
    phi_std  = safe_std(masked_std(phi,         tm, phi_mean))

    # Normalize (N, 500)
    pt_norm  = ((log_pT_MeV - pt_mean[:, None])  / pt_std[:, None]).astype(np.float32)
    eta_norm = ((eta         - eta_mean[:, None]) / eta_std[:, None]).astype(np.float32)
    phi_norm = ((phi         - phi_mean[:, None]) / phi_std[:, None]).astype(np.float32)

    # Build truth_arr and mask_arr (N, 600, 4) and (N, 600, 2)
    # Valid particles are already contiguous from index 0 (mask_gen is a prefix mask
    # produced by infer_bsm_grid.py), so no sorting is needed.
    nt = min(500, MP)
    truth_arr = np.zeros((N, MP, 4), dtype=np.float32)
    mask_arr  = np.zeros((N, MP, 2), dtype=bool)

    truth_arr[:, :nt, 0] = pt_norm[:, :nt]
    truth_arr[:, :nt, 1] = eta_norm[:, :nt]
    truth_arr[:, :nt, 2] = phi_norm[:, :nt]
    truth_arr[:, :nt, 3] = charge_cls[:, :nt]
    mask_arr[:, :nt, 0]  = mask_bool[:, :nt]   # truth mask
    mask_arr[:, :nt, 1]  = mask_bool[:, :nt]   # pflow mask = truth mask (1:1 mapping)

    scale_arr = np.stack(
        [pt_mean, pt_std, eta_mean, eta_std, phi_mean, phi_std], axis=1
    ).astype(np.float32)  # (N, 6)

    # Hadron-level physical arrays for audit output
    hadron_phys = np.zeros((N, 500, 3), dtype=np.float32)
    hadron_phys[:, :, 0] = pT_GeV
    hadron_phys[:, :, 1] = eta
    hadron_phys[:, :, 2] = phi

    return truth_arr, mask_arr, scale_arr, hadron_phys, mask_bool


# ── Inference ──────────────────────────────────────────────────────────────────

def run_parnassus_batched(
    model:      FullEventFlowLightning,
    truth_arr:  np.ndarray,   # (N, 600, 4)
    mask_arr:   np.ndarray,   # (N, 600, 2) bool
    scale_arr:  np.ndarray,   # (N, 6)
    batch_size: int,
    device:     str,
    n_steps:    int,
) -> np.ndarray:
    """Run Parnassus in minibatches. Returns x_out (N, 600, 3) in normalized space."""
    N = truth_arr.shape[0]
    x_out_all = np.zeros((N, _MAX_PARTICLES, 3), dtype=np.float32)
    n_batches = (N + batch_size - 1) // batch_size

    for i in range(n_batches):
        s = i * batch_size
        e = min(s + batch_size, N)
        truth_t = torch.from_numpy(truth_arr[s:e]).to(device)
        mask_t  = torch.from_numpy(mask_arr[s:e]).to(device)
        scale_t = torch.from_numpy(scale_arr[s:e]).to(device)

        with torch.no_grad():
            x_out = model.sample(truth_t, mask_t, scale_t, n_steps=n_steps)

        x_out_all[s:e] = x_out.cpu().numpy()
        if (i + 1) % 10 == 0 or i == n_batches - 1:
            print(f"  batch {i + 1}/{n_batches}  events [{s}, {e})", flush=True)

    return x_out_all


# ── Denormalization ────────────────────────────────────────────────────────────

def denormalize_parnassus_output(
    x_out:     np.ndarray,   # (N, 600, 3) normalized [pT, eta, phi]
    scale_arr: np.ndarray,   # (N, 6)
    reco_mask: np.ndarray,   # (N, 600) bool
) -> np.ndarray:
    """
    Undo per-event normalization to recover physical [pT_GeV, eta, phi].

    scale_arr columns: [pt_mean, pt_std, eta_mean, eta_std, phi_mean, phi_std]
    where pt_mean/std are stats of log(pT_MeV).
    """
    pt_mean  = scale_arr[:, 0:1]   # (N, 1) broadcast over 600
    pt_std   = scale_arr[:, 1:2]
    eta_mean = scale_arr[:, 2:3]
    eta_std  = scale_arr[:, 3:4]
    phi_mean = scale_arr[:, 4:5]
    phi_std  = scale_arr[:, 5:6]

    logpt_reco = x_out[..., 0] * pt_std  + pt_mean     # log(pT_MeV)
    pT_reco    = np.exp(logpt_reco) / 1000.             # GeV
    eta_reco   = x_out[..., 1] * eta_std + eta_mean
    phi_raw    = x_out[..., 2] * phi_std + phi_mean
    phi_reco   = _normalize_phi(phi_raw)                # wrap to (-π, π)

    reco_phys = np.stack([pT_reco, eta_reco, phi_reco], axis=-1).astype(np.float32)
    reco_phys[~reco_mask] = 0.
    return reco_phys


# ── Summary printing ───────────────────────────────────────────────────────────

def _print_summary(label: str, particles: np.ndarray, mask: np.ndarray):
    valid = mask.astype(bool)
    mult  = valid.sum(axis=1).astype(float)
    pT    = particles[..., 0][valid]
    eta   = particles[..., 1][valid]
    phi   = particles[..., 2][valid]
    print(f"  [{label}]")
    print(f"    mean multiplicity : {mult.mean():.1f} ± {mult.std():.1f}")
    if pT.size > 0:
        print(f"    pT  [GeV] : mean={pT.mean():.3f}  std={pT.std():.3f}  "
              f"min={pT.min():.4f}  max={pT.max():.1f}")
        print(f"    eta       : mean={eta.mean():.3f}  std={eta.std():.3f}  "
              f"min={eta.min():.2f}  max={eta.max():.2f}")
        print(f"    phi       : mean={phi.mean():.3f}  std={phi.std():.3f}  "
              f"min={phi.min():.2f}  max={phi.max():.2f}")
        n_nan = np.isnan(pT).sum() + np.isnan(eta).sum() + np.isnan(phi).sum()
        n_inf = np.isinf(pT).sum() + np.isinf(eta).sum() + np.isinf(phi).sum()
        if n_nan:
            print(f"  *** WARNING: {n_nan} NaN values in output ***")
        if n_inf:
            print(f"  *** WARNING: {n_inf} Inf values in output ***")
        if (pT < 0).any():
            print(f"  *** WARNING: {(pT < 0).sum()} negative pT values ***")
    else:
        print("    (no valid particles)")


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_output(reco_particles: np.ndarray, reco_mask: np.ndarray):
    valid = reco_mask.astype(bool)
    pT  = reco_particles[..., 0][valid]
    eta = reco_particles[..., 1][valid]
    phi = reco_particles[..., 2][valid]

    if np.isnan(pT).any():
        raise ValueError("NaN values in reco pT — check preprocessing and model output")
    if np.isinf(pT).any():
        raise ValueError("Inf values in reco pT — check preprocessing and model output")
    if np.isnan(eta).any():
        raise ValueError("NaN values in reco eta")
    if not ((phi >= -np.pi - 0.01).all() and (phi <= np.pi + 0.01).all()):
        raise ValueError(f"phi_reco out of [-π, π]: range=[{phi.min():.3f}, {phi.max():.3f}]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = _parse()

    in_path = Path(args.input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # ── Step 1: Load input NPZ ─────────────────────────────────────────────────
    print(f"\n[1/5] Loading input NPZ: {in_path}")
    d = np.load(in_path, allow_pickle=True)
    print(f"  Keys: {list(d.keys())}")
    for k in d.keys():
        v = d[k]
        print(f"    {k}: shape={v.shape}  dtype={v.dtype}", end="")
        if v.ndim >= 1 and v.size > 0:
            print(f"  range=[{v.min():.3f}, {v.max():.3f}]", end="")
        print()

    parts_gen = d["parts_gen"].astype(np.float32)
    mask_gen  = d["mask_gen"].astype(np.float32)
    mass_x    = float(d["mass_x"].flat[0])
    mass_y    = float(d["mass_y"].flat[0])

    N = parts_gen.shape[0]
    if args.max_events is not None and N > args.max_events:
        parts_gen = parts_gen[:args.max_events]
        mask_gen  = mask_gen[:args.max_events]
        N = args.max_events
        print(f"  Capped to {N} events (--max_events)")

    print(f"  Processing {N} events  m_X={mass_x:.0f} GeV  m_Y={mass_y:.0f} GeV")

    # Resolve output directory
    parent = in_path.parent
    run_name = parent.parent.name if parent.name == "infer" else parent.name
    tag = f"mX{mass_x:04.0f}_mY{mass_y:04.0f}"
    out_dir = Path(args.output_dir) if args.output_dir else \
              _DEFAULT_OUT_ROOT / run_name / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recoparticles.hdf5"
    print(f"  Output: {out_path}")

    # ── Step 2: Preprocess ─────────────────────────────────────────────────────
    print(f"\n[2/5] Preprocessing diffusion output → Parnassus format")
    truth_arr, mask_arr, scale_arr, hadron_phys, mask_bool = \
        preprocess_diffusion_to_parnassus(parts_gen, mask_gen)
    print(f"  truth_arr : {truth_arr.shape}  dtype={truth_arr.dtype}")
    print(f"  mask_arr  : {mask_arr.shape}   dtype={mask_arr.dtype}")
    print(f"  scale_arr : {scale_arr.shape}  dtype={scale_arr.dtype}")
    print(f"  scale_arr sample (event 0): {scale_arr[0]}")
    _print_summary("hadron input", hadron_phys, mask_bool.astype(np.float32))

    # ── Step 3: Load Parnassus model ───────────────────────────────────────────
    print(f"\n[3/5] Loading Parnassus model")
    print(f"  Checkpoint: {_FULL_EVENT_CKPT}")
    if not _FULL_EVENT_CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {_FULL_EVENT_CKPT}")

    device = args.device if (args.device == "cpu" or not torch.cuda.is_available()) \
             else "cuda"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("  WARNING: CUDA not available, falling back to CPU")

    model = FullEventFlowLightning.load_from_checkpoint(
        str(_FULL_EVENT_CKPT), config=DEFAULT_CONFIG, map_location=device
    )
    model.eval().to(device)
    print(f"  Loaded on device={device}")

    # ── Step 4: Run Parnassus inference ────────────────────────────────────────
    print(f"\n[4/5] Running Parnassus inference  "
          f"(batch_size={args.batch_size}, n_steps={args.n_steps})")
    x_out = run_parnassus_batched(
        model, truth_arr, mask_arr, scale_arr,
        args.batch_size, device, args.n_steps
    )
    print(f"  x_out range: [{x_out.min():.3f}, {x_out.max():.3f}]")

    reco_mask      = mask_arr[:, :, 1]                            # (N, 600) bool
    reco_particles = denormalize_parnassus_output(x_out, scale_arr, reco_mask)
    _print_summary("reco output", reco_particles, reco_mask)
    _validate_output(reco_particles, reco_mask)

    # ── Step 5: Save HDF5 ──────────────────────────────────────────────────────
    print(f"\n[5/5] Saving HDF5: {out_path}")
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("reco_particles",   data=reco_particles, compression="gzip")
        hf.create_dataset("reco_mask",        data=reco_mask,      compression="gzip")
        hf.create_dataset("hadron_particles", data=hadron_phys,    compression="gzip")
        hf.create_dataset("hadron_mask",      data=mask_bool,      compression="gzip")
        hf.attrs["m_X"]         = mass_x
        hf.attrs["m_Y"]         = mass_y
        hf.attrs["event_count"] = N

    print(f"  reco_particles  : {reco_particles.shape}")
    print(f"  reco_mask       : {reco_mask.shape}")
    print(f"  hadron_particles: {hadron_phys.shape}")
    print(f"  hadron_mask     : {mask_bool.shape}")
    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    main()
