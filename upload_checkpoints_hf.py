#!/usr/bin/env python3
"""
Upload all 4 PET-pp model checkpoints to Hugging Face Hub.

Usage:
    export HF_TOKEN=<your_token>
    conda run -n base python3 upload_checkpoints_hf.py

Repo: https://huggingface.co/Levicondren/pet-pp-checkpoints  (private)

Models uploaded:
  e029_sm_stage2_ep178/   -- SM 4-proc stage-2 (layers4), epoch 178 [complete]
  e030_sm_stage1_ep153/   -- SM 5-proc stage-1, epoch 153 [complete]
  e031_wprime_stage2/     -- W' stage-2 (layers4), mid-training [epoch from training_state.json]
  e032_wprime_stage1/     -- W' stage-1, mid-training [epoch from training_state.json]
"""

import os, json
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID   = "Levicondren/pet-pp-checkpoints"
REPO_TYPE = "model"

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("Set HF_TOKEN environment variable before running.")

api = HfApi(token=token)

# ── Create repo (no-op if already exists) ────────────────────────────────────
api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True)
print(f"Repo ready: https://huggingface.co/{REPO_ID}")

# ── File manifest: (local_path, path_in_repo) ────────────────────────────────
SM4   = "/pscratch/sd/l/lcondren/MCsim/full_event_mixed"
SM5   = "/pscratch/sd/l/lcondren/MCsim/full_event_mixed"
WP    = "/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi"

def _epoch(ckpt_dir):
    ts = Path(ckpt_dir) / "training_state.json"
    return json.loads(ts.read_text())["epochs_done"] if ts.exists() else "?"

e029_ep = _epoch(f"{SM4}/checkpoints_sm_4proc/sm_4proc_event_c_layers4_full")
e030_ep = _epoch(f"{SM5}/checkpoints_sm_5proc/sm_5proc_event_c_stage1")
e031_ep = _epoch(f"{WP}/checkpoints_bsm_grid/bsm_grid_event_c_layers4_mpi")
e032_ep = _epoch(f"{WP}/checkpoints_bsm_grid/bsm_grid_event_c_stage1_mpi")

print(f"Epochs: E029={e029_ep}  E030={e030_ep}  E031={e031_ep}  E032={e032_ep}")

MANIFEST = [
    # E029 — SM stage-2 (layers4), complete
    (f"{SM4}/checkpoints_sm_4proc/sm_4proc_event_c_layers4_full/pet_pp.weights.h5",
     f"e029_sm_stage2_ep{e029_ep}/pet_pp.weights.h5"),
    (f"{SM4}/checkpoints_sm_4proc/sm_4proc_event_c_layers4_full/training_state.json",
     f"e029_sm_stage2_ep{e029_ep}/training_state.json"),
    (f"{SM4}/normalisation_stats_sm4proc.json",
     f"e029_sm_stage2_ep{e029_ep}/normalisation_stats_sm4proc.json"),
    (f"{SM4}/normalisation_stats_event_c_sm4proc.json",
     f"e029_sm_stage2_ep{e029_ep}/normalisation_stats_event_c_sm4proc.json"),

    # E030 — SM stage-1, complete
    (f"{SM5}/checkpoints_sm_5proc/sm_5proc_event_c_stage1/pet_pp.weights.h5",
     f"e030_sm_stage1_ep{e030_ep}/pet_pp.weights.h5"),
    (f"{SM5}/checkpoints_sm_5proc/sm_5proc_event_c_stage1/training_state.json",
     f"e030_sm_stage1_ep{e030_ep}/training_state.json"),
    (f"{SM5}/checkpoints_sm_5proc/normalisation_stats_sm5proc_stage1.json",
     f"e030_sm_stage1_ep{e030_ep}/normalisation_stats_sm5proc_stage1.json"),

    # E031 — W' stage-2 (layers4), mid-training
    (f"{WP}/checkpoints_bsm_grid/bsm_grid_event_c_layers4_mpi/pet_pp.weights.h5",
     f"e031_wprime_stage2_ep{e031_ep}/pet_pp.weights.h5"),
    (f"{WP}/checkpoints_bsm_grid/bsm_grid_event_c_layers4_mpi/training_state.json",
     f"e031_wprime_stage2_ep{e031_ep}/training_state.json"),
    (f"{WP}/normalisation_stats.json",
     f"e031_wprime_stage2_ep{e031_ep}/normalisation_stats.json"),
    (f"{WP}/normalisation_stats_event_c.json",
     f"e031_wprime_stage2_ep{e031_ep}/normalisation_stats_event_c.json"),

    # E032 — W' stage-1, mid-training
    (f"{WP}/checkpoints_bsm_grid/bsm_grid_event_c_stage1_mpi/pet_pp.weights.h5",
     f"e032_wprime_stage1_ep{e032_ep}/pet_pp.weights.h5"),
    (f"{WP}/checkpoints_bsm_grid/bsm_grid_event_c_stage1_mpi/training_state.json",
     f"e032_wprime_stage1_ep{e032_ep}/training_state.json"),
    (f"{WP}/normalisation_stats_event_c.json",
     f"e032_wprime_stage1_ep{e032_ep}/normalisation_stats_event_c.json"),
]

# ── Upload ────────────────────────────────────────────────────────────────────
for local, remote in MANIFEST:
    if not Path(local).exists():
        print(f"  MISSING (skipping): {local}")
        continue
    size_mb = Path(local).stat().st_size / 1e6
    print(f"  uploading {remote}  ({size_mb:.1f} MB) ...", flush=True)
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print(f"    done.")

print(f"\nAll done. View at: https://huggingface.co/{REPO_ID}")
print("Download on another host:")
print(f"  huggingface-cli download {REPO_ID} --repo-type model --token $HF_TOKEN")
