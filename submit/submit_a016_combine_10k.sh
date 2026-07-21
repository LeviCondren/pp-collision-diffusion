#!/bin/bash
#SBATCH --account=m2616
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=a016_combine_10k
#SBATCH --output=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/logs/%j_a016_combine_10k.out
#SBATCH --error=/pscratch/sd/l/lcondren/MCsim/full_event_mixed/logs/%j_a016_combine_10k.err

# A016 — Combine holdout inference (A013/A013-t) with val5k inference (A015/A015-t).
# For each model, concatenates 5k holdout events + 5k new MC events = 10k combined.
# Input:
#   E029 holdout: checkpoints_sm_4proc/sm_4proc_event_c_layers4_full/infer_holdout_truth_ep178/
#   E029 val5k:   checkpoints_sm_4proc/sm_4proc_event_c_layers4_full/infer_val5k_ep178/
#   E030 holdout: checkpoints_sm_5proc/sm_5proc_event_c_stage1/infer_holdout_truth_ep153/
#   E030 val5k:   checkpoints_sm_5proc/sm_5proc_event_c_stage1/infer_val5k_ep153/
# Output:
#   E029 combined: checkpoints_sm_4proc/sm_4proc_event_c_layers4_full/infer_combined_10k_ep178/
#   E030 combined: checkpoints_sm_5proc/sm_5proc_event_c_stage1/infer_combined_10k_ep153/

SM_DIR=/pscratch/sd/l/lcondren/MCsim/full_event_mixed

echo "Job ${SLURM_JOB_ID} started: $(date)"

python3 - << 'EOF'
import numpy as np
import os

SM_DIR = "/pscratch/sd/l/lcondren/MCsim/full_event_mixed"
PROCS  = ["dijet", "ttbar", "wjets", "zjets"]

PAIRS = [
    # (holdout_dir, val5k_dir, combined_dir, label)
    (
        f"{SM_DIR}/checkpoints_sm_4proc/sm_4proc_event_c_layers4_full/infer_holdout_truth_ep178",
        f"{SM_DIR}/checkpoints_sm_4proc/sm_4proc_event_c_layers4_full/infer_val5k_ep178",
        f"{SM_DIR}/checkpoints_sm_4proc/sm_4proc_event_c_layers4_full/infer_combined_10k_ep178",
        "E029",
    ),
    (
        f"{SM_DIR}/checkpoints_sm_5proc/sm_5proc_event_c_stage1/infer_holdout_truth_ep153",
        f"{SM_DIR}/checkpoints_sm_5proc/sm_5proc_event_c_stage1/infer_val5k_ep153",
        f"{SM_DIR}/checkpoints_sm_5proc/sm_5proc_event_c_stage1/infer_combined_10k_ep153",
        "E030",
    ),
]

for holdout_dir, val5k_dir, out_dir, label in PAIRS:
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {label} ===")
    for proc in PROCS:
        path_a = f"{holdout_dir}/{proc}.npz"
        path_b = f"{val5k_dir}/{proc}.npz"
        if not os.path.exists(path_a):
            print(f"  [{proc}] MISSING holdout: {path_a}")
            continue
        if not os.path.exists(path_b):
            print(f"  [{proc}] MISSING val5k: {path_b}")
            continue
        a = np.load(path_a)
        b = np.load(path_b)
        combined = {}
        n_a = n_b = 0
        for key in a.keys():
            if a[key].ndim == 0:
                combined[key] = a[key]
            else:
                combined[key] = np.concatenate([a[key], b[key]], axis=0)
                if n_a == 0:
                    n_a = a[key].shape[0]
                    n_b = b[key].shape[0]
        out_path = f"{out_dir}/{proc}.npz"
        np.savez(out_path, **combined)
        n_tot = n_a + n_b
        print(f"  {proc}: {n_a} + {n_b} = {n_tot} events → {out_path}")

print("\nDone.")
EOF

echo "Job ${SLURM_JOB_ID} finished: $(date)"
echo "=== E029 combined ==="
ls -lh ${SM_DIR}/checkpoints_sm_4proc/sm_4proc_event_c_layers4_full/infer_combined_10k_ep178/
echo "=== E030 combined ==="
ls -lh ${SM_DIR}/checkpoints_sm_5proc/sm_5proc_event_c_stage1/infer_combined_10k_ep153/
