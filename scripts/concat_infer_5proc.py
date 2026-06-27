#!/usr/bin/env python3
"""
Concatenate per-rank inference outputs into one npz per process.
Extends concat_infer.py to handle all 5 processes.

Usage:
    python concat_infer_5proc.py [--run_name parton_mixed_5proc] [--world_size 1]
"""

import os, argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--run_name',   type=str, default='parton_mixed_5proc')
p.add_argument('--world_size', type=int, default=1)
p.add_argument('--data_dir',   type=str,
               default='/pscratch/sd/l/lcondren/MCsim/full_event_mixed')
p.add_argument('--out_dir',    type=str, default=None)
args = p.parse_args()

out_dir = args.out_dir or f'{args.data_dir}/checkpoints/{args.run_name}/infer_20k'

for proc in ['dijet', 'zjets', 'ttbar', 'wjets', 'wprime']:
    files = [f'{out_dir}/{proc}_rank{r:02d}_of{args.world_size:02d}.npz'
             for r in range(args.world_size)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f'{proc}: missing files: {missing}')
        continue

    arrs = [np.load(f) for f in files]
    combined = {k: np.concatenate([a[k] for a in arrs], axis=0) for k in arrs[0].files}
    for a in arrs:
        a.close()

    out_file = f'{out_dir}/{proc}_20k.npz'
    np.savez_compressed(out_file, **combined)
    print(f'{proc}: {len(combined["parts_truth"])} events -> {out_file}')
