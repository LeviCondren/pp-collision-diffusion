#!/usr/bin/env python3
"""
Concatenate per-rank inference outputs into one npz per process type.

Usage:
    python concat_infer.py [--run_name parton_v1_1node] [--world_size 4]

Reads:  {out_dir}/{proc}_rank{r:02d}_of{world_size:02d}.npz  for r in range(world_size)
Writes: {out_dir}/{proc}_100k.npz
"""

import os, argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--run_name',   type=str, default='parton_v1_1node')
p.add_argument('--world_size', type=int, default=4)
p.add_argument('--data_dir',   type=str,
               default='/pscratch/sd/l/lcondren/MCsim/full_event_fpcd')
p.add_argument('--out_dir',    type=str, default=None)
args = p.parse_args()

out_dir = args.out_dir or (
    f'{args.data_dir}/checkpoints_pet_pp/{args.run_name}/infer_100k')

for proc in ['dijet', 'zjets']:
    files = [f'{out_dir}/{proc}_rank{r:02d}_of{args.world_size:02d}.npz'
             for r in range(args.world_size)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f'{proc}: missing {len(missing)} file(s): {missing}')
        continue

    arrs = [np.load(f) for f in files]
    combined = {k: np.concatenate([a[k] for a in arrs], axis=0)
                for k in arrs[0].files}
    for a in arrs:
        a.close()

    out_file = f'{out_dir}/{proc}_100k.npz'
    np.savez_compressed(out_file, **combined)
    n = len(combined['parts_truth'])
    print(f'{proc}: {n} events -> {out_file}')
