#!/usr/bin/env python3
"""Extract a compact subset of SM process HDF5 files for transfer to HPC3.

Reads the real source files (not the symlinks in full_event_mixed/) and
concatenates three slices:
  [0 : n_train]              → training
  [val_start : val_start+n_val]         → validation
  [hold_start : hold_start+n_holdout]   → holdout

The output files have indices:
  [0 : n_train]              → training   (pass --val_start n_train to train script)
  [n_train : n_train+n_val]  → validation
  [n_train+n_val : end]      → holdout    (pass --holdout_start n_train+n_val to infer)

Default: 20k train + 10k val + 10k holdout = 40k events per file (~160 MB each, ~640 MB total).

Usage:
  python3 extract_sm_slim.py --out_dir /path/to/output
  rsync -avP /path/to/output/ hpc3:/pub/lcondren/MCsim/full_event_mixed/
"""

import h5py, numpy as np, argparse, os, sys, time

SOURCE_MAP = {
    'dijet': '/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/dijet.hdf5',
    'zjets': '/pscratch/sd/l/lcondren/MCsim/full_event_fpcd/zjets.hdf5',
    'ttbar': '/pscratch/sd/l/lcondren/MCsim/full_event_fpcd_more/ttbar.hdf5',
    'wjets': '/pscratch/sd/l/lcondren/MCsim/full_event_fpcd_more/wjets.hdf5',
}
PROCESSES = ['dijet', 'ttbar', 'wjets', 'zjets']


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir',     default='/tmp/sm_slim',
                   help='Directory to write compact HDF5 files')
    p.add_argument('--n_train',     type=int, default=20000,
                   help='Training events to keep from [0:n_train]')
    p.add_argument('--n_val',       type=int, default=10000,
                   help='Validation events to keep')
    p.add_argument('--n_holdout',   type=int, default=10000,
                   help='Holdout events to keep')
    p.add_argument('--val_start',   type=int, default=480000,
                   help='Source val start index (default 480000)')
    p.add_argument('--hold_start',  type=int, default=490000,
                   help='Source holdout start index (default 490000)')
    p.add_argument('--processes',   nargs='+', default=PROCESSES,
                   help='Processes to extract (default: all 4)')
    return p.parse_args()


def extract_process(proc, src_path, out_path, args):
    slices_src = [
        (0,                args.n_train),
        (args.val_start,   args.val_start  + args.n_val),
        (args.hold_start,  args.hold_start + args.n_holdout),
    ]

    t0 = time.perf_counter()
    with h5py.File(src_path, 'r') as fin:
        n_avail = fin['particle_features'].shape[0]
        for i, (s, e) in enumerate(slices_src):
            if e > n_avail:
                print(f"  WARNING: slice {i} [{s}:{e}] clamped to [{s}:{n_avail}]")
                slices_src[i] = (s, n_avail)

        print(f"  source events: {n_avail}")
        print(f"  extracting: {slices_src}")

        with h5py.File(out_path, 'w') as fout:
            for key in fin.keys():
                ds = fin[key]
                parts = [ds[s:e] for s, e in slices_src]
                data  = np.concatenate(parts, axis=0)
                fout.create_dataset(key, data=data,
                                    compression='gzip', compression_opts=4)
                print(f"  {key}: {data.shape}  dtype={data.dtype}")

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  → {out_path}  ({size_mb:.0f} MB)  in {elapsed:.1f}s")


def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)

    val_out  = args.n_train
    hold_out = args.n_train + args.n_val
    total    = args.n_train + args.n_val + args.n_holdout

    print(f"Compact file layout ({total} events/file):")
    print(f"  [0 : {args.n_train}]  = training")
    print(f"  [{val_out} : {hold_out}] = validation")
    print(f"  [{hold_out} : {total}] = holdout")
    print()
    print("HPC3 submit args to use:")
    print(f"  --val_start {val_out}  --holdout_start {hold_out}")
    print()

    for proc in args.processes:
        src = SOURCE_MAP[proc]
        dst = os.path.join(args.out_dir, f'{proc}.hdf5')
        if not os.path.exists(src):
            print(f"[{proc}] SKIP — source not found: {src}")
            continue
        print(f"[{proc}] {src}")
        extract_process(proc, src, dst, args)
        print()

    print("=== Done ===")
    print(f"\nTransfer command:")
    print(f"  rsync -avP --progress {args.out_dir}/ "
          f"lcondren@hpc3.rcic.uci.edu:/pub/lcondren/MCsim/full_event_mixed/")
    print(f"\nAlso copy normalisation stats (if already computed on NERSC):")
    print(f"  scp /pscratch/sd/l/lcondren/MCsim/full_event_mixed/normalisation_stats_sm4proc.json \\")
    print(f"      /pscratch/sd/l/lcondren/MCsim/full_event_mixed/normalisation_stats_event_c_sm4proc.json \\")
    print(f"      lcondren@hpc3.rcic.uci.edu:/pub/lcondren/MCsim/full_event_mixed/")


if __name__ == '__main__':
    main()
