#!/usr/bin/env python3
"""
Standalone PET_pp inference for one rank of a parallel multi-GPU job.

Usage (single GPU test):
    python infer_pp.py --rank 0 --world_size 1 --num_steps 50

Usage (called by launch_infer_4gpu.sh for 4-GPU parallel run):
    python infer_pp.py --rank 2 --world_size 4 --gpu_id 2 --num_steps 50

Each rank saves:
    {out_dir}/{proc}_rank{rank:02d}_of{world_size:02d}.npz
with keys: parts_truth, parts_gen, mask, mask_gen, parton_feat
"""

import os, sys, json, argparse, time
import numpy as np

# ── Args (parse before any TF import so CUDA_VISIBLE_DEVICES is set first) ───
def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--rank',       type=int,
                   default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    p.add_argument('--world_size', type=int,
                   default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)))
    p.add_argument('--gpu_id',     type=int, default=None,
                   help='CUDA device index for this process (default: rank mod n_gpus)')
    p.add_argument('--num_steps',  type=int, default=50,
                   help='DDPM steps (training used 500; 50 is ~10x faster with minimal quality loss)')
    p.add_argument('--chunk_size', type=int, default=200,
                   help='Events per generation chunk (notebook used 20; 200 fills the A100 better)')
    p.add_argument('--val_start',  type=int, default=400000,
                   help='HDF5 row index where the validation split begins')
    p.add_argument('--n_total',    type=int, default=100000,
                   help='Total reserved events to distribute across world_size ranks')
    p.add_argument('--npart',      type=int, default=500)
    p.add_argument('--proj_dim',   type=int, default=128)
    p.add_argument('--num_layers', type=int, default=8)
    p.add_argument('--run_name',   type=str, default='parton_v1_nlo_npart')
    p.add_argument('--data_dir',   type=str,
                   default='/pscratch/sd/l/lcondren/MCsim/full_event_fpcd')
    p.add_argument('--stats_dir',  type=str, default=None,
                   help='Directory containing normalisation_stats.json (default: data_dir)')
    p.add_argument('--ckpt_dir',   type=str, default=None,
                   help='Checkpoint base directory (default: {data_dir}/checkpoints_pet_pp)')
    p.add_argument('--out_dir',    type=str, default=None,
                   help='Output directory (default: {ckpt_base}/{run_name}/infer_100k)')
    return p.parse_args()

args = _parse()

# Set GPU visibility before TF loads CUDA
_gpu_id = args.gpu_id if args.gpu_id is not None else args.rank
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_id)
os.environ['TF_GPU_ALLOCATOR']     = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_FLAGS']            = (
    '--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2')

import ctypes as _ctypes
for _lib in [
    '/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64/libcudart.so.12',
    '/global/common/software/nersc9/cudnn/8.9.3-cuda12/lib/libcudnn.so.8',
]:
    try: _ctypes.CDLL(_lib, mode=_ctypes.RTLD_GLOBAL)
    except OSError: pass

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.random.set_seed(42 + args.rank)

print(f'[rank {args.rank}/{args.world_size}] CUDA_VISIBLE_DEVICES={_gpu_id} '
      f'  visible TF GPUs: {len(gpus)}')

# ── Model path ────────────────────────────────────────────────────────────────
CKPT_BASE  = args.ckpt_dir  or f'{args.data_dir}/checkpoints_pet_pp'
STATS_DIR  = args.stats_dir or args.data_dir
CKPT_PATH  = f'{CKPT_BASE}/{args.run_name}/pet_pp.weights.h5'
OUT_DIR    = args.out_dir or f'{CKPT_BASE}/{args.run_name}/infer_100k'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data slice for this rank ──────────────────────────────────────────────────
# Distribute n_total events contiguously: rank r gets [my_start, my_end)
n_per_rank = args.n_total // args.world_size
remainder  = args.n_total % args.world_size
my_n       = n_per_rank + (1 if args.rank < remainder else 0)
my_start   = args.val_start + args.rank * n_per_rank + min(args.rank, remainder)
my_end     = my_start + my_n

print(f'[rank {args.rank}] event range [{my_start}, {my_end}) = {my_n} events')

# ── Load normalisation stats ──────────────────────────────────────────────────
import h5py
with open(f'{STATS_DIR}/normalisation_stats.json') as f:
    stats = json.load(f)

npart_stats_path = f'{CKPT_BASE}/{args.run_name}/npart_stats.json'
with open(npart_stats_path) as f:
    npart_stats = json.load(f)
npart_mean = float(npart_stats['npart_mean'])
npart_std  = float(npart_stats['npart_std'])

jet_mean  = float(stats['jet_mean'][0])
jet_std   = float(stats['jet_std'][0])
cond_mean = np.array(stats['cond_mean'], dtype=np.float32)
cond_std  = np.where(np.array(stats['cond_std'], dtype=np.float32) > 0,
                     np.array(stats['cond_std'], dtype=np.float32), 1.0)
part_mean = np.array(stats['part_mean'], dtype=np.float32)
part_std  = np.array(stats['part_std'],  dtype=np.float32)

# ── Load HDF5 slices ──────────────────────────────────────────────────────────
per_proc = {}
for proc in ['dijet', 'zjets']:
    path = f'{args.data_dir}/{proc}.hdf5'
    with h5py.File(path, 'r') as f:
        total = f['particle_features'].shape[0]
        s = min(my_start, total)
        e = min(my_end,   total)
    if s >= e:
        print(f'[rank {args.rank}] {proc}: no events in [{s},{e}), skipping')
        per_proc[proc] = None
        continue
    with h5py.File(path, 'r') as f:
        pf   = f['particle_features'][s:e].astype(np.float32)
        part = f['parton_features'][s:e].astype(np.float32)
    mask  = pf[:, :, 6]
    X_raw = pf[:, :, :6]
    cond_parton = (part.reshape(len(pf), 24) - cond_mean) / cond_std
    npart_truth = mask.sum(axis=1)
    npart_norm  = ((npart_truth - npart_mean) / npart_std)[:, None].astype(np.float32)
    cond = np.concatenate([cond_parton, npart_norm], axis=1)
    per_proc[proc] = {
        'X_raw':       X_raw,
        'mask':        mask,
        'y':           cond,
        'parton_feat': part.reshape(len(pf), 4, 6),
    }
    print(f'[rank {args.rank}] {proc}: loaded {len(pf)} events '
          f'(mean npart={mask.sum(axis=1).mean():.1f})')

# ── Load model ────────────────────────────────────────────────────────────────
_scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts)
from PET_pp_parton import PET_pp_parton

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f'Checkpoint not found: {CKPT_PATH}')

model = PET_pp_parton(
    num_feat=6, num_jet=1, num_cond=25,
    num_partons=4, parton_feat=6, num_part=args.npart,
    projection_dim=args.proj_dim, local=True, K=5,
    num_layers=args.num_layers, drop_probability=0.0,
    simple=False, layer_scale=True, talking_head=False,
    mode='generator',
)
model.load_weights(CKPT_PATH)
print(f'[rank {args.rank}] Loaded {CKPT_PATH}')

# ── Inference ─────────────────────────────────────────────────────────────────
t0_total = time.perf_counter()

for proc in ['dijet', 'zjets']:
    if per_proc[proc] is None:
        continue
    d    = per_proc[proc]
    cond = d['y']
    N    = len(cond)

    # Ensure all chunks are the same size for a single tf.function trace
    nsplit = max(1, N // args.chunk_size)
    actual_chunk = N // nsplit
    print(f'[rank {args.rank}] {proc}: {N} events, '
          f'nsplit={nsplit} ({actual_chunk} events/chunk), '
          f'num_steps={args.num_steps}')

    t1 = time.perf_counter()
    parts_gen, jets_gen = model.generate(
        cond=cond,
        jet_mean=jet_mean,
        jet_std=jet_std,
        nsplit=nsplit,
        num_steps=args.num_steps,
        use_tqdm=True,
    )
    dt = time.perf_counter() - t1
    print(f'[rank {args.rank}] {proc}: {dt/60:.2f} min  ({dt/N*1000:.0f} ms/event)')

    # Reconstruct mask and denormalise
    log_npart_gen = jets_gen[:, 0] * jet_std + jet_mean
    npart_gen = np.clip(np.round(np.exp(log_npart_gen)).astype(int), 1, args.npart)
    mask_gen  = (np.arange(args.npart)[None, :] < npart_gen[:, None]).astype(np.float32)
    parts_phys = (parts_gen * part_std + part_mean) * mask_gen[:, :, None]
    parts_phys[:, :, 5] = np.round(parts_phys[:, :, 5])  # charge must be integer

    out_file = f'{OUT_DIR}/{proc}_rank{args.rank:02d}_of{args.world_size:02d}.npz'
    np.savez_compressed(out_file,
        parts_truth  = d['X_raw'],
        parts_gen    = parts_phys,
        mask         = d['mask'],
        mask_gen     = mask_gen,
        parton_feat  = d['parton_feat'],
    )
    print(f'[rank {args.rank}] {proc}: saved -> {out_file}')

print(f'[rank {args.rank}] All done in {(time.perf_counter()-t0_total)/60:.2f} min')
