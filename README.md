# pp-collision-diffusion

Parton-conditioned diffusion model for full LHC pp event generation. Given a set of hard-scatter partons from a MadGraph5 event, the model learns P(final state | hard scatter) and can generate complete Pythia-showered final states at a fraction of the cost.

Architecture: **PET-pp-parton** — a Point Edge Transformer with per-parton cross-attention in the generator head, plus a ResNet jet head that predicts particle multiplicity (log N_part). Training uses matched (MG5 parton-level, Pythia hadron-level) event pairs for 5 SM processes simultaneously: dijet, Z+jets, ttbar, W+jets, and W' signal.

---

## Repository layout

```
scripts/           Model architecture, training, inference, and plotting
data_generation/   MG5+Pythia8 event generation scripts (produces the HDF5 training data)
bsm_pipeline/      MG5/Pythia8 interface package (dependency of data_generation/)
submit/            NERSC Perlmutter Slurm submit scripts
EXPERIMENTS.md     Experiment ledger — source of truth for all runs
CHANGELOG.md       Record of all code changes
```

---

## Dependencies

- TensorFlow 2.15 with Keras
- Horovod (distributed training)
- h5py, numpy
- pythia8 (Python bindings, for data generation)
- MadGraph5 (for data generation; path set via `MG5_BIN` in `bsm_pipeline/mg5_runner.py`)
- matplotlib, scipy, pyjet (for plotting)

On NERSC Perlmutter: `module load tensorflow/2.15.0`

---

## Pipeline overview

```
data_generation/          →      scripts/           →      scripts/
generate_full_events.py          proc_label_train.py        infer_pp_proc_label.py
(MG5+Pythia → HDF5)              (trains the model)         (generates events)
                                                            ↓
                                                     scripts/plot_infer_5proc.py
```

---

## Step 1 — Generate training data

Three scripts cover the five processes:

**dijet + Z+jets** (MG5+Pythia8 matched pairs):
```bash
python3 data_generation/generate_full_events.py \
  --out_dir /path/to/output \
  --process dijet        # or zjets
  --n_events 500000
```

**ttbar + W+jets** (LO MG5+Pythia8):
```bash
python3 data_generation/generate_full_events_lo_ttbar_wjets.py \
  --out_dir /path/to/output \
  --process ttbar        # or wjets
  --n_events 500000
```

**W' signal** (Pythia8-only parametric):
```bash
python3 data_generation/generate_wprime_signal.py \
  --out_dir /path/to/output \
  --mX 500 --mY 100      # W' → X(mX) + Y(mY)
  --n_events 500000
```

Each script produces an HDF5 file with fields:
- `particle_features`: (N, num_part, 7) — η, sin φ, cos φ, log pT, pid, charge, mask
- `parton_features`: (N, 4, 6) — same 6 features per hard-scatter parton
- `event_weights`: (N,) — per-event MC weight

After generating, compute normalization statistics once per dataset:
```bash
python3 scripts/compute_norm_stats.py --data_dir /path/to/output
```
This writes `normalisation_stats.json` to the data directory. Do not reuse stats across datasets generated with different settings.

---

## Step 2 — Train

**Baseline / CFG training** (E001):
```bash
# Interactive single-GPU (for testing):
export PYTHONPATH=/path/to/repo/scripts
OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_SIZE=1 OMPI_COMM_WORLD_LOCAL_RANK=0 \
python3 scripts/proc_label_train.py \
  --data_dir /path/to/data \
  --run_name my_run \
  --processes dijet zjets ttbar wjets wprime \
  --epoch 200 --batch 128 --lr 3e-4 \
  --num_layers 8 --num_gen_layers 2 --proj_dim 128

# Multi-GPU via Horovod (4 GPUs):
export PYTHONPATH=/path/to/repo/scripts
horovodrun -np 4 python3 scripts/proc_label_train.py [same args]
```

Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Root directory containing per-process HDF5 files |
| `--run_name` | required | Checkpoint subdirectory name |
| `--processes` | — | Space-separated list: `dijet zjets ttbar wjets wprime` |
| `--val_start` | 400000 | Row index where validation data begins |
| `--cfg_drop_prob` | 0.0 | Process-label dropout probability for CFG training |
| `--epoch` | 200 | Total epochs |
| `--patience` | 30 | Early stopping patience |
| `--time_limit_hours` | — | Soft wall-clock limit (for self-resubmitting Slurm jobs) |

Checkpoints are written to `{data_dir}/checkpoints/{run_name}/pet_pp.weights.h5`.

**Auxiliary classification variant** (E007):
```bash
python3 scripts/proc_label_train_auxcls_body.py \
  --data_dir /path/to/data \
  --run_name my_run_auxcls \
  --processes dijet zjets ttbar wjets wprime \
  --aux_weight 0.1 \
  [... same other args]
```
This variant adds a cross-entropy classification loss on the body's mean-pooled output, encouraging the body encoder to learn process-discriminative features.

**NERSC Slurm submission:**
```bash
sbatch submit/submit_cfg_5proc_dropout10.sh     # E001: CFG training
sbatch submit/submit_auxcls_body_5proc.sh        # E007: aux-cls body training
```
Submit scripts self-resubmit via `--dependency=afterany` until `training_state.json` marks `done=true`.

---

## Step 3 — Inference

```bash
export PYTHONPATH=/path/to/repo/scripts
OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_SIZE=1 OMPI_COMM_WORLD_LOCAL_RANK=0 \
python3 scripts/infer_pp_proc_label.py \
  --data_dir /path/to/data \
  --ckpt_dir /path/to/data/checkpoints \
  --run_name my_run \
  --processes dijet zjets ttbar wjets wprime \
  --n_total 20000 \
  --num_steps 50 \
  --guidance_scale 0.0    # set > 0 only if trained with cfg_drop_prob > 0
```

Output: one `.npz` file per process in `{ckpt_dir}/{run_name}/infer_20k/`.

**Diagnostic: compare sampled vs true multiplicity (E002):**
```bash
# Sampled stage-1 log_npart (default):
python3 scripts/infer_pp_5proc_truelogn_comparison.py \
  --data_dir /path/to/data --ckpt_dir /path/to/checkpoints --run_name my_run \
  --n_total 20000 --num_steps 50

# True log_npart supplied from validation data:
python3 scripts/infer_pp_5proc_truelogn_comparison.py \
  --data_dir /path/to/data --ckpt_dir /path/to/checkpoints --run_name my_run \
  --use_true_jet --n_total 20000 --num_steps 50
```
Comparing the two modes isolates stage-1 (multiplicity prediction) errors from stage-2 (particle generation) errors.

---

## Step 4 — Plot

```bash
python3 scripts/plot_infer_5proc.py \
  --infer_dir /path/to/checkpoints/my_run/infer_20k \
  --data_dir /path/to/data \
  --out_dir figures/my_run
```

Produces per-process kinematic distribution plots (η, φ, log pT, multiplicity, jet observables) comparing generated events to validation data.

---

## Architecture

The model has two stages:

**Stage 1 — Jet head (`model_jet`):** A ResNet that takes parton features and predicts log N_part (particle multiplicity). Trained jointly with stage 2.

**Stage 2 — Particle head (`model_part`):** A denoising diffusion model over the particle cloud. The body is a PET (Point Edge Transformer) that processes the noisy particle cloud. The generator head applies per-parton cross-attention: 4 parton tokens (optionally + 1 process-label token) form the key/value set, and particle-level queries attend over them at each generator layer.

Key files:
- [scripts/PET_pp_parton_vpar.py](scripts/PET_pp_parton_vpar.py) — canonical architecture (`ProcLabelPET`)
- [scripts/PET_pp_parton_vpar_auxcls_body.py](scripts/PET_pp_parton_vpar_auxcls_body.py) — E007 variant (same arch class, paired with `proc_label_train_auxcls_body.py`)
- [scripts/PET.py](scripts/PET.py) — upstream PET base class (OmniLearn framework)
- [scripts/layers.py](scripts/layers.py) — custom Keras layers (LayerScale, StochasticDepth, RandomDrop, TalkingHeadAttention)

---

## Active models (as of 2026-07-21)

Four checkpoints are published at **[Levicondren/pet-pp-checkpoints](https://huggingface.co/Levicondren/pet-pp-checkpoints)** (private). Download with:

```bash
pip install huggingface_hub
export HF_TOKEN=<your_token>
huggingface-cli download Levicondren/pet-pp-checkpoints \
    --repo-type model --local-dir ./pet-pp-checkpoints --token $HF_TOKEN
```

Each HF directory contains `pet_pp.weights.h5`, `training_state.json`, and the relevant normalisation stats JSON(s). Place the weights file at the `ckpt_dir` path expected by the training script (see below), and put the stats JSON(s) at the paths listed.

---

### E029 — SM stage-2 (layers4), epoch 178/200 — **complete**

- **Architecture:** `PET_pp_parton_vpar_bsm_event_c_layers4` (`num_gen_layers=4`)
- **Training script:** `scripts/sm_4proc_train_event_c_layers4.py`
- **Submit script:** `submit/submit_e029_sm_4proc_infer.sh` (reference for args)
- **HF directory:** `e029_sm_stage2_ep178/`
- **Checkpoint path:** `<data_dir>/checkpoints_sm_4proc/sm_4proc_event_c_layers4_full/pet_pp.weights.h5`
- **Stats files:**
  - `normalisation_stats_sm4proc.json` → `<data_dir>/normalisation_stats_sm4proc.json`
  - `normalisation_stats_event_c_sm4proc.json` → `<data_dir>/normalisation_stats_event_c_sm4proc.json`
- **Data:** 4 SM processes (dijet, ttbar, wjets, zjets); train [0:480k], val [480k:490k], holdout [490k:500k]
- **Resume command:**
```bash
export PYTHONPATH=/path/to/repo/scripts
horovodrun --gloo -np 1 python3 scripts/sm_4proc_train_event_c_layers4.py \
    --data_dir <data_dir> \
    --run_name sm_4proc_event_c_layers4_full \
    --val_start 480000 \
    --epoch 200 --batch 128 --lr 3e-4 --lr_body 1e-4 \
    --num_layers 8 --num_gen_layers 4 --proj_dim 128
```

---

### E030 — SM stage-1, epoch 153/200 — **complete**

- **Architecture:** `PET_pp_parton_vpar_bsm_event_c_stage1` (`num_gen_layers=2`, `num_jet_mlp=512`, `num_jet=8`)
- **Training script:** `scripts/sm_5proc_train_event_c_stage1.py`
- **Submit script:** `submit/submit_e031_bsm_grid_mpi_layers4.sh` (reference for args structure)
- **HF directory:** `e030_sm_stage1_ep153/`
- **Checkpoint path:** `<data_dir>/checkpoints_sm_5proc/sm_5proc_event_c_stage1/pet_pp.weights.h5`
- **Stats files:**
  - `normalisation_stats_sm5proc_stage1.json` → `<data_dir>/checkpoints_sm_5proc/normalisation_stats_sm5proc_stage1.json`
- **Data:** 5 SM processes (dijet, ttbar, wjets, zjets, wprime); train [0:480k], val [480k:490k]
- **Resume command:**
```bash
export PYTHONPATH=/path/to/repo/scripts
horovodrun --gloo -np 1 python3 scripts/sm_5proc_train_event_c_stage1.py \
    --data_dir <data_dir> \
    --run_name sm_5proc_event_c_stage1 \
    --val_start 480000 \
    --epoch 200 --batch 128 --lr 3e-4 --lr_body 1e-4 \
    --num_layers 8 --num_gen_layers 2 --proj_dim 128 --num_jet_mlp 512
```

---

### E031 — W' stage-2 (layers4), mid-training — **in progress**

- **Architecture:** `PET_pp_parton_vpar_bsm_event_c_layers4` (`num_gen_layers=4`)
- **Training script:** `scripts/bsm_grid_train_event_c_layers4.py`
- **Submit script:** `submit/submit_e031_bsm_grid_mpi_layers4.sh`
- **HF directory:** `e031_wprime_stage2_ep<N>/`
- **Checkpoint path:** `<grid_dir>/checkpoints_bsm_grid/bsm_grid_event_c_layers4_mpi/pet_pp.weights.h5`
- **Stats files:**
  - `normalisation_stats.json` → `<grid_dir>/normalisation_stats.json`
  - `normalisation_stats_event_c.json` → `<grid_dir>/normalisation_stats_event_c.json`
- **Data:** 144-point W' → WZ → 4q mass grid + background HDF5 (MPI=on); val_start=80000, n_train=20000/file. Holdout: (250,250) (250,300) (300,250) (300,300).
- **Resume command:**
```bash
export PYTHONPATH=/path/to/repo/scripts
horovodrun --gloo -np 1 python3 scripts/bsm_grid_train_event_c_layers4.py \
    --grid_dir <grid_dir> \
    --ckpt_dir <grid_dir>/checkpoints_bsm_grid \
    --run_name bsm_grid_event_c_layers4_mpi \
    --val_start 80000 --n_train 20000 \
    --epoch 200 --batch 128 --lr 3e-4 --lr_body 1e-4 \
    --num_layers 8 --num_gen_layers 4 --proj_dim 128
```

---

### E032 — W' stage-1, mid-training — **in progress**

- **Architecture:** `PET_pp_parton_vpar_bsm_event_c_stage1` (`num_gen_layers=2`, `num_jet_mlp=512`, `num_jet=8`)
- **Training script:** `scripts/bsm_grid_train_event_c_stage1.py`
- **Submit script:** `submit/submit_e032_bsm_grid_mpi_stage1.sh`
- **HF directory:** `e032_wprime_stage1_ep<N>/`
- **Checkpoint path:** `<grid_dir>/checkpoints_bsm_grid/bsm_grid_event_c_stage1_mpi/pet_pp.weights.h5`
- **Stats files:**
  - `normalisation_stats_event_c.json` → `<grid_dir>/normalisation_stats_event_c.json`
- **Data:** Same 144-point W' grid as E031 (MPI=on).
- **Resume command:**
```bash
export PYTHONPATH=/path/to/repo/scripts
horovodrun --gloo -np 1 python3 scripts/bsm_grid_train_event_c_stage1.py \
    --grid_dir <grid_dir> \
    --ckpt_dir <grid_dir>/checkpoints_bsm_grid \
    --run_name bsm_grid_event_c_stage1_mpi \
    --val_start 80000 --n_train 20000 \
    --epoch 200 --batch 128 --lr 3e-4 --lr_body 1e-4 \
    --num_layers 8 --num_gen_layers 2 --proj_dim 128 --num_jet_mlp 512
```

---

## Experiments

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full ledger of all runs, results, and interpretations.

---

## Data paths (NERSC Perlmutter)

- SM training data: `/pscratch/sd/l/lcondren/MCsim/full_event_mixed/`
- SM checkpoints: `/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints_sm_4proc/` and `checkpoints_sm_5proc/`
- W' signal data (MPI=on): `/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/`
- W' checkpoints: `/pscratch/sd/l/lcondren/MCsim/wprime_signal_mpi/checkpoints_bsm_grid/`
