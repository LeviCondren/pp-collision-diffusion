# pp-collision-diffusion — Claude Code context

This repository tracks scripts and experiments for a **parton-conditioned diffusion model** for full-event generation at the LHC. Architecture: PET_pp_parton — a Point Edge Transformer with per-parton cross-attention. Trained on matched (MG5 parton-level, Pythia hadron-level) event pairs to learn P(final state | hard scatter).

---

## Repository layout

```
scripts/          Python training, inference, and architecture files
submit/           Slurm batch submit scripts (self-resubmitting)
bsm_pipeline/     BSM surrogate pipeline package
EXPERIMENTS.md    Source of truth for what's running and what each result means
CHANGELOG.md      File-level change log with SHA-256 hashes
```

The **working copy** of all scripts is at:
```
/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts/
```
This repo mirrors those scripts. When you clone this repo and run scripts, use the absolute paths below — all submit scripts already hardcode them.

---

## Cluster / environment

- **Cluster:** NERSC Perlmutter (`perlmutter.nersc.gov`), username `lcondren`
- **Python env:** `module load tensorflow/2.15.0` (TF 2.15, Horovod)
- **PYTHONPATH:** `/global/u2/l/lcondren/ContinuousParamFit/omnilearn_pp/scripts`
- **Slurm account:** `m2616` — **always use `--account=m2616`**
- **GPU queue:** `--constraint=gpu --qos=regular`

---

## Data paths

| Resource | Path |
|----------|------|
| W' signal grid data | `/pscratch/sd/l/lcondren/MCsim/wprime_signal/` |
| Checkpoint root | `/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid/` |
| Slurm logs | `/pscratch/sd/l/lcondren/MCsim/wprime_signal/logs/` |

Normalisation stats are stored inside each run's checkpoint directory, **not** in the grid data directory. This is critical — do not write stats to the grid_dir.

---

## Active training jobs (as of 2026-07-01)

Read `EXPERIMENTS.md` for full status. The two jobs that should be running or queued:

### E022 — 4-layer generator head (bsm_grid_event_c_layers4)
- **What:** E020c variant with `num_gen_layers=4` (up from 2). Tests whether a deeper generator head improves particle-level fidelity.
- **Scripts:**
  - Architecture: `scripts/PET_pp_parton_vpar_bsm_event_c_layers4.py`
  - Training: `scripts/bsm_grid_train_event_c_layers4.py`
  - Submit: `submit/submit_e022_bsm_grid_event.sh`
- **Checkpoint dir:** `.../checkpoints_bsm_grid/bsm_grid_event_c_layers4/`
- **State file:** `.../checkpoints_bsm_grid/bsm_grid_event_c_layers4/training_state.json`
- **Last known Slurm job:** 55352668 (self-resubmitting; each job runs ~3.5 h)
- **Target:** 200 epochs total, patience=30, val_loss should track ~4.9

### E023 — 8-dim stage-1 diffusion (bsm_grid_event_c_stage1)
- **What:** Expands stage-1 ResNet from predicting 1-dim log_npart to predicting 8-dim [log_npart, log1p(MET), sin(MET_phi), cos(MET_phi), log1p(cone_pT_X), log1p(cone_mass_X), log1p(cone_pT_Y), log1p(cone_mass_Y)] as a v-parameterized diffusion model. Stage-2 (particle generator) is architecturally unchanged. ResNet width 512 (FPCD default).
- **Scripts:**
  - Architecture: `scripts/PET_pp_parton_vpar_bsm_event_c_stage1.py`
  - Training: `scripts/bsm_grid_train_event_c_stage1.py`
  - Submit: `submit/submit_e023_bsm_grid_event_stage1.sh`
- **Checkpoint dir:** `.../checkpoints_bsm_grid/bsm_grid_event_c_stage1/`
- **State file:** `.../checkpoints_bsm_grid/bsm_grid_event_c_stage1/training_state.json`
- **Stats file:** `.../checkpoints_bsm_grid/bsm_grid_event_c_stage1/normalisation_stats_event_c_stage1.json` (8-dim; auto-computed on first run)
- **Last known Slurm job:** 55355425 (self-resubmitting)
- **Target:** 200 epochs total, patience=30, val_loss should track ~5 (similar to E020c early training)

---

## How to check job status

```bash
# Check whether jobs are still queued/running
squeue -u lcondren

# Check training progress for E022
python3 -c "import json; s=json.load(open('/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid/bsm_grid_event_c_layers4/training_state.json')); print(s)"

# Check training progress for E023
python3 -c "import json; s=json.load(open('/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid/bsm_grid_event_c_stage1/training_state.json')); print(s)"
```

---

## How to resubmit a stopped job

The submit scripts are self-resubmitting — they schedule their own continuation at the start of each run. If a job stopped (queue issue, node failure, etc.) and `done=false` in `training_state.json`, resubmit manually:

```bash
# Resubmit E022
sbatch /global/u2/l/lcondren/pp-collision-diffusion/submit/submit_e022_bsm_grid_event.sh

# Resubmit E023
sbatch /global/u2/l/lcondren/pp-collision-diffusion/submit/submit_e023_bsm_grid_event_stage1.sh
```

Both scripts check `training_state.json` at startup and exit cleanly if `done=true`, so it is safe to resubmit even if the job is already complete.

---

## Coding conventions

- TensorFlow / Keras + Horovod. Do not propose PyTorch.
- Minimal targeted edits over rewrites.
- No try/except blocks to suppress errors. Understand and fix the root cause.
- Do not modify `PET.py` or `layers.py` (upstream OmniLearn framework files).
- Preserve argparse structure so smoke-test invocations continue to work.
- Do not invent physics parameters, PDG IDs, or argument values not already in the scripts.
- Stats files belong in `ckpt_dir`, **never** in `grid_dir`.

---

## Experiment ledger protocol

`EXPERIMENTS.md` is append-only for completed work. Before submitting any job, add a STAGED row. After submission, update to RUNNING with the Slurm job ID. When done, move to "Recently completed" and add a 2-3 sentence interpretation.

Do not edit `EXPERIMENTS.md` from multiple parallel Claude Code sessions simultaneously.

---

## Known pitfalls

- **Stats contamination:** If you run a smoke test pointing `--grid_dir` at the production data directory but `--ckpt_dir` at a temp path, and the script defaults stats to `grid_dir`, the smoke test will overwrite production stats with tiny-data values. Always verify stats default to `ckpt_dir`. E023 was fixed for this; earlier variants (E020a/b/c) use a separate `--stats_path` flag.
- **Checkpoint path mismatch:** Inference scripts default to the checkpoint path from training. If you change `--ckpt_dir` or `--run_name`, pass the matching path explicitly.
- **Normalization stats reuse:** Do not copy stats from one dataset to another. Stats must be recomputed whenever the dataset or feature engineering changes.
- **`--num_part` must match data:** Smoke tests use smaller `--num_part` than production (50 vs 500). Stats computed with the wrong `--num_part` produce wrong log_npart variance.
