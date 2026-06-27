# Experiment Ledger

Source of truth for what's running, what's done, and what each result means.

Last updated: 2026-06-25 (E022 submitted — Slurm job 55094220; num_gen_layers=4 variant of E020c)

---

## Active experiments (running, queued, or staged)

| ID | Status | Submitted | Type | Run name | Slurm job | Notes |
|----|--------|-----------|------|----------|-----------|-------|
| E008 | RUNNING | 2026-06-14 | training | `bsm_grid` | 54707121 | Epoch 55/200, val_loss=4.900; continuation job (prior jobs: 54455691 and chain) |
| A007 | RUNNING | 2026-06-19 | diagnostic (inference) | `bsm_grid → infer_holdout_ep055_5k` | 54716471 | Mid-training holdout inference at epoch ~55; 4 holdout pts (250,250)(250,300)(300,250)(300,300), 5k events, 500 steps; submit: `submit_e008_holdout_infer_ep055.sh` |
| A008 | RUNNING | 2026-06-19 | diagnostic (inference) | `bsm_grid → infer_trained_ep019_5k` | 54677288 | Trained-point inference for mass-overlay comparison; 4 trained pts (200,200)(200,350)(350,200)(350,350), used with `plot_e008_mass_overlay.py` to confirm model fans out with mass |
| E016 | PLANNED | — | validation | `parnassus_validation` | — | Test Parnassus output on W'→4q signal; compare to dijet training-domain behavior |
| E020a | RUNNING | 2026-06-19 | training | `bsm_grid_event_a` | 54738498 | Event-level MET conditioning (3 feat); epoch 49/200 val_loss=4.899; self-resubmitting |
| E020b | RUNNING | 2026-06-19 | training | `bsm_grid_event_b` | 54738499 | Event-level cone_X conditioning (2 feat); epoch 51/200 val_loss=4.899; self-resubmitting |
| E020c | RUNNING | 2026-06-19 | training | `bsm_grid_event_c` | 54738501 | Event-level all-7 event features; epoch 56/200 val_loss=4.896; self-resubmitting |
| E022  | RUNNING | 2026-06-25 | training | `bsm_grid_event_c_layers4` | 55094220 | E020c + num_gen_layers=4; self-resubmitting; submit: `submit_e022_bsm_grid_event.sh` |
| A009a-t | RUNNING | 2026-06-25 | inference (holdout, truth-cond) | `bsm_grid_event_a/infer_holdout_truth` | 55037853 | E020a truth-conditioned holdout; 4 pts × 5k events × 500 steps; 4 GPUs parallel |
| A009b-t | RUNNING | 2026-06-25 | inference (holdout, truth-cond) | `bsm_grid_event_b/infer_holdout_truth` | 55037857 | E020b truth-conditioned holdout; 4 pts × 5k events × 500 steps; 4 GPUs parallel |
| A009c-t | RUNNING | 2026-06-25 | inference (holdout, truth-cond) | `bsm_grid_event_c/infer_holdout_truth` | 55037860 | E020c truth-conditioned holdout; 4 pts × 5k events × 500 steps; 4 GPUs parallel |

---

## Deferred experiments (ready to re-submit)

| ID | Status | Submitted | Type | Run name | Slurm job | Notes |
|----|--------|-----------|------|----------|-----------|-------|
| A004 | DEFERRED | 2026-06-13 | diagnostic (inference) | `proc_label_5proc_p3 → infer_truejet_steps500_ext` | 54408659 (cancelled) | 500-step baseline extension; 2k/proc (10k total); val_start=402000; submit script: `submit_a004_steps500_ext.sh` |
| A005 | DEFERRED | 2026-06-13 | diagnostic (inference) | `cfg_5proc_dropout10 → infer_e001_partial_steps500` | 54408661 (cancelled) | E001 partial ckpt (61/200 ep); 1k/proc (5k total); 500 steps; gs=0.0; submit script: `submit_a005_e001_partial.sh` |
| A006 | DEFERRED | 2026-06-13 | diagnostic (inference) | `auxcls_body_5proc → infer_e007_partial_steps500` | 54408662 (cancelled) | E007 partial ckpt (19/200 ep); 1k/proc (5k total); 500 steps; submit script: `submit_a006_e007_partial.sh` |

---

## Recently completed experiments

| ID | Completed | Type | Run name | Key result | Notes |
|----|-----------|------|----------|------------|-------|
| E015 | 2026-06-15 | infrastructure | `parnassus_integration_setup` | Wrapper built at `omnilearn_pp/scripts/parnassus_wrapper.py`; smoke test 100 events passed; mean mult 190→190, pT 17.6→8.8 GeV, no NaN/Inf | Full-event model (fm_full_event-epoch=034); run in pipeline_copy-gpu2 env |
| E007 | 2026-06-13 | training (CANCELLED) | `auxcls_body_5proc` | 19/200 epochs; val_loss=5.447 at cancellation | Cancelled to run partial-ckpt inference (A006); body auxcls loss was decreasing normally |
| E001 | 2026-06-13 | training (CANCELLED) | `cfg_5proc_dropout10` | 61/200 epochs; val_loss=5.421 at cancellation | Cancelled to run partial-ckpt inference (A005); CFG training proceeding normally |
| A002 | 2026-06-13 | diagnostic (inference) | `proc_label_5proc_p3 → infer_truejet_steps{100,200,500}` | η W1 drops ~60x (0.56→0.009) from 50→200 steps; 200 steps is convergence point | 50-step inference was severely undersampled — over-dispersion was a sampler artifact |
| E003 | 2026-06-11 | plotting/analysis | `E002a vs E002b` | Stage-1 errors limited to multiplicity; η/pT mismatches are stage-2 intrinsic | Figures at `.../proc_label_5proc_p3/figures/E003_truejet_comparison/` |
| E002b | 2026-06-11 | inference (diagnostic) | `proc_label_5proc_p3 → infer_20k_truejet` | 20k events/process; npart exact_match=1.000 all processes | E000 ckpt; true log_npart bypass confirmed working |
| E002a | 2026-06-11 | inference (diagnostic) | `proc_label_5proc_p3 → infer_20k_sampled` | 20k events/process; W' npart bias +14, SM processes within ~5 | E000 ckpt; stage-1 over-disperses multiplicity (std too wide) |
| E006 | 2026-06-11 | training | `auxcls_5proc` | Cancelled at epoch 20; aux loss collapsed to ~1e-7 (trivial identity task) | Design flaw: classifier on proc_token = identity mapping; ran 20/200 epochs |
| E000 | 2026-06-08 | training | `proc_label_5proc_p3` | val_loss plateau ~5.418 at epoch 91 | Phase 3 xattn proc_token; baseline for all subsequent work |
| A001 | 2026-06-09 | ablation | `ablation_proc_paths` | Resnet path C dominates; xattn path A modest; additive path B negligible | Post-load weight zeroing on E000 checkpoint |

---

## Experiment details

(Most recent first.)

---

### E015 — Parnassus integration (parnassus_integration_setup)

- **Date staged:** 2026-06-15
- **Goal:** Chain learned detector simulation/reconstruction (Parnassus) after the BSM diffusion generator to bridge from hadron-level output to detector-level reconstructed objects suitable for PAWS analysis. Pipeline: diffusion (infer_bsm_grid.py) → Parnassus → downstream PAWS.
- **Phase A findings (complete):**
  - **Code already on machine.** Two versions present:
    - Jet Parnassus (arXiv:2406.01620): `/pscratch/sd/l/lcondren/MCsim/Parnassus/` — CMS jet model, max_particles=201, pretrained as `fm_cms_J800_1000_epoch=49.ckpt` (jets 800–1000 GeV). NOT suitable for full events.
    - Full-event model (custom, same architecture): checkpoint `fm_full_event-epoch=034-val_loss=1.0545.ckpt` at `/pscratch/sd/l/lcondren/MCsim/full_event_detector_data/checkpoints/`. Trained on pp→jj (MG5 dijet) + Pythia + Delphes, 50k events, max_particles=600. This is the one to use.
    - The Dreyer et al. 2025 full-event Parnassus paper (arXiv:2503.19981) code/weights are NOT present — the user built an equivalent independently.
  - **Infrastructure already built.** A complete `bsm_pipeline` package exists at `bsm_pipeline/bsm_pipeline/detector.py` with `ParnassusBackend` class that handles: particle-event → numpy conversion, subprocess invocation of `infer_particles.py` in `pipeline_copy-gpu2` env, output → `RecoEvent` conversion. Previously used for the FPCD → Parnassus dark-photon demo.
  - **Conda environment:** `pipeline_copy-gpu2` (torch 2.8.0+cu128, pytorch_lightning 1.9.2, torchcfm). Parnassus must run in this env; our diffusion code runs in `tensorflow/2.15.0` module. These are separate environments — the wrapper must handle the subprocess boundary.
  - **Input format (Parnassus full-event):** `(N, 600, 4)` float32 = [log_pT_normed, eta_normed, phi_normed, charge_class_binary]; mask `(N, 600, 2)` bool (col 0=truth, col 1=pflow); scale `(N, 6)` per-event normalization stats. Per-event normalization uses log(pT_GeV) mean/std.
  - **Output format:** `(N, 600, 3)` float32 = [pT_GeV, eta, phi] physical; mask `(N, 600)` bool.
  - **Compatibility gaps vs. infer_bsm_grid.py output:**
    1. Feature space: our output is 6-feature normalized (η, sin_φ, cos_φ, log_pT, pdg_norm, charge) — must denormalize, recover φ = atan2(sin_φ, cos_φ), compute pT_GeV, map charge_float → charge_class_binary.
    2. Charge class: all truth particles from our W'→4q diffusion output carry charge info; set class=1 if |charge|>0.5 else 0. This matches what `ParnassusBackend._events_to_numpy` already does.
    3. pdg_norm feature unused by Parnassus — drop it.
    4. No neutral particles in our current encoding? Our charge feature has ~50% zeros (neutral pions etc.) — this is fine, the binary conversion handles it.
  - **Physics applicability caveats:**
    1. Full-event model was trained only on QCD dijet events. W'→4q signal events have a different topology (4 hard quarks, different multiplicity/pT distribution). Detector response is approximately process-agnostic, but extrapolation uncertainty exists.
    2. The Dreyer et al. 2025 weights (trained on more diverse events or official CMS full-reco) would be preferable — not available on machine.
    3. Jet Parnassus (CMS 800–1000 GeV) is strictly inapplicable: it processes one jet cluster, not a full event.
- **Phase B result (complete 2026-06-15):**
  - Wrapper written at `omnilearn_pp/scripts/parnassus_wrapper.py`. Runs in `pipeline_copy-gpu2` conda env (documented at top of file).
  - Pipeline: loads NPZ → denormalize 6-feature diffusion output → per-event Parnassus normalization (log(pT_MeV) z-score, matching FullEventDataset exactly) → batch inference via FullEventFlowLightning.sample() → denormalize → HDF5.
  - Key design decision: pflow mask = truth mask (1:1 particle mapping), since infer_bsm_grid.py generates contiguous prefix masks (no sorting needed).
  - Smoke test: 100 events from holdout NPZ (mX=350, mY=350, `pipeline_copy-gpu2` Python, CUDA A100).
    - Input: mean multiplicity 190.5 ± 54.4, pT mean=17.6 GeV, eta mean=0.10, phi ≈ uniform
    - Output: mean multiplicity 190.5 ± 54.4 (same — pflow mask=truth mask), pT mean=8.8 GeV, eta mean=−0.08, phi ≈ uniform
    - No NaN, Inf, or negative pT; phi wrapped to (−π, π) ✓; HDF5 attrs {m_X=350, m_Y=350, event_count=100} ✓
    - Output: `/pscratch/sd/l/lcondren/MCsim/parnassus_output/infer_holdout/mX0350_mY0350/recoparticles.hdf5`
  - pT note: max reco pT = 2672.8 GeV (1 outlier in 19k valid particles). Input was from an older wprimeGrid checkpoint on OOD input for Parnassus (W'→4q vs its QCD-dijet training domain). Expected to improve on E008 checkpoint output; see E016 for validation.
- **Steps:**
  - Phase A: Investigation (complete)
  - Phase B: Build wrapper (complete)
  - Phase C (E016): Validate Parnassus output on E008 inference → compare multiplicity/pT distributions to QCD training domain

---

### E008 — Phase 2 BSM grid training (bsm_grid)

- **Date staged:** 2026-06-13
- **Goal:** Train the Phase 2 mass-parameterized diffusion model on the W'→XY→4q signal grid + QCD background. The model learns P(final state | hard-scatter partons, m_X, m_Y), enabling generation conditioned on arbitrary signal mass hypotheses.
- **Architecture:** `PET_pp_parton_vpar_bsm` — identical PET body + variable-parton cross-attention head as Phase 1, but with:
  - `max_partons=4` (2 incoming + X + Y)
  - `parton_feat=7` (6 kinematic + mass/600 as 7th feature)
  - `num_cond=32` (4×7 + 4 mask bits, no process label)
  - No proc_token, no guidance_scale
- **Data:**
  - Signal: 144 HDF5 files at `/pscratch/sd/l/lcondren/MCsim/wprime_signal/signal_mX*.hdf5` (12×12 mass grid, m_X/m_Y ∈ {50,100,150,200,250,300,350,400,450,500,550,600} GeV), ~100k events each
  - Background: `background.hdf5` (QCD, mass_x=mass_y=0, 100k events)
  - Mass conditioning: slot 2 gets m_X/600, slot 3 gets m_Y/600; beam slots (0,1) get 0
  - Stats: `/pscratch/sd/l/lcondren/MCsim/wprime_signal/normalisation_stats.json` (auto-computed on first run, cond_dims=28)
- **Scripts:**
  - Training: `omnilearn_pp/scripts/bsm_grid_train.py`
  - Inference: `omnilearn_pp/scripts/infer_bsm_grid.py`
  - Architecture: `omnilearn_pp/scripts/PET_pp_parton_vpar_bsm.py`
- **Checkpoint dir:** `/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid/bsm_grid/`
- **Status:** RUNNING — Slurm job 54455691, submitted 2026-06-14. Resuming from epoch 4 (val_loss=4.943 after interactive run). Two OOM bugs fixed: (1) list-accumulate-then-concatenate in load_bsm_shard → two-pass pre-allocation; (2) from_tensor_slices doubles memory → fixed with --n_train 20000 (705k events/rank vs 2.82M) + staggered del so train numpy freed before val dataset built. Self-resubmitting via afterany dependency until `training_state.json` marks `done=true`.
- **Holdout:** 4 points excluded from training — `(250,250), (250,300), (300,250), (300,300)` (2×2 block near grid centre). Grid is full 12×12 (all m_X, m_Y combinations; not triangular), so m_X < m_Y cases exist. All 4 holdout files confirmed present on disk. Evaluate post-training with `infer_bsm_grid.py --m_X 250 --m_Y 250` etc.
- **Stats note:** `/pscratch/sd/l/lcondren/MCsim/wprime_signal/normalisation_stats.json` was correctly computed from all 140 non-holdout training files during the first (OOM-killed) run; valid and in place.
- **Known caveats:**
  - PDG norm in X/Y parton slots (2/3) is 0 for all signal events due to a generation bug (gluon PDG encoded instead of ±24/23). Mass feature alone differentiates signal from background; gluon PDG mismatch is non-blocking for Phase 2 proof-of-concept.
  - SM data (pdg_norm /16) is incompatible with wprime_signal data (pdg_norm /10). Do NOT mix datasets without recomputing stats.
- **Next step:** When training completes, submit `omnilearn_pp/submit_e008_holdout_infer.sh` — runs all 4 holdout points at 500 steps/event, 10k events each, outputs 4 separate .npz files (one per mass point) so plotting code can treat each as a distinct process. Inference default is 500 steps.

---

### A008 — E008 trained-point inference (mass-overlay comparison)

- **Date submitted:** 2026-06-19
- **Slurm job:** 54677288
- **Goal:** Generate events at 4 mass points that were **in the training set** — (200,200), (200,350), (350,200), (350,350) — to pair with the 4 holdout points in `plot_e008_mass_overlay.py`. Together they answer: does the model produce visually distinct kinematic distributions for different mass hypotheses (confirming mass conditioning is active), or does it output the same distribution regardless of input mass?
- **Setup:**
  - Checkpoint: `/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid/bsm_grid/pet_pp.weights.h5` (epoch ~55 at submission)
  - 4 trained mass points; event count and step count per the submit script
  - Output: `/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid/bsm_grid/infer_trained_ep019_5k/`
  - Analysis script: `omnilearn_pp/scripts/plot_e008_mass_overlay.py` — overlays HT, MET, leading pT, top-4 pT across all 8 mass points; computes W1 vs mSum scatter
- **Status:** RUNNING — job 54677288 submitted 2026-06-19, currently PENDING.
- **Linked experiments:** A007 (holdout inference, complementary input to mass-overlay plot), E008 (training source).

---

### A007 — E008 epoch-55 holdout inference diagnostic

- **Date submitted:** 2026-06-19
- **Slurm job:** 54716471
- **Goal:** Evaluate holdout generation quality at epoch 55 (val_loss=4.900) vs the earlier epoch-19 diagnostic (A-series, ~val_loss=4.904). Training is still ongoing (55/200 epochs, job 54707121). This is a mid-training snapshot to check whether the cone-mass and global-observable distributions have improved.
- **Setup:**
  - Checkpoint: `/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid/bsm_grid/pet_pp.weights.h5` (epoch 55)
  - Holdout points: (250,250), (250,300), (300,250), (300,300)
  - 5k events each, 500 steps, anti-kT R=0.4 jet clustering
  - Output: `/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid/bsm_grid/infer_holdout_ep055_5k/`
  - Plot script: `omnilearn_pp/scripts/plot_e008_bsm_holdout.py` → output to `plots_ep055/`
  - Submit script: `omnilearn_pp/submit_e008_holdout_infer_ep055.sh`
- **Motivation:** Cone-mass and parton-cone distributions looked poor at epoch ~19. The val_loss drop from 4.904 → 4.900 is small but training is still in early descent; checking whether multi-particle correlations (jet mass, cone mass) improve with more training.
- **Status:** RUNNING — job 54716471 submitted 2026-06-19.
- **Linked experiments:** E008 (training source), prior ep019 diagnostic (A-series).

---

### A006 — E007 partial checkpoint inference

- **Date submitted:** 2026-06-13
- **Goal:** Evaluate model generation quality from E007's auxcls-body checkpoint after 19/200 training epochs. Training was cancelled to get an early diagnostic look before full training completes.
- **Hypothesis / open question:** Does 19 epochs of auxcls-body training already shift per-process kinematic distributions toward truth vs the E000 baseline? Even a weak signal would confirm the aux loss is propagating useful gradients into the body.
- **Setup:**
  - Checkpoint: `/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/auxcls_body_5proc/pet_pp.weights.h5` (epoch 19/200, val_loss=5.447)
  - Script: `infer_pp_5proc_truelogn_comparison.py`, loaded into base `ProcLabelPET` (aux classifier layers in checkpoint silently skipped by TF2 H5 loader — confirmed safe)
  - 1k events/proc (5k total), 500 steps, val_start=400000, `--use_true_jet`
  - Output: `.../checkpoints/auxcls_body_5proc/infer_e007_partial_steps500/`
  - Job: 54408662
- **Status:** DEFERRED — job 54408662 cancelled while PENDING (2026-06-13). Re-submit with `omnilearn_pp/submit_a006_e007_partial.sh` when ready.
- **Linked experiments:** E007 (source checkpoint), A005 (E001 same-protocol run), A004 (baseline extension)

---

### A005 — E001 partial checkpoint inference

- **Date submitted:** 2026-06-13
- **Goal:** Evaluate model generation quality from E001's CFG checkpoint after 61/200 training epochs. Training was cancelled to get an early diagnostic look.
- **Hypothesis / open question:** Does 61 epochs of CFG dropout training already change per-process distributions vs E000 baseline (even at guidance_scale=0.0)? CFG training teaches the model unconditional denoising alongside conditional — the question is whether this reshapes the conditional distribution beneficially even before guidance is applied at inference.
- **Setup:**
  - Checkpoint: `/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/cfg_5proc_dropout10/pet_pp.weights.h5` (epoch 61/200, val_loss=5.421)
  - Script: `infer_pp_5proc_truelogn_comparison.py`, standard `ProcLabelPET` (same model class as E001 training)
  - guidance_scale=0.0 (base inference script; CFG boost not yet implemented in this script)
  - 1k events/proc (5k total), 500 steps, val_start=400000, `--use_true_jet`
  - Output: `.../checkpoints/cfg_5proc_dropout10/infer_e001_partial_steps500/`
  - Job: 54408661
- **Status:** DEFERRED — job 54408661 cancelled while PENDING (2026-06-13). Re-submit with `omnilearn_pp/submit_a005_e001_partial.sh` when ready.
- **Linked experiments:** E001 (source checkpoint), A006 (E007 same-protocol run), A004 (baseline extension)

---

### A004 — baseline 500-step inference extension

- **Date submitted:** 2026-06-13
- **Goal:** Extend the A002-s500 baseline run with 10k additional events (val_start=402000, no overlap with A002-s500's 400000–401999) to improve statistics for comparison against A005 (E001) and A006 (E007) partial-checkpoint results.
- **Setup:**
  - Checkpoint: E000 (`proc_label_5proc_p3/pet_pp.weights.h5`)
  - 2k events/proc (10k total), 500 steps, val_start=402000, `--use_true_jet`
  - Output: `.../checkpoints/proc_label_5proc_p3/infer_truejet_steps500_ext/`
  - Job: 54408659
- **Status:** DEFERRED — job 54408659 cancelled while PENDING (2026-06-13). Re-submit with `omnilearn_pp/submit_a004_steps500_ext.sh` when ready.
- **Linked experiments:** A002-s500 (prior baseline at same step count), A005/A006 (comparison targets)

---

### E003 — truejet comparison plots

- **Date:** 2026-06-11
- **Goal:** Visually and quantitatively compare sampled log_npart (E002a) vs true log_npart (E002b) inference against validation ground truth. Determine whether per-process distribution mismatches are caused by stage-1 multiplicity errors or stage-2 particle generation quality.
- **Script:** `omnilearn_pp/scripts/plot_e003_truejet_comparison.py`
- **Inputs:** `infer_20k_sampled/` (E002a), `infer_20k_truejet/` (E002b)
- **Outputs:** `/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/figures/E003_truejet_comparison/`
  - `particle_dists_comparison.png` — η, φ, pT, npart per process, 3-way overlay
  - `global_obs_comparison.png` — HT, MET, sphericity per process, 3-way overlay
- **JSD results (sampled vs truejet, lower = better):**

  | Process | η | pT | npart | HT |
  |---------|---|----|-------|----|
  | dijet  | ~0.019 / ~0.020 (~) | ~0.003 / ~0.003 (~) | 0.019 / **0.000** (<<) | 0.032 / 0.050 (>) |
  | zjets  | ~0.017 / ~0.017 (~) | ~0.003 / ~0.003 (~) | 0.002 / **0.000** (<<) | 0.011 / 0.010 (~) |
  | ttbar  | ~0.022 / ~0.023 (~) | ~0.004 / ~0.005 (~) | 0.028 / **0.000** (<<) | 0.027 / 0.037 (>) |
  | wjets  | ~0.018 / ~0.018 (~) | ~0.003 / ~0.003 (~) | 0.003 / **0.000** (<<) | 0.010 / 0.012 (>) |
  | wprime | ~0.030 / ~0.028 (~) | ~0.009 / ~0.010 (>) | 0.042 / **0.000** (<<) | 0.088 / 0.092 (>) |

- **Interpretation:** Stage-1 errors are confined to multiplicity. The "truejet" mode is conditioning on the true count of stable final-state hadrons, bypassing stage-1 entirely. Supplying that true count does not improve η or pT JSD for any process — those mismatches are intrinsic to how stage-2 generates individual particle kinematics, regardless of how many particles it is told to make. HT gets marginally *worse* with true npart for most processes, likely because stage-2 is calibrated to stage-1's slightly underestimated multiplicity and pT values partially compensate. W' has the worst stage-1 multiplicity bias (JSD 0.042 → 0.000 when bypassed), but even after fixing that, pT and η mismatches persist. **Conclusion:** the binding constraint is stage-2 process conditioning — specifically, how well the model distinguishes processes when generating particle kinematics — not the accuracy of stage-1 multiplicity prediction. This makes E001 (CFG to amplify process conditioning) and E007 (aux-cls to push process information into the body encoder) the correct next experiments.
- **Linked experiments:** E002a (sampled inference input), E002b (truejet inference input). Motivates continuing E001 and E007.

---

### E007 — auxcls_body_5proc

- **Goal:** Properly test whether auxiliary classification of process label from a learned representation strengthens process-discriminative features. E006 used the proc_token (a direct Dense projection of the one-hot label) as input to the classifier, which made the task trivially solvable and provided no useful gradient signal. E007 moves the classifier to the body output, where it must extract process info from a representation shaped by the diffusion task.
- **Hypothesis:** Forcing the body's mean-pooled representation to be process-discriminative will propagate process info more thoroughly through the model. The body's features influence the cross-attention queries in the head, so a process-aware body should produce more process-distinct denoising.
- **Setup:**
  - Scripts: `omnilearn_pp/scripts/proc_label_train_auxcls_body.py`, `omnilearn_pp/scripts/PET_pp_parton_vpar_auxcls_body.py`
  - Implementation: body output is mean-pooled (masked) → `auxcls_dense1` Dense(D, gelu) → `auxcls_classifier` Dense(n_proc). Gradients from cross-entropy loss flow back through the body, not through the proc_token path. Body optimizer updated with `loss_part + aux_weight * loss_aux`; head optimizer updated with `loss_jet + loss_part + aux_weight * loss_aux` including `aux_classifier.trainable_variables`.
  - Auxiliary loss weight: 0.1; `--aux_weight` CLI arg
  - No CFG dropout; always uses true label
  - All other hyperparameters: matched to E000/E001/E006 (epoch=200, batch=128, proj_dim=128, num_layers=8, num_gen_layers=2, lr=3e-4)
  - Data: 5-process joint training (dijet, zjets, ttbar, wjets, wprime)
  - Checkpoint dir: `/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/auxcls_body_5proc/`
- **Status:** RUNNING (job 54319521). Smoke test passed: loss_aux epoch 1=1.5998, epoch 2=1.5541 (started near log(5)=1.609; not collapsed). Three bugs fixed during smoke test: perturbed_part/perturbed_x name collision in train_step; RandomDrop in-place tensor assignment in layers.py; loss_body computed outside GradientTape scope.
- **Expected behavior:**
  - `loss_aux` starts near log(5) ≈ 1.6 and decreases over training; must NOT collapse to ~1e-7 (if it does, another implementation flaw exists).
  - val_loss expected near E000/E001/E006 range (5.4–5.5).
  - Real test: per-process kinematic distributions after training vs E000 (baseline), E001 (CFG), E006 (no-op aux control).
- **Linked experiments:** Corrects E006's design flaw. Comparison against E000, E001, E006 after all complete.

---

### E006 — auxcls_5proc

- **Goal:** Test whether adding an auxiliary cross-entropy classification loss on the proc_token embedding strengthens process conditioning. The hypothesis is that the diffusion loss provides weak gradient signal for what the proc_token should encode (A001 ablation showed only ~0.034 part-loss contribution); adding an explicit classification objective forces the embedding to be informative.
- **Hypothesis:** The proc_token, currently shaped only by diffusion gradients, will become more discriminative when explicitly trained to predict the process label. This should propagate to stronger per-process behavior in particle generation.
- **Setup:**
  - Scripts: `omnilearn_pp/scripts/proc_label_train_auxcls.py`, `omnilearn_pp/scripts/PET_pp_parton_vpar_auxcls.py`
  - Implementation: `proc_emb_1`/`proc_emb_2` Dense layers in `_build_vpar_generator_head` are shared with a post-hoc `cls_model` that adds a `proc_classifier` Dense. Gradients flow back through shared layers into the cross-attention proc_token path.
  - Auxiliary loss weight: 0.1; `--aux_weight` CLI arg
  - No CFG dropout (`cfg_drop_prob=0.0`); always uses true label
  - All other hyperparameters: matched to E000 baseline (epochs=200, batch=128, proj_dim=128, num_layers=8, num_gen_layers=2, lr=3e-4)
  - Data: 5-process joint training (dijet, zjets, ttbar, wjets, wprime)
  - Checkpoint dir: `/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/auxcls_5proc/`
- **Status:** RUNNING (job 54262560 → self-resubmitted as 54298778). Epoch 20/200 completed; val_loss ~5.424.
- **Status update (mid-training, ~epoch 20):** Auxiliary classifier loss collapsed to ~1e-7 by epoch 20, indicating the classifier is trivially solving the identity-mapping task: it predicts the one-hot label from the proc_token, which is itself derived from the same one-hot via two Dense layers. The aux loss provides essentially no gradient signal past the first few epochs, and the rest of training is functionally near-baseline. This is an implementation flaw: the auxiliary head sits on a Dense projection of the label, not on a learned representation that needs to encode process info.
- **Decision:** Let E006 run to completion as a control (no compute is saved by stopping). The result will tell us whether adding a no-op auxiliary head changes anything (it shouldn't — used as sanity check on experimental infrastructure). E007 will properly test the auxiliary classification idea by moving the classifier head to a learned representation.
- **Expected behavior:**
  - val_loss expected ~5.4–5.5 (similar to E000; auxiliary loss collapsed early so effectively baseline)
  - Real test: whether final kinematic distributions differ from E000 (they probably won't)
- **Linked experiments:** Comparison against E000 baseline. E007 is the corrected version of this experiment.

---

### E002 — infer_truejet_5proc

- **Goal:** Diagnostic to isolate stage-2 quality from stage-1 quality. Stage 1 is a DDPM that predicts log(N_part) conditioned on the hard-scatter partons; stage 2 then generates exactly N_part stable final-state hadrons given that multiplicity. The question: how much of the per-process kinematic distribution mismatch is attributable to stage-1 getting the wrong multiplicity, vs stage-2 generating the wrong particle kinematics?
- **Hypothesis:** Some fraction of the distribution problems are caused by stage 1 producing biased log_npart for some processes. If supplying the true multiplicity (bypassing stage 1) produces noticeably better per-process distributions, stage 1 is a major contributor and may need its own fix. If both modes look similar, stage 2 has problems beyond multiplicity.
- **Setup:**
  - Script: `omnilearn_pp/scripts/infer_pp_5proc_truelogn_comparison.py`
  - Checkpoint: E000's checkpoint at `/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/pet_pp.weights.h5`
  - Mode A (default, `--use_true_jet` off): stage-1 DDPM samples log_npart from parton conditioning → mask built from that sample → stage-2 generates that many hadrons
  - Mode B (`--use_true_jet`): true N_part (actual count of stable hadrons in the validation event) supplied directly → stage-1 DDPM bypassed entirely → stage-2 generates exactly the ground-truth number of hadrons
  - n_total=20000 events per process, all 5 processes, num_steps=50
- **Status:** COMPLETE (2026-06-11). Both modes ran interactively; 20k events per process per mode.
- **E002a result:** W' npart bias +14 particles (sampled 194.6 vs truth 181.3); SM processes within ~5 particles; multiplicity std over-dispersed for dijet/ttbar (109/113 vs truth 86/87).
- **E002b result:** npart exact_match=1.000 for all processes (true-jet bypass confirmed).
- **Analysis:** See E003 for full comparison.
- **Linked experiments:** Uses E000's checkpoint. E003 (planned plotting) depends on this.

---

### E001 — cfg_5proc_dropout10

- **Goal:** Test whether classifier-free guidance with 10% process-label dropout strengthens process conditioning. Phase 3 ablation (A001) showed the xattn proc_token contributes only ~0.034 to part loss when removed, indicating the model under-uses this signal. CFG is the standard fix for "model has conditioning info but uses it weakly."
- **Hypothesis:** Training with proc_label dropout teaches the model both conditional and unconditional denoising. At inference with guidance_scale > 0, the model is pushed to amplify process-specific differences, producing more process-distinct kinematic distributions than the baseline.
- **Setup:**
  - Training script: `omnilearn_pp/scripts/proc_label_train.py` with `cfg_drop_prob=0.10`
  - Inference script: `omnilearn_pp/scripts/infer_pp_proc_label.py` with `--guidance_scale {0.0, 1.0, 3.0, 5.0}`
  - Data: 5-process joint training (dijet, zjets, ttbar, wjets, wprime), same dataset as E000
  - Hyperparameters: matched to E000 baseline (epochs=200, batch=128, proj_dim=128, num_layers=8, num_gen_layers=2, lr=3e-4)
  - Expected checkpoint path: `/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/cfg_5proc_dropout10/pet_pp.weights.h5`
- **Status:** STAGED. Training script and inference script modified, smoke test passed (2 epochs training + inference at guidance_scale=0.0 and 1.0). Sanity check with guidance_scale=0.0 on E000 checkpoint produced mean_npart=248.8 identical to reference; feature distributions statistically consistent. Ready for full training submission.
- **Expected behavior:**
  - val_loss likely slightly higher than E000 baseline (5.418) because the model also learns unconditional denoising — expect 5.4–5.6 range.
  - Real test is generated kinematic distributions at guidance_scale > 0 after training completes.

---

### A002 — diffstep_sweep

- **Date submitted:** 2026-06-11
- **Goal:** Test whether η over-dispersion (generated normalized std ≈ 1.27 vs truth 1.0, causing particles outside |η| < 5 acceptance) is partially caused by coarse DDPM sampling at 50 steps.
- **Hypothesis:** If 200–500 steps significantly reduces normalized std toward 1.0, the issue is partially sampler quality and a simple fix (higher step count at inference) could close part of the gap. If higher steps barely change the result, the over-dispersion is structural to the trained model and requires conditioning interventions (E001 CFG, E007 aux-cls) to address.
- **Setup:**
  - Checkpoint: E000 (`proc_label_5proc_p3/pet_pp.weights.h5`)
  - Script: `infer_pp_5proc_truelogn_comparison.py` with `--use_true_jet` (isolates stage-2)
  - Step counts: 100 (job 54329939, 5k events), 200 (job 54329940, 5k events), 500 (job 54329941, 2k events — reduced from 5k as 500 steps × 5k × 5 processes ≈ 7.8h exceeds 4h wall)
  - 50-step baseline: existing E002b outputs at `infer_20k_truejet/`
  - Output dirs: `infer_truejet_steps100/`, `infer_truejet_steps200/`, `infer_truejet_steps500/`
  - Analysis script: `omnilearn_pp/scripts/analyze_a002_diffstep_sweep.py`
  - Output: `figures/A002_diffstep_sweep/summary_table.csv`, per-process eta overlay PDFs, summary metrics plot
- **Metrics:** η normalized-space std (target 1.0), Wasserstein distance to truth, fraction |η| > 5
- **Status:** COMPLETE (2026-06-13). Jobs finished 2026-06-12 (s100: 04:47, s200: 07:25, s500: 08:14).
- **Results:**

  | Process | W1 (50) | W1 (100) | W1 (200) | W1 (500) | norm_std (200) | oor% (200) |
  |---------|---------|----------|----------|----------|----------------|------------|
  | dijet   | 0.560   | 0.074    | **0.009**| 0.010    | 1.039          | 0.14%      |
  | zjets   | 0.533   | 0.081    | **0.015**| 0.008    | 1.123          | 0.18%      |
  | ttbar   | 0.620   | 0.087    | **0.011**| 0.006    | 1.002          | 0.13%      |
  | wjets   | 0.531   | 0.067    | **0.005**| 0.018    | 1.118          | 0.17%      |
  | wprime  | 0.616   | 0.092    | **0.020**| 0.010    | 0.665          | 0.05%      |

- **Interpretation:** The η over-dispersion observed at 50 steps was almost entirely a sampler artifact — the DDPM was not converging in 50 steps. η W1 improves ~60× going from 50 to 200 steps (e.g., dijet 0.560 → 0.009). Out-of-acceptance fraction (|η| > 5) drops from 1.0–1.3% at 50 steps to 0.13–0.19% at 200 steps. **200 steps is the convergence point** — 500 steps gives essentially identical results. The A003 conclusion that η over-dispersion was "structural to the model weights" was incorrect; it was a severely undersampled DDPM. Residual issues at 200 steps are small but real: zjets and wjets show normalized std ~1.12 (slightly over-dispersed), and wprime shows std ~0.67 (under-dispersed), consistent with process conditioning weakness. **All E001/E007 inference evaluations must be run at ≥ 200 steps**; prior 50-step numbers are misleading for absolute quality.
- **Outputs:** `figures/A002_diffstep_sweep/summary_table.csv`, per-process eta overlay PDFs, `summary_metrics_vs_steps.pdf` — all in `/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/figures/A002_diffstep_sweep/`
- **Linked experiments:** Motivated by A003. Revises interpretation of E002/E003 results (relative npart comparisons remain valid; absolute kinematic quality numbers at 50 steps are misleading). Informs E001/E007 evaluation protocol.

---

### A003 — normalization diagnostic

- **Date:** 2026-06-11
- **Motivation:** Generated η is peaked at ±5 in the 5-process model (Wasserstein distance ~0.6) while the 2-process model has W₁~0.006. Suspected cause: normalization mismatch — wrong stats applied during de-normalization, or stats computed on wrong dataset.
- **Findings (six-point diagnostic):**

  1. **5-process stats.json contents** (`/pscratch/sd/l/lcondren/MCsim/full_event_mixed/normalisation_stats.json`, created 2026-06-03):
     - `part_mean`: [−0.0025, 0.0006, 0.0007, +0.1168, 0.768, 0.0016]
     - `part_std`:  [2.319, 0.707, 0.707, 1.178, 0.933, 0.768]
     - `jet_mean/std`: [5.293] / [0.460]
     - `cond_mean/std`: 36-dim (6 partons × 6 features)

  2. **2-process stats comparison** (`full_event_fpcd/normalisation_stats.json`): part_std for η is 1.759 (vs 2.319 for 5-proc) and for log_pT is 0.682 (vs 1.178). Large differences are *expected* — different datasets and different physics processes.

  3. **Stats provenance:** `compute_norm_stats.py` produces 24-dim cond stats (4 partons × 6 features), but the 5-process stats.json has 36-dim cond stats. The stats were NOT generated by the current `compute_norm_stats.py`. They were generated by an earlier version or an ad-hoc script. However, the training/inference code handles size mismatches gracefully (truncates or pads to `max_partons × PARTON_FEAT`). The stats ARE correctly computed on the actual 5-process data — verified by matching SM+wprime combined statistics exactly:
     - logpT_mean SM+wprime: +0.1184 vs stats.json: +0.1168 ✓
     - logpT_std SM+wprime: 1.181 vs stats.json: 1.178 ✓
     Including wprime is correct (it's one of the 5 training processes).

  4. **Training script stats path:** `stats_path = os.path.join(flags.data_dir, 'normalisation_stats.json')` where `flags.data_dir` = `/pscratch/sd/l/lcondren/MCsim/full_event_mixed`. Confirmed reads the 5-process stats.

  5. **Inference script stats path:** `STATS_DIR = args.stats_dir or args.data_dir`, then `f'{STATS_DIR}/normalisation_stats.json'`. Default `args.data_dir` = same 5-process root. Both inference scripts (infer_pp_proc_label.py, infer_pp_5proc_truelogn_comparison.py) load the same file. **Training and inference normalization are self-consistent — no mismatch.**

  6. **Round-trip normalization check (dijet, 20k events):**
     - Raw η: mean=−0.015, std=2.398, range=[−5, +5]
     - After normalization: mean≈0, std≈1.03 ✓ (correct)
     - De-normalized *generated* η: mean=−0.065, std=**2.949**, range=[**−7.1, +7.8**]
     - De-normalized *truth* η: mean=+0.001, std=2.402, range=[−5, +5]
     - **The model generates normalized η with std=1.27 (should be ~1.0), corresponding to physical η up to ±7.8 — outside the [−5, +5] acceptance.**

- **Conclusion:** There is NO normalization mismatch. Stats are self-consistent between training and inference, and were correctly computed on the 5-process training data. The "peaked at ±5" symptom is **histogram overflow** — the plotting scripts likely bin from −5 to +5, causing out-of-acceptance generated particles (η up to ±7.8) to pile up at the boundary. The root cause is model quality: the diffusion model is generating particles outside the physical acceptance, with η std ~27% too wide. This is a stage-2 generation problem, consistent with the E003 conclusion that stage-2 kinematics are the binding constraint.
- **Action:** No normalization fix needed. Out-of-acceptance particles could be clipped at inference, but the underlying issue is generation quality. Monitoring E001/E007 to see if further training improves η calibration.
- **Revision (2026-06-13, A002 result):** The η over-dispersion diagnosis here was partially incorrect. The normalized std of 1.27 observed in point 6 was measured on 50-step inference output, which is severely undersampled. At 200 steps, normalized std drops to 1.00–1.12 for SM processes (see A002). The conclusion that it was "structural to the model weights" was wrong; it was a sampler convergence artifact.

---

### A001 — ablation_proc_paths

- **Date:** 2026-06-09
- **Goal:** Quantify the contribution of each process-label conditioning path to val_loss on the E000 checkpoint.
- **Method:** Post-load weight zeroing (no graph changes). Identified Dense layers by input shape `(None, 5)` via tensor connectivity tracing. For each config: reload checkpoint, zero target layers, call `model.evaluate()` on 10k val events.
- **Configs and results:**

  | Config | val_loss | part   | jet    | What was zeroed |
  |--------|----------|--------|--------|-----------------|
  | Baseline | 5.3243 | 4.4839 | 0.8404 | — |
  | A (xattn proc_token) | 5.3689 | 4.5178 | 0.8510 | Dense 51/52 in head |
  | B (additive cond_token) | 5.3431 | 4.4756 | 0.8676 | Dense 44/45 in head |
  | C (resnet proc_emb) | 5.4652 | 4.4843 | 0.9809 | Dense 59 in model_jet |
  | D (A+C combined) | 5.5167 | 4.5298 | 0.9869 | Dense 51/52 + 59 |

- **Interpretation:** Resnet jet-head path (C) dominates process conditioning — most process signal routes through stage 1 multiplicity prediction (jet loss Δ +0.1405 vs baseline). Cross-attention proc_token (A) contributes modestly to particle generation (part loss Δ +0.0339). Additive cond_token path (B) contributes essentially nothing useful to part loss (Δ −0.0083) and may be slightly harmful to jet loss (Δ +0.0272). Path D confirms C and A contributions are roughly additive.
- **Motivation for E001:** The small but real contribution from path A (xattn proc_token) suggests the model has the right mechanism but uses it weakly. CFG is the standard intervention to amplify weak conditioning signal.
- **Script:** `omnilearn_pp/scripts/ablation_eval.py`

---

### E000 — proc_label_5proc_phase3

- **Date:** ~2026-06-08
- **Goal:** Establish a multi-process baseline with Phase 3 architecture (process label as cross-attention token in generator head).
- **Setup:**
  - Architecture: `PET_pp_parton_vpar.py` (ProcLabelPET), Phase 3 xattn proc_token — process label projected to `(N, 1, D)` token, concatenated into cross-attention key/value set alongside 6 parton tokens giving `(N, P+1, D)` context per generator layer.
  - Data: 5-process joint training (dijet, zjets, ttbar, wjets, wprime), 400k events per process (80/20 train/val split)
  - Training: 200 epochs, Adam lr=3e-4, batch=128
  - Checkpoint: `/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/pet_pp.weights.h5`
- **Key result:** val_loss plateau ~5.418 at epoch 91 (part ~4.48, jet ~0.84). Similar to pre-Phase-3 baseline of 5.42. Phase 3 did not improve over Phase 2, likely because the additive cond_token path (B) was also active and potentially cancelling the xattn signal.
- **Baseline for:** A001 ablation, E001 CFG experiment, E002 true-jet comparison.

---

## Planned experiments (not yet submitted)

### E022 — bsm_grid_event_c_layers4

- **Date staged:** 2026-06-25
- **Slurm job:** 55094220; self-resubmitting
- **Goal:** Test whether increasing num_gen_layers from 2 to 4 in the E020c architecture improves event-level observable agreement.
- **Hypothesis:** More cross-attention layers in the generator head give particles more rounds of attention to the event token, allowing stronger use of the all-7 event conditioning. E020c showed residual error in conditioned observables that may be capacity-limited at num_gen_layers=2.
- **Architecture:** Identical to E020c (PET_pp_parton_vpar_bsm_event_c) except num_gen_layers=4. Event token carries all 7 features (MET×3 + cone_X×2 + cone_Y×2). All other hyperparameters unchanged: num_layers=8, proj_dim=128, batch=128, lr=3e-4, lr_body=1e-4.
- **Scripts:**
  - Architecture: `PET_pp_parton_vpar_bsm_event_c_layers4.py`
  - Training: `bsm_grid_train_event_c_layers4.py`
  - Inference: `infer_bsm_grid_event_c_layers4.py`
  - Submit: `submit_e022_bsm_grid_event.sh`
- **Checkpoint dir:** `/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid/bsm_grid_event_c_layers4/`
- **Smoke test:** PASS (2026-06-25) — 2 epochs, val_loss 4.64→3.76, num_gen_layers=4 confirmed, inference output written.
- **Diff from E020c:** exactly 1 line in arch (default), 3 lines in train (import, run_name, default), 3 lines in infer (import, run_name, default), 1 line in submit script (num_gen_layers arg).
- **Status:** RUNNING — Slurm job 55094220.
- **Comparison:** Run holdout inference with `infer_bsm_grid_event_c_layers4.py` after training; compare to E020c holdout inference using same observables.

---

### A009a/b/c — E020 mid-training holdout inference (oracle/truth-conditioned)

- **Date submitted:** 2026-06-25
- **Slurm jobs:** A009a: 55037853, A009b: 55037857, A009c: 55037860
- **Goal:** Early diagnostic of E020a/b/c at epoch ~49–51/200 (oracle/truth-conditioned only).
  Event features computed from HDF5 truth particles and passed directly to the model — upper bound on what event conditioning can achieve.
- **Setup:** 4 holdout points (250,250),(250,300),(300,250),(300,300); 5k events each; 4 GPUs in parallel (one holdout point per GPU, all 4 in same job). 500 steps.
- **Scripts:** `infer_bsm_grid_event_{a,b,c}.py` via `submit_e020{a,b,c}_holdout_infer_truth.sh`
- **Output dirs:** `/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid/bsm_grid_event_{a,b,c}/infer_holdout_truth/`
- **Status:** RUNNING — 3 jobs submitted 2026-06-25.

---

### E021 — Event conditioning comparison plots and Wasserstein analysis

- **Date staged:** 2026-06-19
- **Goal:** Quantify whether E020a/b/c event-level token conditioning improves generation of event-level observables vs the baseline (E010). Produces per-(mass-point, observable) overlay plots and a Wasserstein distance summary table.
- **Depends on:** E008 baseline holdout inference + E020a/b/c training + A009a/b/c oracle holdout inference complete.
- **Script:** `omnilearn_pp/scripts/compare_event_conditioning.py`
- **Observables:** MET magnitude, MET φ, cone_pT_X, cone_mass_X, cone_pT_Y, cone_mass_Y (primary); marginal η, log_pT (sanity check).
- **Outputs:**
  - `figures/E020_event_conditioning_comparison/<obs>_<mass_point>.pdf` — 1 PDF per (mass_point × observable) overlay; 4 curves (baseline, E020a, b, c) + truth reference.
  - `figures/E020_event_conditioning_comparison/wasserstein_table.csv` — W1(generated, truth) per (observable, mass_point, variant).
- **Status:** PLANNED — waiting for E020a/b/c to train and produce holdout inference outputs.
- **How to run:**
  ```
  python3 compare_event_conditioning.py \
      --baseline_dir  .../bsm_grid/infer_holdout_5k \
      --e020a_dir     .../bsm_grid_event_a/infer_holdout \
      --e020b_dir     .../bsm_grid_event_b/infer_holdout \
      --e020c_dir     .../bsm_grid_event_c/infer_holdout
  ```

---

### E020a/b/c — Event-level conditioning diagnostics (stage-2 isolation)

- **Date staged:** 2026-06-19
- **Goal:** Test whether adding a learned event-level token to the cross-attention KV set in the generator head improves generation quality for stage 2 of the BSM grid pipeline. Three parallel variants differing only in which event features are used.
- **Motivation:** The current model conditions each particle on hard-scatter parton kinematics but has no access to global event-level observables (MET, cone mass) that correlate with final-state topology. Adding an event token tests whether this information helps the diffusion score function.
- **Architecture change (all variants):** A new `inp_event` input is processed through two Dense layers → reshaped to `(N,1,D)` → concatenated with the 4 parton embeddings to form `cond_set = (N,5,D)` for cross-attention. Attention mask extended to `(N,1,5)`. The `inp_jet` global conditioning path (log_npart) is unchanged. Background events receive all-zero event features. Stats computed from signal files only; sin/cos features normalized with mean=0, std=1 (no-op z-score).
- **Variants:**

  | ID | Features | `num_event_feat` | `_SINCOS_INDICES` | Arch | Train | Infer |
  |----|----------|-----------------|-------------------|------|-------|-------|
  | E020a | log1p(MET_mag), sin(MET_phi), cos(MET_phi) | 3 | [1,2] | `PET_pp_parton_vpar_bsm_event_a.py` | `bsm_grid_train_event_a.py` | `infer_bsm_grid_event_a.py` |
  | E020b | log1p(cone_pT_X), log1p(cone_mass_X) | 2 | [] | `PET_pp_parton_vpar_bsm_event_b.py` | `bsm_grid_train_event_b.py` | `infer_bsm_grid_event_b.py` |
  | E020c | All 7: MET (3) + cone_X (2) + cone_Y (2) | 7 | [1,2] | `PET_pp_parton_vpar_bsm_event_c.py` | `bsm_grid_train_event_c.py` | `infer_bsm_grid_event_c.py` |

- **Event feature computation:** R_CONE=1.0; cone features use parton slots 2 (X) and 3 (Y) as cone axes; dR < 1.0 particle selection; massless 4-vector sum for invariant mass. All log1p-transformed scalar features normalized with signal-only z-score; sin/cos features left at mean=0 std=1.
- **Stats files:** `normalisation_stats_event_a/b/c.json` at `/pscratch/sd/l/lcondren/MCsim/wprime_signal/` — auto-computed by training script on first run if absent.
- **Inference:** Reads truth event features from HDF5 particle data, normalizes, passes as `event_feat` positional arg to `model.generate()`. This is a stage-2 isolation test: truth event features are used (not predicted by a stage-1 model).
- **Smoke test results (2026-06-19):** All three variants passed end-to-end (train 1 epoch → inference 50 events). val_loss (1-epoch, untrained): a=16.14, b=14.47, c=18.85. Expected overflow warnings in npart rounding from untrained model; safely clipped.
- **Pre-training sanity checks (2026-06-19):** All passed. Cone observables: cone_mass_X=196.7 GeV (expected ~100–350 for m_X=200), cone_mass_Y=295.1 GeV (expected ~150–400 for m_Y=300), MET=4.45 GeV. Stats files show no NaN/zero-std anomalies. Model forward pass (4, 20, 6) output, no NaN/Inf.
- **Status:** RUNNING — submitted 2026-06-19. Slurm: E020a=54738498, E020b=54738499, E020c=54738501. All self-resubmitting.
- **Submit scripts:** `submit_e020a/b/c_bsm_grid_event.sh`
- **Checkpoint dir:** `/pscratch/sd/l/lcondren/MCsim/wprime_signal/checkpoints_bsm_grid/{bsm_grid_event_a,b,c}/`
- **Linked experiments:** E008 (baseline model without event conditioning); compare holdout W1 at equal epochs.

---

### Phase 1 (SM multi-process)

- **E003 (complete 2026-06-11):** See detail section above.
- **E004 (planned, conditional on E001):** If E001 CFG training produces meaningfully better kinematic distributions at guidance_scale > 0, retrain with the additive cond_token path (path B from A001) removed. Tests whether eliminating the redundant path further strengthens the xattn proc_token contribution.
- **E005 (planned, conditional):** If CFG doesn't fully resolve process-conditioning weakness, try (a) PDG-feature dropout in parton tokens during training, and (b) auxiliary classification loss on the proc_token embedding. Run independently to isolate each effect.

### Phase 2 (BSM parameterized diffusion)

| ID | Status | Type | Run name | Notes |
|----|--------|------|----------|-------|
| E008 | RUNNING | training | `bsm_grid` | Slurm job 54707121; epoch 55/200, val_loss=4.900 |
| A007 | RUNNING | diagnostic (inference) | `bsm_grid → infer_holdout_ep055_5k` | Slurm job 54716471; 4 holdout pts, 5k events, 500 steps; plots → `plots_ep055/` |
| A008 | RUNNING | diagnostic (inference) | `bsm_grid → infer_trained_ep019_5k` | Slurm job 54677288; 4 trained pts (200,200)(200,350)(350,200)(350,350); input for mass-overlay plot |
| E016 | PLANNED | validation | `parnassus_validation` | Run parnassus_wrapper.py on E008 holdout inference output; compare reco distributions (multiplicity, pT, η) vs QCD training domain; critical check for OOD behavior |
| E020a | RUNNING | training | `bsm_grid_event_a` | Slurm 54738498; MET conditioning (3 feat); self-resubmitting |
| E020b | RUNNING | training | `bsm_grid_event_b` | Slurm 54738499; cone_X conditioning (2 feat); self-resubmitting |
| E020c | RUNNING | training | `bsm_grid_event_c` | Slurm 54738501; all-7 event features; self-resubmitting |

---

## Open questions

- Does CFG amplify the cross-attention proc_token's empirical contribution to particle generation? (E001 will answer)
- How much of the per-process kinematic distribution mismatch is caused by stage 1's multiplicity prediction errors? **Answered by E003:** η/pT mismatches are stage-2 intrinsic; stage-1 errors are limited to multiplicity only.
- Is the multi-process val_loss plateau primarily a process-conditioning failure or a different bottleneck (capacity, training duration, balance)? (Combination of E001 and E002 will narrow this)

---

## Resolved questions

- **Is per-parton cross-attention intact in the multi-process model?** Yes (Phase 1 code review, 2026-06-08).
- **Can the architecture fit individual processes?** Yes. Phase 2 confirmed: dijet alone reached val_loss 4.86 in 20 epochs on minimal data; ttbar alone 4.89. Capacity is not the bottleneck for individual processes.
- **Does the architecture have a working process-token path?** Partially. A001 ablation confirms the xattn proc_token contributes Δ +0.034 to part loss when zeroed (real but modest). Resnet path C (stage-1 multiplicity head) is doing most of the process-conditioning work.
