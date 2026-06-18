# Changelog

All significant changes to this codebase are recorded here.
Backups of modified files are stored in `.changelog_backups/<timestamp>/`.
Each entry records SHA-256 hashes so changes can be verified or reverted.

---

## [2026-06-18] — E008 holdout plots: parton-cone bug fix

### Summary
Fixed a parton-direction bug in `plot_e008_bsm_holdout.py` that caused all parton-cone truth histograms to be empty. The bug: `eta_p` was set to `float(parton_feat[slot, 0])`, but feature[0] in E008 parton data is `log_E ≈ 7–8`, not η. This placed all partons outside detector acceptance so no truth particles fell within ΔR < 0.4 of any parton. Fix: derive η from `pz/p = feature[3]` via `η = 0.5·ln((1+pz/p)/(1−pz/p))`, matching the original `plot_infer_wprime_holdout.py` approach.

### `omnilearn_pp/scripts/plot_e008_bsm_holdout.py` — bug fix
- `_jet_fl`: replaced `eta_p = float(pf_ev[slot, 0])` with `pz_e = clip(pf_ev[slot, 3]); eta_p = 0.5·log((1+pz_e)/(1−pz_e))`.
- `_parton_cone_measure`: same fix.
- Corrected stale comment on parton feature format (was `[η,sinφ,cosφ,logpT,…]`; correct is `[log_E,sinφ,cosφ,pz/p,…]`).
- **To revert:** restore `eta_p = float(parton_feat_arr[i, slot, 0])` in both functions.

---

## [2026-06-15] — E015: Parnassus full-event detector wrapper

### Summary
Built `parnassus_wrapper.py` to chain BSM diffusion output (infer_bsm_grid.py NPZ) through the full-event Parnassus detector surrogate and save HDF5. Smoke-tested on 100 events; no NaN/Inf, mean multiplicity preserved, pT mean 17.6→8.8 GeV (detector response applied). Runs in `pipeline_copy-gpu2` conda env. See EXPERIMENTS.md §E015 for full design notes.

### `omnilearn_pp/scripts/parnassus_wrapper.py` — new file
- Imports `FullEventFlowLightning` directly from `train_full_event_detector.py` (same `pipeline_copy-gpu2` env; no subprocess needed).
- Preprocessing replicates `FullEventDataset.__init__` exactly: `log(pT_GeV × 1000)` z-score per event for pT, η, φ.
- pflow mask = truth mask (1:1 particle mapping; valid because infer_bsm_grid.py output is a contiguous prefix mask).
- Output HDF5: `reco_particles (N,600,3)`, `reco_mask (N,600)`, `hadron_particles (N,500,3)`, `hadron_mask (N,500)`, attrs `{m_X, m_Y, event_count}`.
- Output path: `/pscratch/sd/l/lcondren/MCsim/parnassus_output/{run_name}/{mX}_{mY}/recoparticles.hdf5`.
- **To revert:** delete this file.

---

## [2026-06-14] — E008: BSM mass-grid diffusion (training, inference, plotting)

### Summary
Full W'→4q (X→YY→4q) BSM mass grid experiment: new mass-conditioned architecture, training on 141 signal+background files, 4-point 2×2 holdout, mid-training diagnostic inference (epochs ~19–24) on 4 held-out mass points, and holdout comparison plots. See EXPERIMENTS.md §E008 for full experimental protocol.

### `omnilearn_pp/scripts/PET_pp_parton_vpar_bsm.py` — new file
- Extends `PET_pp_parton_vpar.py` with two scalar mass-conditioning slots: `m_X/600` replaces parton slot 2 feature 2, `m_Y/600` replaces slot 3 feature 2. All other architecture identical to the vpar baseline.
- Four-parton layout: slots 0/1 = beam partons (never conditioned on mass); slots 2/3 = X and Y boson mass tokens.
- **To revert:** delete this file; revert any training scripts pointing at it.

### `omnilearn_pp/scripts/bsm_grid_train.py` — new file
- Training script for BSM mass-grid model. Loads `bsm_*.h5` files from grid dir (140 signal mass points + 1 QCD background). Weighted loss (`|w|`-weighted). Supports `--n_train` cap for OOM mitigation.
- Uses `PET_pp_parton_vpar_bsm` architecture, `--num_gen_layers 2` for memory efficiency.
- **To revert:** delete this file.

### `omnilearn_pp/scripts/infer_bsm_grid.py` — new file
- Inference script for BSM mass-grid model. Accepts `--m_X`, `--m_Y` to construct mass conditioning. Outputs `bsm_mX{mX}_mY{mY}.npz` with `parts_truth`, `parts_gen`, `mask`, `mask_gen`, `parton_feat`, `mass_x`, `mass_y`.
- Output particles are fully denormalized (physical units) before saving.
- **To revert:** delete this file.

### `omnilearn_pp/scripts/plot_e008_bsm_holdout.py` — new file
- Plots for E008 holdout inference: 5 plot types (particle dists, global obs, jet obs, jet images, parton cone), 4 mass points treated as separate processes. Adapted from `plot_infer_wprime_holdout.py` for E008 parton feature format (7-feature with mass/600 in slot 6; log_E in slot 0, pz/p in slot 3).
- **To revert:** delete this file.

### Submit scripts — new files in `omnilearn_pp/submit/`
- `submit_e008_bsm_grid.sh` — self-resubmitting training job (4h/job, 1 GPU, account m2616).
- `submit_e008_holdout_infer.sh` — final 10k-event holdout inference (post-convergence; 4 mass points × 5k).
- `submit_e008_holdout_infer_ep019.sh` — mid-training diagnostic, first 2 mass points (timed out at 4h).
- `submit_e008_holdout_infer_ep019_remaining.sh` — mid-training diagnostic, remaining 2 mass points.

### `omnilearn_pp/scripts/plot_infer_wprime_holdout.py` — new file (precursor)
- Precursor plot script for wprimeGrid NPZ format. Loads `mX*.npz` files. Same 5 plot types. Used for earlier W'→jj holdout runs before E008.
- **To revert:** delete this file.

### Submit scripts (wprimeGrid)
- `submit_wprimeGrid_1node.sh` — W'→jj grid training job (precursor to E008).
- `submit_infer_wprime_holdout.sh` — W'→jj holdout inference job.

---

## [2026-06-11] — E007: Auxiliary classification on body output (auxcls_body experiment)

### Summary
Created isolated copies of the E006 scripts (`_body` suffix) and moved the auxiliary classifier from the proc_token (E006's design, which was trivially solvable) to the mean-pooled body output. Also fixed three bugs found during smoke test: a variable name collision in train_step, an in-place tensor assignment incompatible with graph mode in `layers.py`, and a loss computation placed outside the GradientTape scope. See EXPERIMENTS.md §E007 for hypothesis and setup.

### `omnilearn_pp/scripts/PET_pp_parton_vpar_auxcls_body.py` — new file (copy)
- Exact copy of `PET_pp_parton_vpar_auxcls.py` at this date. No functional changes to the base architecture.
- Exists for E007 experiment isolation. The `_body` suffix indicates the classifier sits on body output, not the proc_token.
- **To revert:** delete this file.

### `omnilearn_pp/scripts/proc_label_train_auxcls_body.py` — new file (modified copy)
Copy of `proc_label_train_auxcls.py` with the following changes:

#### Import: point at _body architecture copy
- `from PET_pp_parton_vpar_auxcls_body import PET_pp_parton_vpar`
- **To revert:** change back to `from PET_pp_parton_vpar_auxcls import PET_pp_parton_vpar`.

#### `ProcLabelPET.__init__` — replace cls_model with aux_classifier
- `self.cls_model = self._build_cls_model()` → `self.aux_classifier = self._build_aux_classifier()`
- **To revert:** restore `self.cls_model = self._build_cls_model()`.

#### `ProcLabelPET._build_cls_model` → `_build_aux_classifier`
- Removed `_build_cls_model` (which fetched `proc_emb_1`/`proc_emb_2` named layers from the head).
- Added `_build_aux_classifier`: masked mean-pool of body output → Dense(D, gelu) → Dense(n_proc). Two inputs: `(None, D)` body tensor and `(None, 1)` mask.
- **To revert:** restore `_build_cls_model` method and remove `_build_aux_classifier`.

#### `ProcLabelPET._build_vpar_generator_head` — strip layer names
- Removed `name='proc_emb_1'` and `name='proc_emb_2'` from proc_token Dense layers (no longer needed by a classifier).
- **To revert:** add names back.

#### `WeightedProcLabelPET.train_step` — body-output aux loss
- Removed old `cls_model(y)` aux loss (label→proc_token path).
- Added explicit `body(perturbed_part, ...)` call before `model_part` call; passes `[body_out, mask]` to `aux_classifier`.
- `body_optimizer` now minimizes `loss_part + aux_weight * loss_aux` (body gets aux gradients).
- `optimizer` minimizes `loss_jet + loss_part + aux_weight * loss_aux` over `model_jet + head + aux_classifier.trainable_variables`.
- `loss_body` computed inside GradientTape scope (bug fix).
- **To revert:** restore the old `cls_model`-based aux loss block and optimizer calls.

#### Bug fix: perturbed_part/perturbed_x name collision
- Particle-level `perturbed_x` renamed to `perturbed_part` to avoid collision with the jet-level `perturbed_x` introduced later in the same tape block.
- **To revert:** would re-introduce the collision bug.

#### Bug fix (`layers.py`): RandomDrop in-place tensor assignment
- `RandomDrop.call` used `x[:,:,s:] = ...` in-place assignment which fails for symbolic tensors in graph mode.
- Replaced with `tf.concat([x[:,:,:s], x[:,:,s:] * drop_mask], axis=-1)`.
- **To revert:** restores the graph-mode incompatibility.

### `omnilearn_pp/submit_auxcls_body_5proc.sh` — new file
- Submit script for E007. Points at `proc_label_train_auxcls_body.py`, run_name=`auxcls_body_5proc`. Identical structure to `submit_auxcls_5proc.sh` (4h/job, 4 GPUs, self-resubmitting).
- **To revert:** delete this file.

---

## [2026-06-10] — E006: Auxiliary classification loss on proc_token (auxcls experiment)

### Summary
Created isolated copies of the architecture and training scripts for experiment E006. Added an auxiliary cross-entropy classification loss that shares the proc_token Dense layers from the generator head, forcing those layers to produce discriminative process embeddings. See EXPERIMENTS.md §E006 for hypothesis and setup.

### `omnilearn_pp/scripts/PET_pp_parton_vpar_auxcls.py` — new file (copy)
- Exact copy of `PET_pp_parton_vpar.py` at this date. No functional changes.
- Exists so the auxcls training script imports from a self-contained file rather than the canonical architecture file (which may change independently).
- **To revert:** delete this file.

### `omnilearn_pp/scripts/proc_label_train_auxcls.py` — new file (modified copy)
Copy of `proc_label_train.py` with the following changes:

#### Import: point at auxcls architecture copy
- `from PET_pp_parton_vpar_auxcls import PET_pp_parton_vpar`
- **To revert:** change back to `from PET_pp_parton_vpar import PET_pp_parton_vpar`.

#### `ProcLabelPET.__init__` — add aux_weight, build cls_model
- `cfg_drop_prob` default changed to `0.0` (this is not a CFG experiment).
- Added `aux_weight=0.1` parameter, stored as `self.aux_weight`.
- After `super().__init__()`, calls `self._build_cls_model()` and creates `self.loss_aux_tracker`.
- **To revert:** remove `aux_weight` param and `self.cls_model` / `self.loss_aux_tracker` construction.

#### `ProcLabelPET._build_vpar_generator_head` — name proc Dense layers
- Two Dense layers that build `proc_token` are now named `'proc_emb_1'` and `'proc_emb_2'`.
- Required so `_build_cls_model` can retrieve and share them by name.
- **To revert:** remove the `name=` kwargs from those two Dense calls.

#### `ProcLabelPET._build_cls_model` — new method
- Retrieves `proc_emb_1` and `proc_emb_2` from `self.head` via `get_layer()`.
- Adds a `proc_classifier` Dense on top and returns a `keras.Model` (`proc_cls_head`).
- Gradients through `cls_model` flow back through the shared Dense layers into the cross-attention proc_token path.
- **To revert:** delete this method.

#### `ProcLabelPET.metrics` property — new override
- Adds `loss_aux_tracker` to the metrics list so `aux` is logged each epoch.
- `val_aux` is always 0 (test_step does not call `cls_model`), which is intentional.
- **To revert:** delete this property override.

#### `WeightedProcLabelPET.train_step` — add auxiliary loss, remove CFG dropout
- CFG dropout block removed entirely (no `y_train`; uses `y` directly).
- Inside `GradientTape`: computes `proc_logits = self.cls_model(y)`, then `loss_aux = mean(softmax_cross_entropy(proc_label_true, proc_logits))`.
- Total loss: `loss = loss_jet + loss_part + aux_weight * loss_aux`.
- Optimizer receives `model_jet.trainable_variables + head.trainable_variables + cls_extra_vars`, where `cls_extra_vars` is only the `proc_classifier` layer's weights (avoids double-applying gradients to shared `proc_emb_1/2`).
- **To revert:** restore original CFG train_step from `proc_label_train.py`.

#### `parse_args()` — add `--aux_weight`
- New float argument `--aux_weight` (default 0.1).
- **To revert:** remove the `add_argument` line.

#### `main()` — pass `aux_weight` to model
- Added `'aux_weight': flags.aux_weight` to `model_kw`.
- **To revert:** remove from `model_kw`.

### `omnilearn_pp/submit_auxcls_5proc.sh` — new submission script
- Copy of `submit_proc_label_5proc_p3.sh` pointing at `proc_label_train_auxcls.py`, `RUN_NAME=auxcls_5proc`, with `--aux_weight 0.1`.
- Self-resubmitting pattern identical to baseline.
- **To revert:** delete this file (does not modify any existing file).

### Verification
- Smoke test (2 epochs, minimal data, 5 processes): PASSED.
- loss_aux epoch 1: 1.526 (near log(5)=1.609 for uniform 5-class init), epoch 2: 1.403 — decreasing as expected.
- No layer name lookup errors (proc_emb_1, proc_emb_2, proc_classifier all found correctly).
- val_loss finite; no NaN.

---

## [2026-06-10] — Classifier-free guidance (CFG) for ProcLabelPET

### Summary
Added CFG support to the 5-process pp pipeline so that process-label conditioning strength can be amplified at inference time. The mechanism: during training, randomly null the one-hot process label for a fraction of events so the model learns both conditional and unconditional denoising; at inference, combine the two predictions to steer generation toward the desired process.

### `omnilearn_pp/scripts/proc_label_train.py`

#### `ProcLabelPET.__init__` — add `cfg_drop_prob` argument
- Added `cfg_drop_prob=0.1` kwarg (float, default 0.10 = 10% dropout rate).
- Stored as `self.cfg_drop_prob`; inherited by `WeightedProcLabelPET` without any further change.
- **To revert**: remove the `cfg_drop_prob` arg and `self.cfg_drop_prob = cfg_drop_prob` line.

#### `WeightedProcLabelPET.train_step` — process-label dropout
- Before the `tf.GradientTape` block, compute a per-event binary drop mask: `drop = (tf.random.uniform([batch_size, 1]) < self.cfg_drop_prob)`.
- Build `y_train` by zeroing `y[:, P*PF+P:]` (the one-hot process label slice) for dropped events, keeping parton kinematics and parton mask intact.
- Pass `y_train` (not `y`) to both `model_part` and `model_jet` so both heads see the same conditioning per event.
- `test_step` is **unchanged**: validation always uses full conditioning so val_loss measures conditional denoising quality.
- **To revert**: remove the CFG comment block and `y_train` construction; change `y_train` back to `y` in the two model calls.

### `omnilearn_pp/scripts/PET_pp_parton_vpar.py`

#### `evaluate_models` — add `guidance_scale` parameter
- New signature: `evaluate_models(self, head, body, x, jet, mask, t, cond, w=0.0, guidance_scale=0.0)`.
- When `guidance_scale > 0`: runs head twice — once with `cond` (conditional) and once with `cond` where the process label slice `cond[:, P*PF+P:]` is zeroed (unconditional). Combines: `v_guided = (1 + guidance_scale) * v_cond - guidance_scale * v_uncond`. Body is called only once; its output is shared between both head calls.
- When `guidance_scale == 0` (default): code path is identical to the original (single head call). No performance overhead.
- **To revert**: restore the original two-line body (`v = body(...); v = head(...); return mask * v`), remove `guidance_scale` param.

#### `second_order_correction` — propagate `guidance_scale`
- Added `guidance_scale=0.0` to the `@tf.function` signature.
- Passes `guidance_scale=guidance_scale` to `evaluate_models`.
- **To revert**: remove the param and its forward to `evaluate_models`.

#### `DDPMSampler` — propagate `guidance_scale`
- Added `guidance_scale=0.0` to the `@tf.function` signature.
- Passes `guidance_scale=guidance_scale` to `evaluate_models` and `second_order_correction`.
- **To revert**: remove the param and its two forwards.

#### `generate` — expose `guidance_scale` to callers
- Added `guidance_scale=0.0` kwarg.
- Passes it to the `DDPMSampler` call for particle generation.
- Stage-1 jet sampling is unaffected (jet DDPMSampler call uses `jet is None` path which never calls `evaluate_models`; in practice `jets=` is always supplied so stage-1 is bypassed entirely).
- **To revert**: remove the `guidance_scale` kwarg and its forward.

### `omnilearn_pp/scripts/infer_pp_proc_label.py`

#### Add `--guidance_scale` CLI flag
- New flag: `--guidance_scale` (float, default 0.0).
- Passed as `guidance_scale=args.guidance_scale` to `model.generate(...)`.
- **To revert**: remove the `add_argument` line and the `guidance_scale=args.guidance_scale` kwarg in the generate call.

### Verification
- Smoke test (2-epoch training + 50-step inference at guidance_scale=0.0 and 1.0): PASSED.
- Inference sanity check with guidance_scale=0.0 on proc_label_5proc_p3 checkpoint (200 dijet events, 50 steps): no NaN; mean_npart=248.8 identical to reference; feature means/stds consistent within stochastic sampling noise (different random seeds). Confirms guidance_scale=0.0 path is equivalent to original behaviour.

---

## [2026-06-09] — Ablation evaluation: proc-label conditioning paths

### Summary
Added tools to measure how much each of the three process-label conditioning paths contributes to val_loss, using post-load weight zeroing so the checkpoint loads without issues.

### `omnilearn_pp/scripts/ablation_eval.py` — new evaluation script
- Builds `ProcLabelPET` with unchanged graph (no ablation flags in graph construction).
- Loads checkpoint; identifies proc-label Dense layers by input shape `(None, 5)` using tensor connectivity tracing.
- For each ablation config: reloads checkpoint, zeros target layer weights, calls `model.evaluate()`.
- Runs configs baseline / A / B / D in a single invocation; includes prior C result in the table.
- Mocks horovod at import time so the script runs without horovodrun or srun.

### `omnilearn_pp/scripts/proc_label_train.py` — revert ablation flags
- Removed the three ablation boolean kwargs (`ablate_xattn_proc_token`, `ablate_additive_proc`, `ablate_resnet_proc`) that were added in the previous session. They caused Keras Dense layer counter shifts via `tf.zeros_like` in the functional model graph, breaking weight loading.

### Ablation results (proc_label_5proc_p3 checkpoint, 10k val events)
| Config | val_loss | part   | jet    | What is zeroed |
|--------|----------|--------|--------|----------------|
| Baseline | 5.3243 | 4.4839 | 0.8404 | — |
| A      | 5.3689 | 4.5178 | 0.8510 | proc_token xattn (head Dense 51/52) |
| B      | 5.3431 | 4.4756 | 0.8676 | proc_emb additive cond_token (head Dense 44/45) |
| C      | 5.4652 | 4.4843 | 0.9809 | proc_emb resnet (jet Dense 59) |
| D      | 5.5167 | 4.5298 | 0.9869 | A + C |

---

## [2026-06-09] — num_proc_labels in PET_pp_parton_vpar; jets=jet_truth in inference

### `omnilearn_pp/scripts/PET_pp_parton_vpar.py` — `num_proc_labels` parameter
- Added `num_proc_labels=5` to `PET_pp_parton_vpar.__init__` with process label cross-attention and ResNet conditioning built into the base class.
- `num_cond` = `max_partons * parton_feat + max_partons + num_proc_labels`.
- When `num_proc_labels=0`, behaviour reduces to the original kinematics+mask form.

### `omnilearn_pp/scripts/infer_pp_proc_label.py` — pass true log_npart via `jets=`
- `_load_proc` now computes `jet_truth = (log(npart) - jet_mean) / jet_std` and returns it.
- Inference loop passes `jets=d['jet_truth']` to `model.generate()`, bypassing stage-1 (ema_jet) and conditioning the particle generator on ground-truth multiplicity.
- Verified: 100% npart match between truth and generated events.

---

## [2026-06-08] — Correlation tensor evaluation metric; self-attention confirmed

### MG pipeline — parton_generator

#### Architecture verification: self-attention between outgoing parton tokens (confirmed present, no change needed)
- **PET body** (`omnilearn_pp/scripts/PET.py`, `PET_body`): applies full `MultiHeadAttention(x1, x1)` over all parton tokens in every body layer. Noise prediction for any token depends on the noisy states of all other tokens through the attention computation. ✓
- **Theory generator head** (`parton_generator/models/parton_gen_model.py`, `_build_theory_generator_head`): applies `MultiHeadAttention(self_attn_i)(x1, x1, x1)` (parton self-attention) AND `MultiHeadAttention(xattn_i)(query, key=inp_cond, value=inp_cond)` (cross-attention to theory embedding) in every generator head layer. ✓
- Architecture satisfies the requirement: each outgoing parton is a token, transformer self-attention over the parton set is applied during denoising at both the body and head levels, and theory/process conditioning enters as separate cross-attention keys/values. No model code changes required.

#### `parton_generator/eval/correlations.py` — new correlation tensor computation module
- **Summary**: Implements `compute_correlation_tensor(events, mask) → (P, P, NF, NF)` where `corr[p1, p2, f1, f2]` is the Pearson correlation between feature f1 of parton p1 and feature f2 of parton p2 across N events. NF=5 (features 0-4: log_E, sin_phi, cos_phi, pz/E, pdg_norm; feature 5 = occupancy bit excluded as it is identically 1 for valid partons). Also implements `correlation_frobenius_diff(corr_truth, corr_gen) → float` (Frobenius norm of element-wise difference, NaN positions excluded). Uses `np.corrcoef` on concatenated feature blocks for efficiency.
- **Evaluation only**: this metric is computed on generated vs. truth events for diagnosis. It is not added to the training loss. The model learns correlation structure through the per-event score-matching objective combined with self-attention over outgoing parton tokens.
- **Files created**: `parton_generator/eval/__init__.py`, `parton_generator/eval/correlations.py`

#### `parton_generator/scripts/train.py` — correlation tensor evaluation hook
- **Summary**: Added `evaluate_correlations()` function and call in the training loop. Every `--corr_eval_freq` epochs (default 10), generates `--corr_eval_n` events (default 2000) using `--corr_eval_steps` DDPM steps (default 50), computes correlation tensors for both generated and truth val events, logs the Frobenius norm, and writes `corr_epochNNNN.npz` to the checkpoint directory. Uses the same theory conditioning as the truth events for a fair comparison.
- **New args**: `--corr_eval_freq` (int, default 10), `--corr_eval_n` (int, default 2000), `--corr_eval_steps` (int, default 50). Set `--corr_eval_freq 0` to disable.
- **Files modified**: `parton_generator/scripts/train.py`

---

## [2026-06-08] — Phase 1 simplification: generate outgoing partons only

### MG pipeline — parton_generator

#### `parton_generator/data/make_theory_hdf5.py` — drop initial-state parton slots
- **Summary**: Added `pf = pf[:, 2:, :]` immediately after reading `parton_features` from flat HDF5, discarding slots 0 (incoming beam+, pz > 0) and 1 (incoming beam−, pz < 0). Only final-state parton slots (2+) are written into the theory-group HDF5 used for training.
- **Reason**: Phase 1 simplification. The model now generates only the outgoing hard-scatter partons. No initial-state partons appear in the generation target. The PDF-weighted marginal over outgoing partons is correctly absorbed from the empirical training distribution (MG5 generates events with the full PDF weighting already applied). No LHAPDF or initial-state sampling is needed at inference time.
- **ISR note**: This correctly captures the PDF-weighted cross section in the outgoing distribution. ISR-dependent shower observables are not currently modeled. If later validation against ISR-sensitive observables shows measurable bias, initial-state conditioning will need to be added back.
- **Changed defaults**: `--max_partons` default changed from 6 to 4. `--max_events` now also accepts `--n_events` for smoke-test compatibility.
- **Files modified**: `parton_generator/data/make_theory_hdf5.py`

#### `parton_generator/submit_train.sh` — updated `--max_partons` to 4
- **Summary**: Changed `--max_partons 6` to `--max_partons 4` to match the final-state-only data.
- **Files modified**: `parton_generator/submit_train.sh`

#### Architecture confirmation (no change required)
- The current `TheoryPartonGenModel` already uses a fully tokenized representation: each outgoing parton is a separate token `(B, max_partons, 6)` processed by PET self-attention body + generator head. The generator head applies self-attention over outgoing parton tokens and cross-attention to the theory embedding `(B, SEQ_LEN, d_model)` as keys/values. This satisfies the per-token self-attention + conditioning cross-attention architecture exactly. No model code changes were needed.

#### Smoke-test fixes
- **`scripts/run_mg_smoke_test.sh`**: Fixed wrong `--num_part` argument (→ `--max_partons 4`), wrong `--data_path` (→ `--data_dir`), and invalid `--n_train`/`--n_val` (→ `--max_events 800`).
- **`.claude/agents/smoke-test-mg.md`**: Same fixes to agent defaults.

---

## [2026-06-08] — Session: Phase 3 proc-as-token, parton_generator pipeline, wprimeGrid plots

### Pythia pipeline — omnilearn_pp

#### `omnilearn_pp/scripts/proc_label_train.py` — Phase 3: process label as cross-attention token
- **Summary**: Promote the process label from a global additive signal on `cond_token` to a separate token `(N, 1, D)` concatenated into the cross-attention key/value set alongside the 6 parton tokens, giving `(N, P+1, D)` context for each generator layer.
- **Reason**: Phase 2 confirmed the model had sufficient capacity but was hitting a multi-process compromise floor. Giving each generator layer direct per-position access to the process identity (not just a global bias) is the targeted architectural fix.
- **Files modified**: `omnilearn_pp/scripts/proc_label_train.py`
- **Key change** (`_build_vpar_generator_head`):
  - Added `proc_token = Dense(D, gelu) → Dense(D) → expand_dims(axis=1)` giving shape `(N, 1, D)`
  - Changed `cond_set = concat([parton_emb, proc_token], axis=1)` → shape `(N, P+1, D)`
  - Extended `attn_mask` with a trailing `ones_col` so the process token is always unmasked
  - Old additive path (`cond_token + proc_emb`) kept active during first test run
  - Cross-attention now queries against `cond_set` instead of `parton_emb`

#### `omnilearn_pp/submit_proc_label_5proc_p3.sh` — new Phase 3 training submission
- **Summary**: New sbatch script identical in hyperparameters to `submit_proc_label_5proc.sh` (baseline that plateaued at val_loss 5.42) except `RUN_NAME=proc_label_5proc_p3`.
- **Reason**: Apples-to-apples comparison of Phase 3 architecture against the plateaued baseline.
- **Files created**: `omnilearn_pp/submit_proc_label_5proc_p3.sh`

#### `omnilearn_pp/scripts/plot_infer_wprime_holdout.py` — new wprimeGrid inference plot script
- **Summary**: Full 5-plot-type script for wprimeGrid NPZ inference output. Loads `mX*.npz` files from holdout directory. Plots: particle distributions, global observables, jet observables (per mass point), jet images (per mass point), parton cone (per mass point).
- **Reason**: wprimeGrid inference job (54044802) completed; needed plots equivalent to `plot_infer_5proc.py` but adapted to wprimeGrid NPZ format and per-mass-point structure.
- **Files created**: `omnilearn_pp/scripts/plot_infer_wprime_holdout.py`

---

### MG pipeline — parton_generator

#### `parton_generator/data/make_theory_hdf5.py` — fix missing `proc=None` argument
- **Summary**: Added `None` as the `proc` positional argument in the `write_theory_group` call site.
- **Reason**: `write_theory_group` was updated to accept a `proc` parameter between `params` and `parton_features`, but the call site was not updated. The numpy parton array `pf` was landing in the `proc` slot, causing `AttributeError: 'numpy.ndarray' object has no attribute 'initial_state'`.
- **Files modified**: `parton_generator/data/make_theory_hdf5.py` (line 118)

#### `parton_generator/submit_train.sh` — added `--val_frac 0.2`
- **Summary**: Added `--val_frac 0.2` to use 400k train / 100k val split from the 500k-event-per-process SM dataset.
- **Reason**: Default val_frac was too small relative to the dataset size; 80/20 split gives stable validation metrics.
- **Files modified**: `parton_generator/submit_train.sh`

#### `/pscratch/sd/l/lcondren/MCsim/theory_gen/sm_processes.hdf5` — new SM process dataset
- **Summary**: Theory-group-format HDF5 with all 4 SM processes (dijet, zjets, wjets, ttbar), 500k events each.
- **Reason**: parton_generator training requires theory-group HDF5 format; flat HDF5 from `full_event_mixed` was converted via `make_theory_hdf5.py`.
- **Note**: Data file, not a script — not backed up here; regenerate with `make mg-datagen`.

---

### Infrastructure

#### `Makefile` — new pipeline shortcuts
- **Summary**: Top-level Makefile with targets for datagen, train (sbatch), infer, plots, and smoke tests for both pipelines, plus a `changelog-entry` helper.
- **Files created**: `Makefile`

#### `scripts/append_changelog.sh` — backup + changelog helper
- **Summary**: Backs up named files to `.changelog_backups/<timestamp>/`, records sha256 hashes, appends a structured entry to `CHANGELOG.md`.
- **Files created**: `scripts/append_changelog.sh`

#### `scripts/run_pythia_smoke_test.sh`, `scripts/run_mg_smoke_test.sh` — smoke test shell scripts
- **Summary**: Shell wrappers for the two pipeline smoke tests; use `horovodrun --gloo -np 1`, 1 epoch, minimal data, /tmp scratch.
- **Files created**: `scripts/run_pythia_smoke_test.sh`, `scripts/run_mg_smoke_test.sh`

---
