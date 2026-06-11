# Experiment Ledger

Source of truth for what's running, what's done, and what each result means.

Last updated: 2026-06-11

---

## Active experiments (running, queued, or staged)

| ID | Status | Submitted | Type | Run name | Slurm job | Notes |
|----|--------|-----------|------|----------|-----------|-------|
| E007 | RUNNING | 2026-06-11 | training | `auxcls_body_5proc` | 54319521 | Aux cls on body output (option 4a); fixes E006 design flaw; self-resubmitting; 4h/job |
| E001 | RUNNING | 2026-06-10 | training | `cfg_5proc_dropout10` | 54261590 | CFG p=0.10 proc-label dropout; self-resubmitting; 4h/job |
| E002a | RUNNING | 2026-06-10 | inference (diagnostic) | `proc_label_5proc_p3 → infer_20k_sampled` | interactive/login17 | E000 ckpt; stage-1 sampled log_npart; running directly on login17 A100 |
| E002b | RUNNING | 2026-06-10 | inference (diagnostic) | `proc_label_5proc_p3 → infer_20k_truejet` | interactive/login17 | E000 ckpt; true log_npart (--use_true_jet); running directly on login17 A100 |

---

## Recently completed experiments

| ID | Completed | Type | Run name | Key result | Notes |
|----|-----------|------|----------|------------|-------|
| E006 | 2026-06-11 | training | `auxcls_5proc` | Cancelled at epoch 20; aux loss collapsed to ~1e-7 (trivial identity task) | Design flaw: classifier on proc_token = identity mapping; ran 20/200 epochs |
| E000 | 2026-06-08 | training | `proc_label_5proc_p3` | val_loss plateau ~5.418 at epoch 91 | Phase 3 xattn proc_token; baseline for all subsequent work |
| A001 | 2026-06-09 | ablation | `ablation_proc_paths` | Resnet path C dominates; xattn path A modest; additive path B negligible | Post-load weight zeroing on E000 checkpoint |

---

## Experiment details

(Most recent first.)

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

- **Goal:** Diagnostic to isolate stage 2's quality from stage 1's quality. Compare inference using sampled log_npart (stage 1 output) vs true log_npart (from validation data).
- **Hypothesis:** Some fraction of the kinematic distribution problems are caused by stage 1 producing biased log_npart for some processes. If true-log_npart inference produces noticeably better per-process distributions, stage 1 is a major contributor and may need its own intervention (corrector net, retraining, or being replaced). If both modes look similar, stage 2 has problems beyond multiplicity.
- **Setup:**
  - Script: `omnilearn_pp/scripts/infer_pp_5proc_truelogn_comparison.py`
  - Checkpoint: E000's pre-CFG checkpoint at `/pscratch/sd/l/lcondren/MCsim/full_event_mixed/checkpoints/proc_label_5proc_p3/pet_pp.weights.h5`
  - Mode A (default, `--use_true_jet` off): stage 1 samples log_npart → writes to `infer_20k/`
  - Mode B (`--use_true_jet`): ground-truth log_npart supplied → writes to `infer_20k_truejet/`
  - n_total=20000 events per process, all 5 processes, num_steps=50
- **Status:** STAGED. Script ready and sanity-tested (200 events/process, 10 steps, npart match=1.000 for all processes when `--use_true_jet` is set). Awaiting full inference submission.
- **Sanity check result:** `/tmp/sanity_truejet/` — 5 processes × 200 events; `npart match=1.000` for all processes confirms true-jet bypass is working correctly.
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

- **E003 (planned):** Plotting comparison from E002's outputs. Compare per-process kinematic distributions (η, φ, log_pT, multiplicity, jet observables) between truejet and sampled-jet inference, against validation data ground truth. Output to `figures/E003_truejet_comparison/`. Depends on E002 completing.
- **E004 (planned, conditional on E001):** If E001 CFG training produces meaningfully better kinematic distributions at guidance_scale > 0, retrain with the additive cond_token path (path B from A001) removed. Tests whether eliminating the redundant path further strengthens the xattn proc_token contribution.
- **E005 (planned, conditional):** If CFG doesn't fully resolve process-conditioning weakness, try (a) PDG-feature dropout in parton tokens during training, and (b) auxiliary classification loss on the proc_token embedding. Run independently to isolate each effect.

---

## Open questions

- Does CFG amplify the cross-attention proc_token's empirical contribution to particle generation? (E001 will answer)
- How much of the per-process kinematic distribution mismatch is caused by stage 1's multiplicity prediction errors? (E002 + E003 will answer)
- Is the multi-process val_loss plateau primarily a process-conditioning failure or a different bottleneck (capacity, training duration, balance)? (Combination of E001 and E002 will narrow this)

---

## Resolved questions

- **Is per-parton cross-attention intact in the multi-process model?** Yes (Phase 1 code review, 2026-06-08).
- **Can the architecture fit individual processes?** Yes. Phase 2 confirmed: dijet alone reached val_loss 4.86 in 20 epochs on minimal data; ttbar alone 4.89. Capacity is not the bottleneck for individual processes.
- **Does the architecture have a working process-token path?** Partially. A001 ablation confirms the xattn proc_token contributes Δ +0.034 to part loss when zeroed (real but modest). Resnet path C (stage-1 multiplicity head) is doing most of the process-conditioning work.
