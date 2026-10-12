# Blog assets — four loops of agency (UR_dVRK + GR00T-H)

All GIFs are real data from the dual-UR5 + Robotiq Isaac Sim cell running the
`all_blocks_v2_20261012_012734/checkpoint-10000` GR00T-H policy (10k steps, 134 demos,
frozen VLM + DiT action expert, horizon 25 @ 15 Hz). Task: "pick up the block and place
it on the target". No mock-ups — every clip is a live closed-loop rollout.

## Directory map

| dir | contents |
|---|---|
| `loop1/` | action refinement: flow-matching denoise, step sweeps, DART, tangent/normal geometry, SmolVLA contrast |
| `loop2/` | predictive planning: plan-vs-reality, synced composites, latent MPC, V-JEPA2-AC probe + CEM, Cosmos WAM |
| `loop3/` | receding-horizon control: K-sweep clips, temporal-context frames |
| `loop4/` | self-improvement: old-vs-new policy, RSI compatibility gate |
| `policies/` | **pretrained foundation models driving the cell closed-loop** — per-model GIFs + JSON reports |
| `scripts/` | reusable probe/viz/CEM scripts |
| `pretrained_models_survey.md` | full pretrained-model survey (below) |

## Loop 1 — action refinement (flow matching denoising)

| file | what it shows |
|---|---|
| `loop1/loop1_denoise.gif` | GR00T-H's DiT denoising a 25-step action chunk over Euler steps: the predicted left-EE trajectory starts as pure noise and contracts onto the recorded expert trajectory. Observation panel, 3-D plan, top-down XY vs GT, z-profile + gripper channel. (mid-approach frame, ep0 t=35) |
| `loop1/loop1_denoise_t175.gif` | same render at **grasp onset** (ep0 t=175) — the gripper channel visibly denoises from noise to the close command |
| `loop1/loop1_denoise_16step.gif`, `loop1/loop1_denoise_t175_16step.gif` | finer 16-step sweeps of both |
| `loop1/loop1_denoise.npz` | raw per-step chunks for re-rendering (`viz_denoise.py --render-only --pick <t>`) |
| `loop1/loop1_denoise*.mp4` | source videos |
| `loop1/loop1_steps{1,2,8,32}.gif` | **step-count sweep** — same observation, Euler schedules of 1/2/8/32 steps (deployed = 4). Even 1 step lands within ~5 mm of GT xyz, but path smoothness and the grip channel sharpen with steps (32-step resolves the close command to ~0.4). Mean \|plan−GT\|: 1→5.0, 2→5.6, 4→5.5, 8→6.4, 32→7.3 mm. |
| `loop1/loop1_steps_finals_2x2.png` | static 2×2 tile of converged plans (N=1/4/8/32) — "why iterate at all" figure |
| `loop1/loop1_tangent_normal.{png,gif,npz}` | **manifold-geometry measurement** (52 held-out chunks, 250-d whitened action space): vs the demonstrated trajectory ray, **90% of error energy is off-trajectory**; vs the full 5-D tangent subspace of the demonstrated-chunk manifold (95% variance in just 5/250 dims!), still ~50% off-manifold per sample. The residual wrongness is carried mostly by **rot6d (77% of error energy)** — the policy drifts into gripper orientations demos never visit. Scripts: `sim/gr00t_h/tangent_normal_eval.py` + `tangent_normal_analyze.py`. |
| `loop1/loop1_dart_demos.gif` | real DART-style demo collection: recorder patched with `--noise-mm` (Gaussian noise on commanded targets during `move()` phases only; gates/settle still converge → clean commands producing corrective motion). 3/3 episodes succeeded at 8 mm → `sim/outputs/episodes_physical_block_dart/` — perturbed path + expert recovery = the demonstration "tube". |

## Loop 1/3 — DART-style actuation noise (covariate shift)

| file | what it shows |
|---|---|
| `loop1/loop1_dart_noise8mm_success.gif` | Same seed-42 episode where the clean rerun *failed*: with 8 mm Gaussian noise injected into every executed EE target, the policy still grasps, lifts 87 mm and places (52 mm off). Caption candidate: physical disturbance changes which states the policy visits — robustness can't be learned from clean demos alone. |
| `loop1/loop1_smolvla_base_flail.gif` | **Contrast clip**: `lerobot/smolvla_base` (NOT fine-tuned on our cell) drives the same task — well-formed but semantically wrong actions; the arm flails for 46 s, 1433/700 steps hit safety clamps, and it knocks the block onto its side (90° tilt, 258 mm displaced, never lifted). "Base VLA vs fine-tuned VLA" — what BC actually buys you. Source: `rollouts/smolvla_base_ep0/` |
| `loop1/loop1_smolvla_vs_groot.gif` | side-by-side: SmolVLA-base flail (L) vs GR00T fine-tuned pick-and-place (R) |
| `loop1/loop1_dart_demos.gif` | DART **at demo-collection time** (not policy execution): `demo_record_physical.py --noise-mm 8` injects per-step Gaussian noise into the scripted expert's commanded tool0 xyz during the open-loop move segments; the recorder's gate-driven holds/settle corrections then pull it back — the "expanded demonstration tube". Source: `sim/outputs/episodes_physical_block_dart/episode_001.hdf5` (episodes 0–2 all succeeded at 8 mm, ~340 noisy steps each, slip ≤1.2 mm). |

## Loop 2 — predictive planning / imagination

| file | what it shows |
|---|---|
| `loop2/loop2_plan_vs_actual.gif` | A *failing* episode, plan vs reality: red dashes = the policy's 25-step plan at each query; blue = the EE trajectory actually executed. The plan keeps committing to a descent the real arm never completes — the "imagination gap" a world model could score. |
| `loop2/loop2_plan_vs_actual_success.gif` | Same viz on a successful episode (168 mm move, 87 mm lift): plan and reality track, grip channel (green) fires mid-episode. |
| `loop2/loop2_wam_imagination.gif` | Cosmos-H-Surgical-Simulator (`model_ema_bf16.pt`): one real ESD seed frame + a 44-DoF dual-tool `dissection` action rollout → 4 autoregressive chunks ≈ 5 s of imagined endoscopic video. Same imagine-then-plan loop a latent world model enables, shown in the surgical domain it was trained on. Repro: `scripts/esd_simulator_inference.py` with `COSMOS_INTERNAL=1`, empty-string text embeddings (`checkpoints/cr1_empty_string_text_embeddings.pt`), `checkpoints/wan2pt1_tokenizer.pth` — see `Cosmos-H-Surgical-Simulator/outputs/esd_simulator/blog_wam/run2.log`. |
| `viz_plan_vs_actual.py` (sim/scripts/) | regenerates either viz from a `--log-chunks` trace |
| `loop2/loop_comp_fail.gif` / `loop2/loop_comp_success.gif` | **synced camera + telemetry composites** (`viz_synced_panel.py`): left = cam_high rollout, right = live XY plan ghost / z-profile / grip command. The clearest "what is happening" view — watch the red plan keep diving while the blue EE trail stalls. Sources: `loop2/comp_*.mp4` |
| `loop2/loop2_latent_mpc.{gif,mp4}` + `loop2/loop2_latent_mpc_curves.png` + `loop2/loop2_latent_mpc_report.{md,json}` | **actual gradient-MPC on our system**: MLP world model F(z_t,a_t)→Δz (10-d left-arm EE state; actions = commanded EE poses) trained on 108 block episodes, then multi-step rollout fine-tuned against compounding error (one-step xyz err 1.4 mm mean; 25-step open-loop drift 25 mm mean). From ep0 t=100 (hover above block), Adam optimizes a 25-step chunk vs J(A)=½‖z_T−z_goal‖²+0.1·½Σ_k‖z_k−z_goal‖²+λ_p‖A−A₀‖²+λ_s‖ΔA‖²+λ_rΣ‖Δz‖², goal = the ep's own grasp latent (t=227). Init = GT chunk with the first 18 rows replaced by a hover stall + gripper held open — MPC rescues it into a smooth dive: terminal goal error 66.3→1.9 mm, grip 0.28→0.06, max plan step 26 mm/step (inside the ~30–45 mm demonstrated envelope). EE-state-space world model, not an image JEPA — same math. Script: `scripts/loop2_latent_mpc.py`; `loop2/dynamics_mlp.pt` + `loop2/norm_stats.npz` saved for re-rendering. |
| `loop2/loop2_vjepa_probe.{png,mp4,json}` | **real V-JEPA2-AC on our traces**: 1B-param encoder + 0.31B action-conditioned predictor (official `vjepa2-ac-vitg.pt`, in `~/AIMSgeneral/models/`). Fed a real rollout trace (left-arm 7-DoF `[Δxyz, Δeuler, Δgrip]` at DROID's ~4fps cadence): predicted-vs-true future latents cos-sim 0.86→0.76 over 2.1 s; nearest-neighbor retrieval stays on-manifold. Script `scripts/vjepa2ac_probe.py` reusable. |
| `loop2/loop2_vjepa_cem.{png,mp4,json}` | **V-JEPA2-AC closed-loop CEM — honest negative result**: real repo CEM math (L1 latent energy, maxnorm clip, momentum refit), 256 samples × 8 iters × 8-step rollout, goal = encoded frame t=330. CEM converges smoothly (0.439→0.418) — but only 1.4% below the zero-action baseline (0.4238); the planned path drifts ~150 mm off the real descent. Scripts: `scripts/vjepa2ac_cem{,_diag}.py`. |
| `loop2/loop2_vjepa_cem_energy.png` | **Why it failed, quantified**: energy-sensitivity bars — descend / ascend / close-only / the GT action chunk / "away" all score within ΔE≈10⁻³ of each other (GT wins over zero-action by just 0.0006; only grossly random actions are penalized +0.03). The zero-shot DROID latent moves correctly (probe: cos-sim 0.86→0.76, NN-retrieval on-manifold) but the *energy resolution* is flat at 2 s horizon ~100× outside its domain. Caption: a learned critic or fine-tuning is what closes this loop — that's the Loop-4 argument. |

## Loop 3 — receding-horizon control

| file | what it shows |
|---|---|
| `loop3/loop3_k8_vs_k16_same_episode.gif` | THE headline clip. Same seed-42 episode, same block placement. LEFT: execute 8/25 actions per query (K=8) — arm hovers above the block for 700 steps, never commits. RIGHT: execute 16/25 (K=16, 8 mm actuation noise) — grasps and places. Truncating the plan truncates the grasp. |
| `loop3/loop3_k8_fail_seed42_ep2.gif` | K=8 failure alone (2.3 MB) |
| `loop3/loop3_k8_success_seed123_ep4.gif` | A K=8 run that did succeed (seed 123 ep4) — shows the outcome is stochastic, not strictly K-bound |
| `loop3/loop3_k16_noise5_success_seed7_ep2.gif` | Another success, K=16 + 5 mm noise, seed 7 ep2 (6.2 MB) |
| `loop3/loop3_contact_frames.png` | **Long-WAM / temporal-context figure**: 6 frames spanning the grasp (t=260→335). Frames 3–5 are near-identical stills — "fingers touching" vs "1 mm away" is invisible in a single frame; only consecutive frames reveal contact-induced motion. Why history-conditioned backbones exist. |

## Loop 4 — self-improvement across episodes

| file | what it shows |
|---|---|
| `loop4/loop4_old_vs_new_policy.gif` | Same scene, seed 42 ep2. LEFT: `train108_dropout0` (2k steps, 108 demos) hovers and never touches the block. RIGHT: `all_blocks_v2` (10k steps, 134 demos) picks and places. One data+training cycle → 0/5 → working policy. |
| `loop4/loop4_old_policy_fail_ep2.gif` | the old-policy failure alone |
| `loop4/loop4_compat_gate.png` | **RSI compatibility gate with real numbers**: on the open-loop eval suite, `train108` ckpt-2000 is *worse* than the zero-motion baseline (39.0 vs 29.7 mm xyz, 25.3 vs 5.1 deg rot @ h25) → REJECT; `all_blocks_v2` ckpt-10000 beats it (14.1 vs 37.1 mm, 4.8 vs 8.3 deg) → ACCEPT. The retention rule made concrete. |

## Pretrained models — survey + closed-loop sim runs

| file | what it shows |
|---|---|
| `pretrained_models_survey.md` | **Full online survey** (Oct 2026): every downloadable robot foundation model + JEPA/world-model, with repo URLs, sizes, licenses/gating, action spaces, GB10 feasibility, and whether it ships a real planning/MPC mode. Top picks for the policy section: Cosmos-Policy-Predict2-2B (best-of-N model-based search — Loop-2 in one ckpt, already cached), `facebook/jepa-wms` (pretrained JEPA-WMs + CEM that beat V-JEPA2-AC on DROID planning), MolmoAct2-DROID (5B, spatial-trace plans), Cosmos3-Edge-Policy-DROID (4B, cached), DreamZero-DROID (14B — the GR00T-N2 WAM architecture, downloadable today, CC-BY-NC). Not-downloadable flagged honestly: GR00T N2/N3, π0.6 (only `exla-ai/openpie-0.6` reimpl), Genie 3, Helix, GR-3. |
| `policies/policy_pi0_probe.json` | **π0 ran on GB10**: `lerobot/pi0` PyTorch port (PaliGemma-3B + 300M flow-matching expert, 14 GB fp32, Apache-2.0) — real frames from our rollout video → finite 50-step × 6-DoF action chunk in 2.55 s, no JAX. Caveat: this port's action space is a 6-d tabletop-arm layout, NOT our 20-d dual-UR5 EE space and NOT upstream π0's 14-d dual-arm deltas — "generalist VLA inference runs locally," not a drop-in policy. |
| `policies/policy_cosmos_droid.png` + `.json/.npy` | **Cosmos3-Edge-Policy-DROID ran zero-shot on our frame** (4B MoT, OpenMDW-1.1): our cam_high composed into the DROID 540×640 concat view → coherent 32-step × 8-DoF Franka joint-position chunk @ 15 Hz (205 s inference). Franka-FK EE path is physically plausible — the foundation policy *parses* our surgical scene — but lives in another embodiment's frame. The embodiment gap, measured. (Agent patched a stale `model.safetensors.index.json` + 4 missing pos-embed buffers in the released ckpt to get here.) |

### Closed-loop sim runs — pretrained foundation models driving the cell

Each ran through `rollout_policy.py` untouched: adapter server `sim/scripts/serve_<model>.py` → real Isaac physics → video + trace → GIF + `policies/policy_<model>_sim_report.json`. Same seed-7 scene, same task string, same metrics as the GR00T baselines.

| model | port | adapter | episode outcome (seed 7) |
|---|---|---|---|
| `lerobot/pi0` (3.3B, Apache-2.0) | 8770 | normalized 6-d → Δxyz×gain (auto-calibrated 0.16) + grip affine, hold rot, proprio-blind zeros(6) | 2 eps (seeds 7, 123), 400 steps each, ~0.61 s/query — arm wandered violently (EE path ~4–5 m, min EE→block 628–695 mm), block untouched. `policies/policy_pi0_sim_ep00{0,1}_seed*.gif` |
| Cosmos3-Edge-Policy-DROID (4B, OpenMDW-1.1) | 8771 | Franka FK → EE-delta retarget (≤25 mm/step), hold rot | 200 steps, 9 queries @ 5.5–37 s — drove EE coherently to the −x workspace boundary; block untouched (0 mm, 150 mm, 44 clamps). The retarget *mechanics* worked; the direction was wrong — the embodiment gap is frames, not just action shape. `policies/policy_cosmos_droid_sim_ep000.gif` |
| V-JEPA2-AC CEM (1.3B, CC-BY-NC) | 8772 | per-query CEM replan (48 samples × 6 iters, ~72 s) → 7-DoF EE deltas, fixed goal latent = t=330 grasp frame | 120 steps, 30 genuine replans — EE wandered *away* from the block (0 mm); mean E_best 0.468 vs E_zero 0.482 → real but shallow gradient (ΔE≈0.013). **Legible detail**: goal frame's closed gripper → planner drove grip→1.0 all episode; position channel carried no usable gradient. Latent MPC closed the loop; the flat energy couldn't steer it. `policies/policy_vjepa_cem_sim_ep0_seed7.gif` |
| `facebook/jepa-wms` (DINOv3-L 303M + AdaLN predictor 229M, CC-BY-NC) | 8773 | repo-faithful `CEMPlanner` (L2-on-final-latent, horizon 3) → 7-DoF EE deltas, goal = t=330 grasp frame | 120 steps, 40 queries @ ~100 s — **sharper landscape**: mean E_best 5.5% below zero-action (vs V-JEPA's ~1.5–2.7%), plans include rotation channels — but same physical outcome: min EE→block 791 mm, block untouched. Domain-matched JEPA-WM helps the energy, not enough at this domain gap. `policies/policy_jepawms_sim_ep000_seed7.gif` |
| **GR00T-H ckpt-10000** (3B, fine-tuned) | 8765 | native | **baseline: ~2–3/15** — the only policy that actually grasps the block |
| SmolVLA base (450M) | 8766 | native | flail, knocked block sideways (`loop1/loop1_smolvla_base_flail.gif`) |

**The comparison story**: every zero-shot foundation model produces *well-formed, coherent* action streams — the motion is smooth, the gripper channels fire — but not one moves the block. The only policy that succeeds is the one fine-tuned on 134 demonstrations of *this* cell. That is the blog's Loop-4 thesis demonstrated on the GPU, not asserted in prose.

## Rollout stats (context for captions)

- Closed-loop success so far: ~2–3/15 episodes across seeds/horizons — a real but fragile policy. Failures are almost all "hover above the block, gripper never fires"; a couple of grasp attempts drop the block.
- Same config re-run can flip success↔failure (Isaac physics + small randomization) — the distribution is knife-edge, which is exactly the Loop-1/Loop-3 argument.
- Open-loop (subagent eval): new ckpt beats zero-displacement baseline everywhere (14 mm vs 37 mm xyz err @ step 25); old ckpt loses to it on position and is ~70× worse on rotation.

## Repro commands

```bash
# server
cd sim/gr00t_h && source /home/aimsgroupuol/AIMSgeneral/venv-gr00th-extra/env.sh
python serve_gr00t_h.py --ckpt ../outputs/gr00t_h_finetune/all_blocks_v2_20261012_012734/checkpoint-10000 --port 8765

# rollout with plan logging + actuation noise + video
venv-isaacsim/bin/python -I sim/scripts/rollout_policy.py --policy groot \
  --episodes 3 --seed 7 --exec-steps 16 --noise-mm 5 --video --log-chunks \
  --task "pick up the block and place it on the target" \
  --out sim/outputs/rollouts/<name>

# plan-vs-reality viz
venv-isaacsim/bin/python sim/scripts/viz_plan_vs_actual.py \
  --trace sim/outputs/rollouts/<name>/episode_XXX_seedN_trace.npz \
  --out loop2/loop2_plan_vs_actual.mp4 --fps 12
```
