# Closed-loop policy comparison — dual-UR5 block pick-and-place (Isaac Sim)

Every row drove the **same simulated cell** through the untouched `rollout_policy.py`
HTTP contract (`POST /act` → absolute world-frame 20-d EE-pose chunks @ 15 Hz,
`[L_xyz3, L_rot6d6, L_grip1, R_xyz3, R_rot6d6, R_grip1]`), same seed-7 scene, same task
string `"pick up the block and place it on the target"`, same success metrics
(block moved ≥10 cm + lifted ≥5 cm + placed within 5 cm of target).

Task: pick up a 4×5.5×6 cm block and place it on a target ~15 cm away.

| Model | Params | Trained for | Adapter to our cell | Queries (time/q) | Block moved | Max lift | Place err | Success | Asset |
|---|---|---|---|---|---|---|---|---|---|
| **GR00T-H ckpt-10000** (`all_blocks_v2`, fine-tuned on 134 demos) | 3B | surgical EE-pose pick tasks (601 h Open-H + our demos) | native | 25-step chunks @ ~170 ms | **141–168 mm** | **81–87 mm** | **20–63 mm** | **~2–3/15** | `loop3/loop3_k8_vs_k16_same_episode.gif` |
| GR00T-H ckpt-2000 (`train108`, 108 demos) | 3B | same | native | same | 0 mm (hover) | 0 mm | ~150 mm | 0/5 | `loop4/loop4_old_policy_fail_ep2.gif` |
| `lerobot/smolvla_base` | 450M | SO-100 community demos | native server, joint→EE adapter | ~1 s | knocked over (258 mm displaced, never lifted) | 0 mm | — | 0/1 | `loop1/loop1_smolvla_base_flail.gif` |
| `lerobot/pi0` (PyTorch port of PI π0) | 3.3B | ~10k h OXE + PI robots | normalized 6-d joint-ish → Δxyz×0.16 + grip affine, hold rot, proprio-blind zeros(6) | 25/ep @ ~0.61 s | **0 mm** (min EE→block 628–695 mm, EE travel 3.8–5.0 m) | 0 mm | 150 mm | 0/2 | `policies/policy_pi0_sim_ep00{0,1}_seed*.gif` |
| Cosmos3-Edge-Policy-DROID | 4B | DROID Franka | Franka FK → EE-delta retarget ≤25 mm/step, hold rot, grip affine | 9 @ 5.5–37 s | **0 mm** (coherent drive to −x boundary, 44 clamps) | 0 mm | 150 mm | 0/1 | `policies/policy_cosmos_droid_sim_ep000.gif` |
| V-JEPA2-AC + CEM (latent MPC) | 1.3B | DROID latent dynamics | per-query CEM (48×6) → 7-DoF EE deltas, goal = t=330 grasp latent | 30 @ ~72 s | **0 mm** (mushy wander, grip→1.0 all ep from goal frame) | 0 mm | 150 mm | 0/1 | `policies/policy_vjepa_cem_sim_ep0_seed7.gif` |
| `facebook/jepa-wms` EncPredWM | 0.53B | DROID JEPA-WM | repo-faithful CEMPlanner (L2 latent, H=3) → 7-DoF EE deltas | 40 @ ~100 s | **0 mm** (min EE→block 791 mm) | 0 mm | 150 mm | 0/1 | `policies/policy_jepawms_sim_ep000_seed7.gif` |

## Reading the table

- **Well-formed ≠ useful**: every zero-shot model emitted smooth, plausible action
  streams — π0 moved metres of EE path, Cosmos-DROID drove a consistent direction,
  V-JEPA-CEM's grip channel tracked the goal frame's closed gripper — yet not one
  touched the block. The bottleneck is not motor syntax; it's *task semantics
  grounded in this cell*.
- **The embodiment gap is layered**: (a) action-space shape (6-d joints vs 20-d EE),
  (b) coordinate frames (Franka base ≠ our world — Cosmos-DROID's coherent-but-wrong
  direction), (c) proprioception (π0 got zeros(6) — it never knew where the arm was),
  (d) goal specification (V-JEPA-CEM needed a *remembered frame* as its goal).
- **Latent MPC ran — and failed instructively**: two real latent planners closed the
  loop. V-JEPA2-AC replanned 30 times (energy ≈1.4–2.7% below zero-action — nearly
  flat); jepa-wms' domain-matched DINOv3-WM replanned 40 times with a **sharper**
  landscape (mean 5.5% below zero-action, rotation-aware plans) — and still drove
  away from the block. Better energy helps; at this domain gap it still can't steer.
  Fine-tuning or a learned critic is what closes this loop (Loop-4 argument).
- **The only success is the fine-tuned policy** — which is exactly the blog's thesis:
  foundation models provide the substrate; the data+train cycle (Loop 4) is what
  makes it *work here*.

## Provenance & licenses

| Model | Source | License | Local artifacts |
|---|---|---|---|
| GR00T-H-N1.7 | `nvidia/GR00T-H-N1.7` | NVIDIA Open Model | `sim/outputs/gr00t_h_finetune/all_blocks_v2_*/checkpoint-10000`, server :8765 |
| SmolVLA | `lerobot/smolvla_base` | Apache-2.0 | server :8766 (`venv-smolvla`) |
| π0 | `lerobot/pi0` (safetensors 14 GB) | Apache-2.0 | `sim/scripts/serve_pi0.py` :8770 |
| Cosmos3-Edge-Policy-DROID | `nvidia/Cosmos3-Edge-Policy-DROID` | OpenMDW-1.1 | `sim/scripts/serve_cosmos_droid.py` :8771 (patched snapshot in `Gyanateet/cosmos_policy_spark/models/`) |
| V-JEPA2-AC | `dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt` | CC-BY-NC-4.0 | `sim/scripts/serve_vjepa_cem.py` :8772 |
| jepa-wms | `facebook/jepa-wms` + `github.com/facebookresearch/jepa-wms` | CC-BY-NC-4.0 | `sim/scripts/serve_jepawms.py` :8773 |

## Reproduce

```bash
# any model: start its adapter server, then run the SAME rollout driver
python -I sim/scripts/serve_<model>.py --port <port>          # in its env (see report json)

venv-isaacsim/bin/python -I sim/scripts/rollout_policy.py \
  --policy <label> --policy-url http://127.0.0.1:<port> \
  --episodes 1 --seed 7 --exec-steps 16 --max-steps 400 --video --log-chunks \
  --task "pick up the block and place it on the target" \
  --out sim/outputs/rollouts/<name>
```
