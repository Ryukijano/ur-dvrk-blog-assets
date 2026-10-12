#!/usr/bin/env python
"""V-JEPA 2-AC closed-loop step: CEM action planning toward a goal latent.

From UR_dVRK episode_002_seed7 context ending at t=260 (pre-grasp descent),
plan an 8-step EE-delta action sequence whose predicted latent approaches the
goal latent of frame t=330 (block grasped & lifted). Uses the repo's CEM
algorithm (notebooks/utils/mpc_utils.py: l1 energy, maxnorm clip, momentum
mean/std update) instrumented to log per-iteration candidates + energies.

Outputs: loop2_vjepa_cem.{png,mp4,json}
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

REPO = "/home/aimsgroupuol/AIMSgeneral/models/vjepa2-repo"
CKPT = "/home/aimsgroupuol/AIMSgeneral/models/vjepa2/vjepa2-ac-vitg.pt"
EP = "/home/aimsgroupuol/AIMSgeneral/UR_dVRK/sim/outputs/rollouts/groot_blog_fish"
VID = f"{EP}/videos/episode_002_seed7.mp4"
TRACE = f"{EP}/episode_002_seed7_trace.npz"
OUT = "/home/aimsgroupuol/AIMSgeneral/UR_dVRK/blog_assets/loop2_vjepa_cem"

CROP, PATCH = 256, 16
TOK = (CROP // PATCH) ** 2          # 256 tokens/frame
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
DEV = "cuda"
DT = torch.bfloat16

# ---- planning setup ----
STR = 4                              # predictor step = 4 real frames (~0.27s)
T_PLAN = 260                         # last context frame = planning origin
T_GOAL = 330                         # goal frame (post-grasp lift)
NCTX = 4                             # context frames: 248,252,256,260
ROLLOUT = 8                          # planned steps -> covers ~t260..292
SAMPLES = 256
TOPK = 24
CEM_STEPS = 8
MAXNORM = 0.05                       # repo default 5cm/step clip on dxyz
M_MEAN = 0.15                        # repo momentum_mean
M_STD = 0.15                         # repo momentum_std
M_MEAN_G = 0.15
M_STD_G = 0.15

sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/notebooks")
from src.hub.backbones import _make_vjepa2_ac_model  # noqa: E402
from utils.mpc_utils import compute_new_pose        # noqa: E402


def rot6d_to_euler(r6):
    a1, a2 = r6[:, 0:3], r6[:, 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True).clip(1e-9)
    b2 = a2 - (b1 * a2).sum(1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True).clip(1e-9)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=2)
    return Rotation.from_matrix(R).as_euler("xyz", degrees=False)


def poses_to_diff7(s, e):
    dxyz = e[:3] - s[:3]
    Rs = Rotation.from_euler("xyz", s[3:6]).as_matrix()
    Re = Rotation.from_euler("xyz", e[3:6]).as_matrix()
    dth = Rotation.from_matrix(Re @ Rs.T).as_euler("xyz", degrees=False)
    return np.concatenate([dxyz, dth, e[6:7] - s[6:7]]).astype(np.float32)


def load_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def preprocess(frames_u8):
    out = []
    for f in frames_u8:
        h, w = f.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        f = cv2.resize(f[y0:y0 + s, x0:x0 + s], (CROP, CROP), interpolation=cv2.INTER_LINEAR)
        out.append(torch.from_numpy(f).float().permute(2, 0, 1) / 255.0)
    x = torch.stack(out).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    return (x - MEAN) / STD


@torch.no_grad()
def encode_frames(encoder, frames_u8):
    clips = preprocess(frames_u8).to(DEV, DT)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = encoder(clips)
    z = F.layer_norm(z.float(), (z.size(-1),))
    return z.cpu()                                  # (N,256,1408)


@torch.no_grad()
def predictor_next(pred, z_hist, a_hist, s_hist):
    """z_hist [S,T,N,D] bf16 gpu, a/s [S,T,7] -> next-frame latents [S,1,N,D] fp32 cpu->gpu."""
    S, T, N, D = z_hist.shape
    z = z_hist.reshape(S, T * N, D)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = pred(z, a_hist, s_hist)
    out = out[:, -TOK:].float()
    out = F.layer_norm(out, (out.size(-1),))
    return out.unsqueeze(1)                          # [S,1,N,D]


def cem_plan(pred, z_ctx, s_ctx, a_ctx, z_goal, rollout, samples, topk, cem_steps,
             maxnorm, log_samples_iters=None):
    """Instrumented replica of notebooks/utils/mpc_utils.cem.
    z_ctx [1,C,N,D] gpu bf16; s_ctx/a_ctx [1,C,7] gpu bf16 (a_ctx[-1] slot unused:
    last context-frame action slot is replaced by the sampled candidate);
    z_goal [1,1,N,D].  Returns (mean_traj [R,7], log dict).
    """
    if log_samples_iters is None:
        log_samples_iters = set(range(cem_steps))
    S = samples
    frame0 = z_ctx.repeat(S, 1, 1, 1)                # [S,C,N,D]
    goal = z_goal.repeat(S, 1, 1, 1).float()         # [S,1,N,D]
    pose0 = s_ctx.repeat(S, 1, 1)                    # [S,C,7]
    a_ctx_r = a_ctx[:, :-1].repeat(S, 1, 1)          # [S,C-1,7] real ctx actions

    mean = torch.zeros((rollout, 4), device=DEV, dtype=torch.float32)
    std = torch.cat([torch.ones((rollout, 3), device=DEV) * maxnorm,
                     torch.ones((rollout, 1), device=DEV)], dim=-1)

    log = {"iter_best_E": [], "iter_elite_mean_E": [], "iter_mean_E_all": [],
           "iter_mean": [], "iter_std": [], "cand_actions": {}, "cand_E": {},
           "elite_trajs": {}}

    for it in range(cem_steps):
        action_traj, frame_traj, pose_traj = None, frame0, pose0
        for h in range(rollout):
            a4 = torch.randn(S, 4, device=DEV) * std[h] + mean[h]
            a4[:, :3] = torch.clip(a4[:, :3], -maxnorm, maxnorm)
            a4[:, 3:4] = torch.clip(a4[:, 3:4], -0.75, 0.75)
            a7 = torch.cat([a4[:, :3], torch.zeros(S, 3, device=DEV), a4[:, 3:4]], -1)[:, None]
            action_traj = a7 if action_traj is None else torch.cat([action_traj, a7], 1)
            # predictor needs one (a,s) per frame-block: real ctx actions for
            # first C-1 frames, then the sampled sequence for the rest
            full_a = torch.cat([a_ctx_r, action_traj], 1)          # [S,C-1+h+1,7]
            nxt = predictor_next(pred, frame_traj, full_a.to(DT), pose_traj.to(DT))
            npose = compute_new_pose(pose_traj[:, -1:].float(), action_traj[:, -1:].float()).to(DEV)
            frame_traj = torch.cat([frame_traj, nxt.to(DT)], 1)
            pose_traj = torch.cat([pose_traj, npose.to(DT)], 1)

        # repo l1 energy: mean |final_pred - goal| over N*D
        E = torch.mean(torch.abs(frame_traj[:, -1].float() - goal[:, 0]), dim=(1, 2))
        idx = E.topk(topk, largest=False).indices
        sel = action_traj[idx].float()                             # [K,R,7]->[K,R,4]
        sel4 = torch.cat([sel[..., :3], sel[..., -1:]], -1)
        m_sel, s_sel = sel4.mean(0), sel4.std(0)
        mean = torch.cat([m_sel[..., :3] * (1 - M_MEAN) + mean[..., :3] * M_MEAN,
                          m_sel[..., -1:] * (1 - M_MEAN_G) + mean[..., -1:] * M_MEAN_G], -1)
        std = torch.cat([s_sel[..., :3] * (1 - M_STD) + std[..., :3] * M_STD,
                         s_sel[..., -1:] * (1 - M_STD_G) + std[..., -1:] * M_STD_G], -1)

        log["iter_best_E"].append(float(E[idx[0]]))
        log["iter_elite_mean_E"].append(float(E[idx].mean()))
        log["iter_mean_E_all"].append(float(E.mean()))
        log["iter_mean"].append(mean.cpu().tolist())
        log["iter_std"].append(std.cpu().tolist())
        if it in log_samples_iters:
            log["cand_actions"][it] = action_traj[:, 0].float().cpu().tolist()
            log["cand_E"][it] = E.cpu().tolist()
            # store top-8 elite full trajectories + integrated poses
            elite = idx[:8]
            log["elite_trajs"][it] = {
                "a": action_traj[elite].float().cpu().tolist(),
                "p": pose_traj[elite].float().cpu().tolist(),
            }
        print(f"    cem it{it}: bestE={log['iter_best_E'][-1]:.4f} "
              f"eliteE={log['iter_elite_mean_E'][-1]:.4f} allE={log['iter_mean_E_all'][-1]:.4f} "
              f"mean_sum={mean.sum(0).cpu().numpy().round(3)}", flush=True)
        del frame_traj, pose_traj, action_traj
        torch.cuda.empty_cache()

    # repo-style final action = mean, euler zeros, grip rounded at 0.25
    g = mean[:, -1:].clone()
    g = torch.where(g.abs() < 0.25, torch.zeros_like(g), g)
    final = torch.cat([mean[:, :3], torch.zeros(rollout, 3, device=DEV), g], -1)
    return final.cpu(), log


def integrate_pose(p0, a7):
    """p0 [7] np, a7 [R,7] np -> [R+1,7] np via repo compute_new_pose."""
    pts = [p0.astype(np.float64)]
    p = torch.from_numpy(p0.astype(np.float64))[None, None]
    for h in range(len(a7)):
        a = torch.from_numpy(a7[h].astype(np.float64))[None, None]
        p = compute_new_pose(p, a)
        pts.append(p[0, 0].numpy())
    return np.stack(pts)


def main():
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    print("[1/6] models ...", flush=True)
    enc, pred = _make_vjepa2_ac_model(pretrained=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    clean = lambda d: {k.replace("module.", "").replace("backbone.", ""): v for k, v in d.items()}
    enc.load_state_dict(clean(sd["encoder"]), strict=False)
    pred.load_state_dict(clean(sd["predictor"]), strict=True)
    del sd
    enc = enc.to(DEV).to(DT).eval(); pred = pred.to(DEV).to(DT).eval()
    print(f"    on GPU {time.time()-t0:.0f}s", flush=True)

    print("[2/6] video + trace ...", flush=True)
    frames = load_video_frames(VID)
    tr = np.load(TRACE)
    st = tr["state"]
    L = np.concatenate([st[:, 0:3], rot6d_to_euler(st[:, 3:9]), st[:, 9:10]], 1).astype(np.float32)
    acts7 = np.stack([poses_to_diff7(L[t], L[min(t + STR, len(L) - 1)]) for t in range(len(L))])

    ctx_idx = np.arange(T_PLAN - (NCTX - 1) * STR, T_PLAN + 1, STR)   # [248,252,256,260]
    print(f"    ctx frames {ctx_idx.tolist()}, goal f{T_GOAL}, STR={STR}", flush=True)

    print("[3/6] encode ctx + goal ...", flush=True)
    z_ctx = encode_frames(enc, frames[ctx_idx]).view(1, NCTX, TOK, -1).to(DEV, DT)
    z_goal = encode_frames(enc, frames[T_GOAL:T_GOAL + 1]).view(1, 1, TOK, -1).to(DEV, DT)
    s_ctx = torch.tensor(L[ctx_idx]).unsqueeze(0).to(DEV)             # [1,C,7]
    a_ctx = torch.tensor(acts7[ctx_idx]).unsqueeze(0).to(DEV)         # [1,C,7]
    print(f"    z_ctx {tuple(z_ctx.shape)} z_goal {tuple(z_goal.shape)}", flush=True)

    print("[4/6] CEM planning ...", flush=True)
    plan, log = cem_plan(pred, z_ctx, s_ctx, a_ctx, z_goal, ROLLOUT, SAMPLES, TOPK,
                         CEM_STEPS, MAXNORM)
    plan_np = plan.numpy()                                           # [R,7]
    print("    planned traj (dxyz, dgrip):")
    for h in range(ROLLOUT):
        print(f"      step{h}: dxyz=({plan_np[h,0]:+.4f},{plan_np[h,1]:+.4f},{plan_np[h,2]:+.4f}) "
              f"dgrip={plan_np[h,6]:+.3f}", flush=True)

    print("[5/6] integrate + compare + baseline ...", flush=True)
    p0 = L[T_PLAN].astype(np.float64)
    plan_path = integrate_pose(p0, plan_np)                          # [R+1,7]
    gt_path = L[T_PLAN: T_PLAN + ROLLOUT * STR + 1: STR].astype(np.float64)  # [R+1,7]
    gt_goal = L[T_GOAL].astype(np.float64)
    gt_acts = acts7[T_PLAN: T_PLAN + (ROLLOUT - 1) * STR + 1: STR]     # [R,7]

    err_xyz = np.linalg.norm(plan_path[1:, :3] - gt_path[1:, :3], axis=1) * 1000  # mm
    err_fin = np.linalg.norm(plan_path[-1, :3] - gt_path[-1, :3]) * 1000
    # planned pose vs *goal* pose
    err_goal = np.linalg.norm(plan_path[-1, :3] - gt_goal[:3]) * 1000

    # zero-action baseline energy
    S = 64
    frame_traj = z_ctx.repeat(S, 1, 1, 1)
    pose_traj = s_ctx.repeat(S, 1, 1)
    a_ctx_r = a_ctx[:, :-1].repeat(S, 1, 1)
    with torch.no_grad():
        zero_traj = torch.zeros(S, ROLLOUT, 7, device=DEV)
        for h in range(ROLLOUT):
            full_a = torch.cat([a_ctx_r, zero_traj[:, :h + 1]], 1)
            nxt = predictor_next(pred, frame_traj, full_a.to(DT), pose_traj.to(DT))
            npose = compute_new_pose(pose_traj[:, -1:].float(), zero_traj[:, h:h + 1].float()).to(DEV)
            frame_traj = torch.cat([frame_traj, nxt.to(DT)], 1)
            pose_traj = torch.cat([pose_traj, npose.to(DT)], 1)
        E_zero = float(torch.mean(torch.abs(frame_traj[:, -1].float() - z_goal.float()[:, 0])))
    print(f"    zero-action rollout energy = {E_zero:.4f}", flush=True)

    # planned-traj energy (re-roll the mean action)
    with torch.no_grad():
        S2 = 4
        frame_traj = z_ctx.repeat(S2, 1, 1, 1)
        pose_traj = s_ctx.repeat(S2, 1, 1)
        a_ctx_r = a_ctx[:, :-1].repeat(S2, 1, 1)
        pm = torch.tensor(plan_np, device=DEV, dtype=torch.float32)[None].repeat(S2, 1, 1)
        for h in range(ROLLOUT):
            full_a = torch.cat([a_ctx_r, pm[:, :h + 1]], 1)
            nxt = predictor_next(pred, frame_traj, full_a.to(DT), pose_traj.to(DT))
            npose = compute_new_pose(pose_traj[:, -1:].float(), pm[:, h:h + 1]).to(DEV)
            frame_traj = torch.cat([frame_traj, nxt.to(DT)], 1)
            pose_traj = torch.cat([pose_traj, npose.to(DT)], 1)
        E_plan = float(torch.mean(torch.abs(frame_traj[:, -1].float() - z_goal.float()[:, 0])))
    print(f"    planned-traj rollout energy = {E_plan:.4f}", flush=True)

    print("[6/6] figures ...", flush=True)
    make_png(frames, ctx_idx, plan_path, gt_path, gt_goal, plan_np, gt_acts, log,
             E_zero, E_plan)
    make_mp4(frames, ctx_idx, log, p0, gt_path[:, 2])
    report = {
        "planner": "repo CEM math (notebooks/utils/mpc_utils.py l1 energy, maxnorm clip, "
                   "momentum mean/std), instrumented replica w/ logging; multi-frame ctx C=4",
        "setup": {"STR": STR, "T_PLAN": T_PLAN, "T_GOAL": T_GOAL, "NCTX": NCTX,
                  "ROLLOUT": ROLLOUT, "SAMPLES": SAMPLES, "TOPK": TOPK,
                  "CEM_STEPS": CEM_STEPS, "MAXNORM": MAXNORM},
        "ctx_frames": ctx_idx.tolist(),
        "planned_actions_7dof": plan_np.tolist(),
        "planned_ee_path": plan_path.tolist(),
        "gt_ee_path": gt_path.tolist(),
        "goal_pose_t330": gt_goal.tolist(),
        "gt_actions_7dof": gt_acts.tolist(),
        "per_step_xyz_err_mm": err_xyz.tolist(),
        "final_step_xyz_err_mm": float(err_fin),
        "plan_end_vs_goal_xyz_err_mm": float(err_goal),
        "energy": {"zero_action_baseline": E_zero, "planned_traj": E_plan,
                   "best_per_iter": log["iter_best_E"],
                   "elite_mean_per_iter": log["iter_elite_mean_E"],
                   "pop_mean_per_iter": log["iter_mean_E_all"]},
        "gt_segment": "t260-275 descend z~0.98->0.962; t275-300 grip 0.03->0.30 close; "
                      "t310-340 lift z->1.05",
    }
    with open(f"{OUT}.json", "w") as f:
        json.dump(report, f, indent=2)
    print("saved", f"{OUT}.json", flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


def _thumb(fr, th=170):
    h, w = fr.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(fr[y:y + s, x:x + s], (th, th))


def make_png(frames, ctx_idx, plan_path, gt_path, gt_goal, plan_np, gt_acts, log,
             E_zero, E_plan):
    fig = plt.figure(figsize=(19, 5.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.25, 1.0, 1.0, 1.0, 1.0], wspace=0.3)

    # A: frames
    axA = fig.add_subplot(gs[0]); axA.axis("off")
    th = 170
    mont = np.concatenate([_thumb(frames[T_PLAN], th), np.ones((th, 6, 3), np.uint8) * 30,
                           _thumb(frames[T_GOAL], th)], axis=1)
    axA.imshow(mont)
    axA.text(4, 14, f"context end t={T_PLAN}", color="cyan", fontsize=8, weight="bold")
    axA.text(th + 10, 14, f"goal t={T_GOAL} (block lifted)", color="lime", fontsize=8, weight="bold")
    axA.set_title("CEM planning: context -> goal latent", fontsize=9)

    # B: EE path x-y and z
    axB = fig.add_subplot(gs[1])
    axB.plot(gt_path[:, 0] * 1000, gt_path[:, 1] * 1000, "b-o", ms=4, lw=1.4, label="GT EE path")
    axB.plot(plan_path[:, 0] * 1000, plan_path[:, 1] * 1000, "r--^", ms=5, lw=1.4, label="CEM planned")
    axB.scatter([gt_goal[0] * 1000], [gt_goal[1] * 1000], marker="*", s=160, c="g", label="goal pose t330")
    axB.set_xlabel("x (mm)"); axB.set_ylabel("y (mm)"); axB.legend(fontsize=7)
    axB.set_title("EE path top-down (x,y)", fontsize=9)
    axB.set_aspect("equal", adjustable="datalim")

    # C: z & grip vs step
    axC = fig.add_subplot(gs[2])
    steps = np.arange(len(gt_path))
    axC.plot(steps, gt_path[:, 2], "b-o", ms=4, label="GT z")
    axC.plot(steps, plan_path[:, 2], "r--^", ms=5, label="planned z")
    axC.axhline(gt_goal[2], color="g", ls=":", lw=1, label="goal z")
    axC2 = axC.twinx()
    axC2.plot(steps, gt_path[:, 6], "b-", lw=1, alpha=0.5, label="GT grip")
    axC2.plot(steps, plan_path[:, 6], "r--", lw=1.4, alpha=0.8, label="planned grip")
    axC.set_xlabel(f"plan step (={STR} frames each)"); axC.set_ylabel("z (m)", color="b")
    axC2.set_ylabel("grip", color="r")
    lns = axC.get_lines() + axC2.get_lines()
    axC.legend(lns, [l.get_label() for l in lns], fontsize=6, loc="best")
    axC.set_title("height z + grip vs plan step", fontsize=9)

    # D: energy vs iter
    axD = fig.add_subplot(gs[3])
    it = np.arange(len(log["iter_best_E"]))
    axD.plot(it, log["iter_mean_E_all"], "o-", c="grey", label="population mean E")
    axD.plot(it, log["iter_elite_mean_E"], "s-", c="orange", label=f"top-{TOPK} elite mean E")
    axD.plot(it, log["iter_best_E"], "^-", c="r", label="best candidate E")
    axD.axhline(E_zero, color="k", ls="--", lw=1, label=f"zero-action rollout E={E_zero:.3f}")
    axD.axhline(E_plan, color="m", ls=":", lw=1.4, label=f"planned-traj E={E_plan:.3f}")
    axD.set_xlabel("CEM iteration"); axD.set_ylabel("E = mean|z_pred - z_goal| (L1)")
    axD.legend(fontsize=6); axD.set_title("Energy landscape vs CEM iteration", fontsize=9)

    # E: candidate scatter dx/dz iter0 vs last
    axE = fig.add_subplot(gs[4])
    keys = sorted(log["cand_actions"].keys())
    for k, col, mk in [(keys[0], "lightgrey", "o"), (keys[-1], "r", "^")]:
        A = np.array(log["cand_actions"][k]); E = np.array(log["cand_E"][k])
        sc = axE.scatter(A[:, 0] * 1000, A[:, 2] * 1000, c=E, cmap="viridis_r",
                         s=10 if k == keys[0] else 18, marker=mk, alpha=0.7,
                         label=f"iter {k} samples (n={len(A)})")
    axE.scatter([gt_acts[0, 0] * 1000], [gt_acts[0, 2] * 1000], marker="*", s=200,
                c="cyan", edgecolors="k", label="GT first action")
    m_last = np.array(log["iter_mean"][-1])
    axE.scatter([m_last[0, 0] * 1000], [m_last[0, 2] * 1000], marker="P", s=160,
                c="r", edgecolors="k", label="final CEM mean (step0)")
    axE.axvline(0, color="k", lw=0.5); axE.axhline(0, color="k", lw=0.5)
    axE.set_xlabel("candidate Δx (mm/step)"); axE.set_ylabel("candidate Δz (mm/step)")
    axE.legend(fontsize=6); axE.set_title("Sampled action population (color=energy)", fontsize=9)

    fig.suptitle("V-JEPA 2-AC zero-shot CEM planning on UR_dVRK ep002_seed7 — "
                 f"plan from t={T_PLAN} toward goal-latent t={T_GOAL} "
                 f"({ROLLOUT} steps x {STR} frames, {SAMPLES} samples, {CEM_STEPS} iters)",
                 fontsize=11)
    fig.savefig(f"{OUT}.png", dpi=150, bbox_inches="tight")
    print("saved", f"{OUT}.png", flush=True)


def make_mp4(frames, ctx_idx, log, p0, gt_z):
    """Per-CEM-iter frame: candidate dx/dz scatter + elite integrated z-paths."""
    keys = sorted(log["cand_actions"].keys())
    out = f"{OUT}.mp4"
    vw = None
    iters = list(range(len(log["iter_best_E"])))
    for it in iters:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
        fig.subplots_adjust(wspace=0.3, left=0.07, right=0.98, top=0.82, bottom=0.12)
        # left: candidate scatter (only logged iters get full cloud; else mean/std)
        if it in log["cand_actions"]:
            A = np.array(log["cand_actions"][it]); E = np.array(log["cand_E"][it])
            sc = axL.scatter(A[:, 0] * 1000, A[:, 2] * 1000, c=E, cmap="viridis_r",
                             s=14, alpha=0.8)
            fig.colorbar(sc, ax=axL, label="energy (L1)")
            el = log["elite_trajs"][it]
            ea = np.array(el["a"])[:, 0]
            axL.scatter(ea[:, 0] * 1000, ea[:, 2] * 1000, facecolors="none",
                        edgecolors="red", s=60, lw=1.2, label="elites")
            axL.legend(fontsize=7)
        m = np.array(log["iter_mean"][it]); s = np.array(log["iter_std"][it])
        axL.arrow(0, 0, m[0, 0] * 1000, m[0, 2] * 1000, head_width=3, head_length=4,
                  fc="red", ec="red", lw=2)
        axL.set_xlim(-MAXNORM * 1100, MAXNORM * 1100)
        axL.set_ylim(-MAXNORM * 1100, MAXNORM * 1100)
        axL.axvline(0, color="k", lw=0.5); axL.axhline(0, color="k", lw=0.5)
        axL.set_xlabel("candidate Δx (mm/step)"); axL.set_ylabel("candidate Δz (mm/step)")
        axL.set_title(f"CEM iter {it}: {SAMPLES} sampled step-0 actions\n"
                      f"bestE={log['iter_best_E'][it]:.4f}  eliteE={log['iter_elite_mean_E'][it]:.4f}",
                      fontsize=10)
        # right: elite integrated paths (z + grip) if logged
        if it in log["elite_trajs"]:
            el = log["elite_trajs"][it]
            for a_one in np.array(el["a"]):
                path = integrate_pose(p0, np.asarray(a_one))
                axR.plot(path[:, 2], "o-", ms=3, lw=1, alpha=0.6)
            axR.plot(np.arange(len(gt_z)), gt_z, "k--*", ms=8, lw=2, label="GT z (real rollout)")
            axR.legend(fontsize=7)
        axR.set_xlabel("plan step"); axR.set_ylabel("z (m)")
        axR.set_title("elite candidate z-trajectories", fontsize=10)
        fig.suptitle(f"V-JEPA2-AC CEM iter {it} — sampled actions -> predicted-latent energy",
                     fontsize=11)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        if vw is None:
            vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 1.5,
                                 (img.shape[1], img.shape[0]))
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        plt.close(fig)
    if vw is not None:
        vw.release()
        print("saved", out, flush=True)


if __name__ == "__main__":
    main()
