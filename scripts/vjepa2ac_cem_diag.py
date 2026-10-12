#!/usr/bin/env python
"""Diagnostic: is V-JEPA2-AC's goal-latent energy actually action-sensitive?

Roll a battery of hand-picked 8-step action sequences through the predictor
from the same context (t=248..260) and score predicted final latents against
two goals: near (t=280, grasp contact, INSIDE horizon) and far (t=330, lifted,
beyond horizon). If energies barely differ across wildly different action
sequences -> flat landscape explains the degenerate CEM plan.
"""
import sys, json, time
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
OUT = "/home/aimsgroupuol/AIMSgeneral/UR_dVRK/blog_assets/loop2_vjepa_cem_energy"

CROP, PATCH = 256, 16
TOK = (CROP // PATCH) ** 2
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
DEV, DT = "cuda", torch.bfloat16
STR, T_PLAN, NCTX, ROLLOUT = 4, 260, 4, 8

sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/notebooks")
from src.hub.backbones import _make_vjepa2_ac_model  # noqa: E402
from utils.mpc_utils import compute_new_pose        # noqa: E402


def rot6d_to_euler(r6):
    a1, a2 = r6[:, 0:3], r6[:, 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True).clip(1e-9)
    b2 = a2 - (b1 * a2).sum(1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True).clip(1e-9)
    R = np.stack([a1 / np.linalg.norm(a1, axis=1, keepdims=True).clip(1e-9), b2,
                  np.cross(a1 / np.linalg.norm(a1, axis=1, keepdims=True).clip(1e-9), b2)], 2)
    return Rotation.from_matrix(R).as_euler("xyz", degrees=False)


def poses_to_diff7(s, e):
    dxyz = e[:3] - s[:3]
    Rs = Rotation.from_euler("xyz", s[3:6]).as_matrix()
    Re = Rotation.from_euler("xyz", e[3:6]).as_matrix()
    dth = Rotation.from_matrix(Re @ Rs.T).as_euler("xyz", degrees=False)
    return np.concatenate([dxyz, dth, e[6:7] - s[6:7]]).astype(np.float32)


def load_frames(path, idxs):
    cap = cv2.VideoCapture(path)
    out, i = [], 0
    want = set(idxs)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return np.stack(out)


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
def encode(encoder, frames_u8):
    c = preprocess(frames_u8).to(DEV, DT)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = encoder(c)
    return F.layer_norm(z.float(), (z.size(-1),))


@torch.no_grad()
def rollout_energy(pred, z_ctx, s_ctx, a_ctx_tail, cand_actions, z_goal):
    """cand_actions [K,R,7] float32 cpu. Returns [K] energies."""
    K, R, _ = cand_actions.shape
    frame_traj = z_ctx.repeat(K, 1, 1, 1)
    pose_traj = s_ctx.repeat(K, 1, 1)
    a_ctx = a_ctx_tail.repeat(K, 1, 1)
    a_cand = cand_actions.to(DEV)
    for h in range(R):
        full_a = torch.cat([a_ctx, a_cand[:, :h + 1]], 1)
        z = frame_traj.reshape(K, frame_traj.size(1) * frame_traj.size(2), frame_traj.size(3))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = pred(z, full_a.to(DT), pose_traj.to(DT))
        nxt = F.layer_norm(out[:, -TOK:].float(), (out.size(-1),)).unsqueeze(1)
        npose = compute_new_pose(pose_traj[:, -1:].float(), a_cand[:, h:h + 1].float()).to(DEV)
        frame_traj = torch.cat([frame_traj, nxt.to(DT)], 1)
        pose_traj = torch.cat([pose_traj, npose.to(DT)], 1)
    E = torch.mean(torch.abs(frame_traj[:, -1].float() - z_goal.float()), dim=(1, 2))
    return E.cpu()


def main():
    t0 = time.time()
    enc, pred = _make_vjepa2_ac_model(pretrained=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    clean = lambda d: {k.replace("module.", "").replace("backbone.", ""): v for k, v in d.items()}
    enc.load_state_dict(clean(sd["encoder"]), strict=False)
    pred.load_state_dict(clean(sd["predictor"]), strict=True)
    del sd
    enc = enc.to(DEV).to(DT).eval(); pred = pred.to(DEV).to(DT).eval()
    print(f"models {time.time()-t0:.0f}s", flush=True)

    tr = np.load(TRACE); st = tr["state"]
    L = np.concatenate([st[:, 0:3], rot6d_to_euler(st[:, 3:9]), st[:, 9:10]], 1).astype(np.float32)
    acts7 = np.stack([poses_to_diff7(L[t], L[min(t + STR, len(L) - 1)]) for t in range(len(L))])

    ctx_idx = np.arange(T_PLAN - (NCTX - 1) * STR, T_PLAN + 1, STR)
    goal_near, goal_far = 280, 330
    frames = load_frames(VID, np.concatenate([ctx_idx, [goal_near, goal_far]]))
    z_all = encode(enc, frames)
    z_ctx = z_all[:NCTX].view(1, NCTX, TOK, -1).to(DEV, DT)
    zg_near = z_all[NCTX].view(1, TOK, -1).to(DEV)
    zg_far = z_all[NCTX + 1].view(1, TOK, -1).to(DEV)
    s_ctx = torch.tensor(L[ctx_idx]).unsqueeze(0).to(DEV)
    a_ctx_tail = torch.tensor(acts7[ctx_idx[:-1]]).unsqueeze(0).to(DEV)  # C-1 real

    R = ROLLOUT
    def A(seq):
        """list of [dx,dy,dz,0,0,0,dg] -> tensor [R,7], pad zeros."""
        arr = np.zeros((R, 7), np.float32)
        for i, a in enumerate(seq):
            arr[i] = a
        return torch.tensor(arr)

    cand = {
        "zero (static)": A([[0]*7] * R),
        "descend -10mm/st": A([[0, 0, -0.010, 0, 0, 0, 0]] * R),
        "descend+close": A([[0, 0, -0.010, 0, 0, 0, 0]] * 4 + [[0, 0, 0, 0, 0, 0, 0.15]] * 4),
        "close only +0.2/st": A([[0, 0, 0, 0, 0, 0, 0.2]] * R),
        "ascend +10mm/st": A([[0, 0, 0.010, 0, 0, 0, 0]] * R),
        "descend+lift tail": A([[0, 0, -0.010, 0, 0, 0, 0]] * 3 + [[0, 0, 0, 0, 0, 0, 0.15]] * 2
                              + [[0, 0, 0.015, 0, 0, 0, 0]] * 3),
        "GT action chunk": torch.tensor(acts7[T_PLAN:T_PLAN + R * STR:STR]),
        "random": torch.randn(R, 7).clamp(-0.05, 0.05),
        "away -y 20mm/st": A([[0, -0.020, 0, 0, 0, 0, 0]] * R),
    }
    names = list(cand.keys())
    Aall = torch.stack([cand[n] for n in names])          # [K,R,7]
    print("rolling candidates ...", flush=True)
    E_near = rollout_energy(pred, z_ctx, s_ctx, a_ctx_tail, Aall, zg_near)
    E_far = rollout_energy(pred, z_ctx, s_ctx, a_ctx_tail, Aall, zg_far)

    res = {n: {"E_goal_t280": float(a), "E_goal_t330": float(b)}
           for n, a, b in zip(names, E_near, E_far)}
    for n in names:
        print(f"  {n:22s} E_t280={res[n]['E_goal_t280']:.4f}  E_t330={res[n]['E_goal_t330']:.4f}",
              flush=True)

    # also encode-identity floor: energy between real future frames and goals
    # (how much of E is just visual distance between t and goal?)
    z_ids = load_frames(VID, np.array([T_PLAN, goal_near, goal_far, 340, 380]))
    zid = encode(enc, z_ids)
    floor = {}
    for i, t in enumerate([T_PLAN, goal_near, goal_far, 340, 380]):
        floor[f"real_t{t}_vs_t280"] = float(torch.mean(torch.abs(zid[i] - zg_near.float())))
        floor[f"real_t{t}_vs_t330"] = float(torch.mean(torch.abs(zid[i] - zg_far.float())))
    print("identity floors:", {k: round(v, 4) for k, v in floor.items()}, flush=True)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [res[n]["E_goal_t280"] for n in names], 0.4, label="goal t=280 (grasp, in-horizon)")
    ax.bar(x + 0.2, [res[n]["E_goal_t330"] for n in names], 0.4, label="goal t=330 (lifted, far)")
    for i, n in enumerate(names):
        ax.text(i - 0.2, res[n]["E_goal_t280"] + 0.002, f"{res[n]['E_goal_t280']:.3f}",
                ha="center", fontsize=6)
        ax.text(i + 0.2, res[n]["E_goal_t330"] + 0.002, f"{res[n]['E_goal_t330']:.3f}",
                ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("energy E = mean|z_pred - z_goal|")
    ax.set_title("V-JEPA2-AC predicted-latent energy for hand-picked 8-step action sequences\n"
                 "(context t=248..260, UR_dVRK ep002_seed7) — landscape sensitivity check", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=150)
    with open(f"{OUT}.json", "w") as f:
        json.dump({"candidates": res, "identity_floors": floor,
                   "note": "identity floor = L1 between encoded REAL frames and goal latent; "
                           "predicted-rollout energies should be compared against this"},
                  f, indent=2)
    print("saved", f"{OUT}.png/.json", f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
