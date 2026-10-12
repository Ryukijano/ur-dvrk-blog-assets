#!/usr/bin/env python
"""V-JEPA 2-AC feasibility probe on UR_dVRK UR5 rollout.

Loads facebookresearch/vjepa2 ViT-g encoder + action-conditioned predictor
(checkpoint: dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt, 11.76GB fp32).

Demo: take real context frames + real left-arm EE states/actions from
episode_002_seed7, predict future latents autoregressively, compare against
encoded actual future frames -> PCA + cosine-sim + NN-retrieval figure.
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
OUT_PNG = "/home/aimsgroupuol/AIMSgeneral/UR_dVRK/blog_assets/loop2_vjepa_probe.png"
OUT_GIF = "/home/aimsgroupuol/AIMSgeneral/UR_dVRK/blog_assets/loop2_vjepa_probe.gif"

CROP = 256
PATCH = 16
TOK = (CROP // PATCH) ** 2          # 256 tokens/frame
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
DEV = "cuda"
DT = torch.bfloat16

sys.path.insert(0, REPO)
from src.hub.backbones import _make_vjepa2_ac_model  # noqa: E402


def rot6d_to_euler(r6):
    """(N,6) rot6d -> (N,3) euler xyz radians (Gram-Schmidt)."""
    a1, a2 = r6[:, 0:3], r6[:, 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True).clip(1e-9)
    b2 = a2 - (b1 * a2).sum(1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True).clip(1e-9)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=2)          # columns
    return Rotation.from_matrix(R).as_euler("xyz", degrees=False)


def poses_to_diff7(s, e):
    """[7],[7] -> action [dx,dy,dz, d_euler_xyz(3), dgrip] as in V-JEPA2-AC."""
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
    return np.stack(frames)                       # (T,480,640,3) uint8


def preprocess(frames_u8):
    """(N,480,640,3) uint8 -> (N,3,2,256,256) normalized dup-tubelet clips."""
    out = []
    for f in frames_u8:
        h, w = f.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        f = cv2.resize(f[y0:y0 + s, x0:x0 + s], (CROP, CROP), interpolation=cv2.INTER_LINEAR)
        out.append(torch.from_numpy(f).float().permute(2, 0, 1) / 255.0)
    x = torch.stack(out).unsqueeze(2).repeat(1, 1, 2, 1, 1)   # (N,3,2,256,256)
    return (x - MEAN) / STD


@torch.no_grad()
def encode_frames(encoder, frames_u8, bs=24):
    """Per-frame latents via dup-frame tubelet -> (N,256,1408), layernormed."""
    clips = preprocess(frames_u8)
    zs = []
    for i in range(0, len(clips), bs):
        c = clips[i:i + bs].to(DEV, DT)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = encoder(c)
        zs.append(F.layer_norm(z.float(), (z.size(-1),)).cpu())
    return torch.cat(zs)


@torch.no_grad()
def predict_next(predictor, z_ctx, actions, states):
    """z_ctx (1,T*256,D), actions/states (1,T,7) -> predicted last-frame latents."""
    a = actions.to(DEV, DT); s = states.to(DEV, DT); z = z_ctx.to(DEV, DT)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = predictor(z, a, s)
    out = F.layer_norm(out.float(), (out.size(-1),))
    return out[:, -TOK:].cpu()


def main():
    t0 = time.time()
    print("[1/5] building models ...", flush=True)
    enc, pred = _make_vjepa2_ac_model(pretrained=False)
    print("      loading state dict ...", flush=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("      ckpt keys:", list(sd.keys())[:8])
    for k in list(sd.get("encoder", {}).keys())[:0]:
        pass
    def clean(d):
        return {k.replace("module.", "").replace("backbone.", ""): v for k, v in d.items()}
    enc.load_state_dict(clean(sd["encoder"]), strict=False)
    pred.load_state_dict(clean(sd["predictor"]), strict=True)
    enc = enc.to(DEV).to(DT).eval(); pred = pred.to(DEV).to(DT).eval()
    print(f"      models on GPU ({time.time()-t0:.0f}s), vram_alloc={torch.cuda.memory_allocated()/2**30:.1f}GiB", flush=True)

    print("[2/5] loading video + trace ...", flush=True)
    frames = load_video_frames(VID)
    tr = np.load(TRACE)
    st = tr["state"]                                    # (510,20) bimanual
    L_pos, L_r6, L_g = st[:, 0:3], st[:, 3:9], st[:, 9:10]
    L_eul = rot6d_to_euler(L_r6)
    states7 = np.concatenate([L_pos, L_eul, L_g], axis=1).astype(np.float32)  # (510,7)

    STR = 4                                             # 15fps -> ~3.75fps (DROID train fps=4)
    NCTXT, HOR = 8, 8                                   # 8-frame ctx (fpcs=8), 8-step rollout ~2.1s
    T0 = 100                                            # approach phase (block reach)
    # stride-aware EE-delta actions: a(t) = state(t) -> state(t+STR)
    acts7 = np.stack([poses_to_diff7(states7[t], states7[min(t + STR, len(states7) - 1)])
                      for t in range(len(states7))])                           # (510,7)
    print(f"      video {frames.shape}, states7 {states7.shape}, acts7 {acts7.shape}", flush=True)

    print("[3/5] encoding reference frames ...", flush=True)
    ref_idx = np.arange(0, len(frames), STR)
    z_ref = encode_frames(enc, frames[ref_idx])         # (Nref,256,1408)
    z_ref_pooled = F.layer_norm(z_ref.mean(1), (z_ref.size(-1),))
    print(f"      {z_ref.shape} ref latents", flush=True)

    print("[4/5] autoregressive latent rollout ...", flush=True)
    ctx_idx = np.arange(T0, T0 + NCTXT * STR, STR)      # real context frame ids
    fut_idx = np.arange(T0 + NCTXT * STR, T0 + (NCTXT + HOR) * STR, STR)
    z_ctx = encode_frames(enc, frames[ctx_idx])         # (8,256,1408)
    z_ctx = z_ctx.view(1, -1, z_ctx.size(-1))           # (1,8*256,D)
    z_fut_true = encode_frames(enc, frames[fut_idx])    # (8,256,1408)
    s_seq = torch.tensor(states7[np.concatenate([ctx_idx, fut_idx])]).unsqueeze(0)
    a_seq = torch.tensor(acts7[np.concatenate([ctx_idx, fut_idx])]).unsqueeze(0)

    z_roll = z_ctx.clone()
    s_hist = s_seq[:, :NCTXT].clone()
    a_hist = a_seq[:, :NCTXT].clone()
    preds = []
    for hstep in range(HOR):
        zh = predict_next(pred, z_roll, a_hist, s_hist)
        preds.append(zh)
        z_roll = torch.cat([z_roll, zh], dim=1)
        s_hist = torch.cat([s_hist, s_seq[:, NCTXT + hstep:NCTXT + hstep + 1]], 1)
        a_hist = torch.cat([a_hist, a_seq[:, NCTXT + hstep:NCTXT + hstep + 1]], 1)
    z_pred = torch.cat(preds, 1).view(HOR, TOK, -1)     # (8,256,1408)

    cos = F.cosine_similarity(z_pred, z_fut_true, dim=-1).mean(1)     # (8,)
    l1 = (z_pred - z_fut_true).abs().mean([1, 2])
    # NN retrieval on pooled latents
    zp = F.layer_norm(z_pred.mean(1), (z_pred.size(-1),))
    sims = zp @ z_ref_pooled.T
    nn_idx = sims.argmax(1).numpy()
    print("      horizon cos-sim:", [round(float(c), 3) for c in cos], flush=True)
    print("      horizon l1     :", [round(float(l), 4) for l in l1], flush=True)
    print("      NN frame idx   :", ref_idx[nn_idx], "true:", fut_idx, flush=True)

    print("[5/5] figure ...", flush=True)
    # PCA on pooled ref latents
    X = z_ref_pooled.numpy()
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    def pca(z):  # (N,256,D) -> (N,2)
        return ((z.mean(1).numpy() - mu) @ Vt[:2].T)
    pc_ref = pca(z_ref)
    ctx_z = z_ctx.view(NCTXT, TOK, -1)
    pc_ctx = pca(ctx_z); pc_true = pca(z_fut_true); pc_pred = pca(z_pred)

    fig = plt.figure(figsize=(17, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.8], wspace=0.25)

    # Panel A: real future frames vs NN-retrieved frames
    axA = fig.add_subplot(gs[0]); axA.axis("off"); axA.set_title(
        "Autoregressive predictions vs reality\n(top: true future frame | bottom: frame whose latent is nearest the prediction)", fontsize=9)
    # build 2x8 montage at 160px thumbs
    th = 160
    def thumb(fr):
        h, w = fr.shape[:2]; s = min(h, w); y, x = (h - s)//2, (w - s)//2
        return cv2.resize(fr[y:y+s, x:x+s], (th, th))
    top = np.concatenate([thumb(frames[fut_idx[i]]) for i in range(HOR)], axis=1)
    bot = np.concatenate([thumb(frames[ref_idx[nn_idx[i]]]) for i in range(HOR)], axis=1)
    sep = np.zeros((10, top.shape[1], 3), np.uint8)
    mont = np.concatenate([top, sep, bot], axis=0)
    axA.imshow(mont); axA.set_aspect("equal")
    for i in range(HOR):
        axA.text(i * th + 4, 14, f"+{i+1}", color="cyan", fontsize=7, weight="bold")
        axA.text(i * th + 4, th + 24, f"f{ref_idx[nn_idx[i]]}", color="yellow", fontsize=7)

    # Panel B: PCA
    axB = fig.add_subplot(gs[1])
    axB.scatter(pc_ref[:, 0], pc_ref[:, 1], s=8, c="lightgrey", label="episode frames (encoded)")
    axB.plot(pc_ctx[:, 0], pc_ctx[:, 1], "g-o", ms=4, lw=1.2, label="context (real, 8f)")
    axB.plot(pc_true[:, 0], pc_true[:, 1], "b-o", ms=5, lw=1.5, label="future (real)")
    axB.plot(pc_pred[:, 0], pc_pred[:, 1], "r--^", ms=5, lw=1.5, label="future (predicted)")
    for i in range(HOR):
        axB.annotate(f"+{i+1}", pc_pred[i], fontsize=7, color="r")
        axB.annotate(f"+{i+1}", pc_true[i], fontsize=7, color="b")
    axB.set_title("V-JEPA2 encoder latents (PCA, frame-pooled)\nUR5 pick-and-place, ep002", fontsize=9)
    axB.legend(fontsize=7, loc="best"); axB.set_xlabel("PC1"); axB.set_ylabel("PC2")

    # Panel C: metrics
    axC = fig.add_subplot(gs[2])
    axC.plot(range(1, HOR + 1), cos.numpy(), "b-o", label="cosine sim (pred vs true latent)")
    axC2 = axC.twinx()
    axC2.plot(range(1, HOR + 1), l1.numpy(), "r--s", label="mean |Δ| L1")
    axC.set_xlabel("autoregressive horizon step (~0.27s each)"); axC.set_ylabel("cos sim", color="b")
    axC2.set_ylabel("L1", color="r"); axC.set_title("Prediction fidelity vs horizon", fontsize=9)
    lns = axC.get_lines() + axC2.get_lines()
    axC.legend(lns, [l.get_label() for l in lns], fontsize=7, loc="lower left")
    axC.set_ylim(0, 1.02)

    fig.suptitle("V-JEPA 2-AC (ViT-g, DROID-trained) zero-shot on UR_dVRK UR5 rollout — latent prediction, no finetuning", fontsize=11)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print("saved", OUT_PNG, flush=True)

    # GIF: predicted-vs-true NN retrieval sweep
    try:
        imgs = []
        for i in range(HOR):
            t = np.concatenate([thumb(frames[fut_idx[i]]), np.ones((th, 8, 3), np.uint8) * 40, thumb(frames[ref_idx[nn_idx[i]]])], 1)
            cv2.putText(t, f"step +{i+1} true f{fut_idx[i]} (L) | NN f{ref_idx[nn_idx[i]]} (R)", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            imgs.append(cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
        h, w = imgs[0].shape[:2]
        vw = cv2.VideoWriter(OUT_GIF.replace(".gif", ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2, (w, h))
        for im in imgs:
            vw.write(im)
        vw.release()
        print("saved", OUT_GIF.replace(".gif", ".mp4"), flush=True)
    except Exception as e:
        print("gif/mp4 skipped:", e)

    summary = {
        "context_frames": ctx_idx.tolist(), "future_frames": fut_idx.tolist(),
        "cos_per_step": [float(c) for c in cos], "l1_per_step": [float(l) for l in l1],
        "nn_retrieved_frames": ref_idx[nn_idx].tolist(),
        "model": "vjepa2-ac-vitg (encoder 1.01B + AC predictor 0.31B, fp32 ckpt 11.76GB, run bf16)",
        "action_spec": "7-DoF EE delta [dxyz, deuler_xyz, dgrip]; state = [xyz, euler, grip] left arm",
    }
    with open(OUT_PNG.replace(".png", ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
