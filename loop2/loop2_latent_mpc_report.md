# Loop-2 latent MPC — UR_dVRK block pick

World model F(z_t, a_t) -> Δz_(t+1): MLP 20->256x3->10, EE-state space
(10-d left arm: xyz + rot6d + grip; actions = absolute commanded EE pose).
Trained on 65294 transitions from 108 block episodes,
held out 12 episodes.

## Dynamics accuracy (held-out episodes)
- one-step normalized MSE: 0.4692
- one-step xyz error: mean 1.45 mm, median 1.05 mm, p95 3.23 mm
- gripper Δ MAE: 0.0022
- 25-step open-loop rollout xyz drift: mean 24.6 mm, median 20.9 mm, p95 53.7 mm

## MPC (episode 0, t0=100 hover-above-block, goal = grasp latent frame 227, horizon 25)
J(A) = 0.5||z_T - z_goal||² + 0.1·0.5·Σ_k||z_k - z_goal||²
     + 0.05/2·||A-A0||² + 1.0/2·||ΔA||² + 0.8/2·Σ||Δz_pred||²
(terminal anchors the end, over-horizon term shapes "arrive and stay"), Adam lr=0.05,
240 iters, actions projected onto the demonstrated command range each step.

- terminal ||z_T - z_goal|| xyz: degraded A0 66.3 mm -> optimized 1.9 mm
- terminal grip error: 0.284 -> 0.062
- (reference) recorded GT chunk terminal error: 59.7 mm; GT-init MPC: 6.0 mm
- J: 32.343 -> 5.569

- max predicted per-step EE speed: A0 119 mm, MPC 26 mm
  (dataset fast-transit steps reach ~30-45 mm, so the dive is inside the demonstrated envelope)

Honest caveat: latent space = EE state space (not image-latent JEPA); actions are
commanded EE poses, so F learns tracking/lag dynamics; the plan is optimal
w.r.t. the learned model, not the true env.
