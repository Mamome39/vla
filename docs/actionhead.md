# Action heads

Both heads map the frozen Qwen3.5-2B last-token latent `(2048,)` to a 6 × xy trajectory (3 s, ego frame). Each starts with a LayerNorm on the input, because the latent norms are large (~160). Code: [action_head.py](../src/vla_arch/heads/action_head.py).

## MLP head

- Deterministic: 1 trajectory per input.
- LN → 2048→512 → 512→512 → 512→12, with GELU activations, reshaped to `(6, 2)`.
- Loss: mean L2 displacement (m) to the GT waypoints.

## Flow-matching head

- Stochastic: K trajectories per input (default K=3), one per noise draw.
- Conditional rectified flow over the flattened, per-coordinate normalized trajectory (12-dim). The normalization is fit on train GT.
- Condition: LN → Linear 2048→256.
- Velocity MLP: `[x_t (12), sin-cos(t) (64), cond (256)]` → 512 → 512 → 12, with GELU activations.
- Training: `x0 ~ N(0, I)`, `x1` = normalized GT, `x_t = (1−t)·x0 + t·x1`; MSE on the velocity `x1 − x0`.
- Sampling: 10 Euler steps from noise (t=0) to the trajectory (t=1).
