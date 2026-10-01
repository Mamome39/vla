# vla-arch

## Setup

```bash
conda env create -f environment.yml
conda activate vla_arch
```

Torch is pinned to `2.11.0+cu128`: the driver supports CUDA 12.8, and newer torch on PyPI is built for cu130.

## Qwen3.5-2B

```python
from vla_arch.backbones import Qwen35_2B

bb = Qwen35_2B()                                        # downloads on first use
feats, mask = bb(images=[[img]], texts=["pick up the red cube"])   # (B, T, 2048), (B, T)
bb.chat(images=[[img]], texts=["What is in this image?"])          # text answers
```

Try it: `python scripts/smoke_backbone.py`

## Action heads

Map a backbone latent `(B, 2048)` to a trajectory of `num_steps` xy waypoints.

```python
from vla_arch.heads import FlowMatchingActionHead, MLPActionHead

head = MLPActionHead(in_dim=2048, num_steps=6)
traj = head(latent)                       # (B, 6, 2)
loss = head.loss(latent, gt_traj)         # L2 displacement

head = FlowMatchingActionHead(in_dim=2048, num_steps=6, num_samples=3)
head.fit_normalization(train_traj)        # (N, 6, 2)
loss = head.loss(latent, gt_traj)         # flow-matching velocity MSE
trajs = head(latent)                      # (B, 3, 6, 2): one trajectory per noise draw
```
