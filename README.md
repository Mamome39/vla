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

## nuScenes trajectory pipeline

Set the dataset path (the folder holding `v1.0-trainval/` and `samples/`):

```bash
cp .env.example .env   # then edit NUSCENES_ROOT
```

```bash
python scripts/nuscenes_gt.py --out outputs/nuscenes_gt_all.pt            # 3 s ego-trajectory GT
python scripts/nuscenes_subset.py --out outputs/nuscenes_gt_1000.pt        # train/val/test split by scene
python scripts/nuscenes_latent.py --gt outputs/nuscenes_gt_1000.pt --last-token-only \
    --out outputs/nuscenes_latents_1000.pt                                 # Qwen3.5-2B latents (GPU)
python scripts/train_head.py --head flow --gt outputs/nuscenes_gt_1000.pt \
    --latents outputs/nuscenes_latents_1000.pt --train 700 --val 150 --out-dir outputs/head_run_flow_1000
python scripts/visualize_test.py --run-dir outputs/head_run_flow_1000     # -> vis/test_10.png
```
