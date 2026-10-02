# Reports

## Ground truth

- **Source:** nuScenes `v1.0-trainval` keyframes (2 Hz).
- **Target:** ego future trajectory, 6 waypoints × 0.5 s = 3 s horizon, xy in the current ego frame (x forward, y left, meters), from the LIDAR_TOP ego pose of each future keyframe.
- **Inputs stored per sample:** 6 surround-camera image paths + camera calibration.
- Keyframes without 6 future keyframes in their scene are dropped.
- Full set: `outputs/nuscenes_gt_all.pt` — 29,049 samples from 850 scenes.

```bash
python scripts/nuscenes_gt.py --out outputs/nuscenes_gt_all.pt
```

## Mini train subset (1000)

- Scenes shuffled (`seed=0`), then assigned whole to one split, so no scene leaks across train/val/test.
- Within a scene, every 4th keyframe is kept (`stride=4`, i.e. 2 s apart), since neighbouring keyframes are near-duplicates.
- Result `outputs/nuscenes_gt_1000.pt`, ordered train | val | test:

| Split | Samples | Scenes |
|---|---|---|
| train | 700 | 78 |
| val | 150 | 17 |
| test | 150 | 17 |

```bash
python scripts/nuscenes_subset.py --train 700 --val 150 --test 150 --stride 4 --seed 0 \
    --out outputs/nuscenes_gt_1000.pt
```

## Experiments: action heads on the mini split

Head designs are in [actionhead.md](actionhead.md).

**Input config (shared)**
- GT: `outputs/nuscenes_gt_1000.pt` (700 / 150 / 150).
- Latents: `outputs/nuscenes_latents_1000.pt`, the Qwen3.5-2B last token (2048-d). The model sees the 6 cameras at 800 px wide plus a driving prompt.
- Training: full-batch AdamW (lr 1e-3, wd 1e-2, seed 0). The checkpoint kept is the one with the best val ADE.

| Experiment | Head | Epochs | Trajectories (K) | Best epoch |
|---|---|---|---|---|
| `head_run_mlp_1000` | MLP | 1000 | 1 | 141 |
| `head_run_flow_1000` | Flow matching | 5000 | 3 | 1202 |

**Output** (in `outputs/<experiment>/`)
- `head.pt`: the best-val checkpoint.
- `metrics.json`: metrics for train, val and test, plus the mean-trajectory baseline.
- `test_preds.pt`: test predictions `(N, K, 6, 2)` alongside the GT.
- `vis/test_10.png`: predicted vs GT trajectories for 10 test samples, on CAM_FRONT and in bird's-eye view.

### Results (test, 150 samples, metres)

ADE/FDE are averaged over the K trajectories. minADE/minFDE take the best of the K.

| Model | L2@1s | L2@2s | L2@3s | ADE | FDE | minADE | minFDE |
|---|---|---|---|---|---|---|---|
| Mean-trajectory baseline | 3.54 | 7.16 | 10.83 | 6.27 | 10.83 | – | – |
| MLP | 2.02 | 4.06 | 6.27 | 3.59 | 6.27 | – | – |
| Flow matching (K=3) | **1.80** | **3.70** | **5.80** | **3.28** | **5.80** | **2.70** | **4.67** |

- Both heads roughly halve the baseline error, and flow is best on every metric.
- Both heads overfit (train ADE: MLP 2.11, flow 0.92), so the limit is generalization rather than head capacity.
