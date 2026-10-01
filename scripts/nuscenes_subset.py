"""Pick a train/val/test subset from the full GT file, split by scene (no scene in two splits).

Scenes are shuffled; within a scene every --stride-th keyframe is kept (neighbors 0.5 s apart are near-duplicates).
Output is ordered train | val | test, so train_head.py --train/--val slices it directly.

    python scripts/nuscenes_subset.py --train 700 --val 150 --test 150 --out outputs/nuscenes_gt_1000.pt
"""

import argparse
import random
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/nuscenes_gt_all.pt")
    ap.add_argument("--train", type=int, default=700)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/nuscenes_gt_1000.pt")
    args = ap.parse_args()

    src = torch.load(args.src)
    by_scene = {}
    for i, s in enumerate(src["samples"]):
        by_scene.setdefault(s["scene"], []).append(i)
    scenes = sorted(by_scene)
    random.Random(args.seed).shuffle(scenes)

    picked, split_scenes = [], {}
    it = iter(scenes)
    for name, n in [("train", args.train), ("val", args.val), ("test", args.test)]:
        chosen, used = [], []
        while len(chosen) < n:
            scene = next(it)  # each scene is consumed by exactly one split
            chosen += by_scene[scene][:: args.stride][: n - len(chosen)]
            used.append(scene)
        picked += chosen
        split_scenes[name] = used
        print(f"{name:<5} {len(chosen):>4} samples from {len(used)} scenes")

    out = Path(args.out)
    torch.save({**{k: v for k, v in src.items() if k not in ("trajectory", "samples")},
                "trajectory": src["trajectory"][picked], "samples": [src["samples"][i] for i in picked],
                "split": {"train": args.train, "val": args.val, "test": args.test}, "split_scenes": split_scenes,
                "stride": args.stride, "seed": args.seed}, out)
    print(f"Saved {len(picked)} samples -> {out}")


if __name__ == "__main__":
    main()
