"""Build ego future-trajectory GT from nuScenes: 6 keyframes x 0.5s = 3s, xy in the current ego frame
(x forward, y left, meters). Keyframes without 6 future steps in their scene are skipped.

    python scripts/nuscenes_gt.py                     # every eligible keyframe
    python scripts/nuscenes_gt.py --num-samples 200   # first 200
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from vla_arch.data.nuscenes import CAMS, keyframe_index, load_json, nuscenes_root, scene_sample_tokens, to_ego_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, help="default: all eligible keyframes")
    ap.add_argument("--num-steps", type=int, default=6)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", default="outputs/nuscenes_gt.pt")
    args = ap.parse_args()

    root = nuscenes_root()
    samples = {s["token"]: s for s in load_json(root, args.version, "sample")}
    scenes = load_json(root, args.version, "scene")

    # Pick (current, 6 future) keyframe chains, scene by scene, until we have enough.
    chains = []
    for scene in scenes:
        toks = scene_sample_tokens(samples, scene)
        for i in range(len(toks) - args.num_steps):
            chains.append((scene["name"], toks[i : i + args.num_steps + 1]))
            if len(chains) == args.num_samples:
                break
        if len(chains) == args.num_samples:
            break
    print(f"{len(chains)} samples from {len({n for n, _ in chains})} scenes")

    print("Indexing sample_data + ego_pose (large, takes a bit)...")
    index = keyframe_index(root, args.version, {t for _, ch in chains for t in ch})
    poses = {p["token"]: p for p in load_json(root, args.version, "ego_pose")}
    calibs = {c["token"]: c for c in load_json(root, args.version, "calibrated_sensor")}

    records, trajs = [], []
    for scene_name, chain in chains:
        chain_poses = [poses[index[t]["ego_pose_token"]] for t in chain]
        future_xy = np.array([p["translation"][:2] for p in chain_poses[1:]])
        trajs.append(to_ego_frame(future_xy, chain_poses[0]))
        cur = chain[0]
        records.append({
            "sample_token": cur,
            "scene": scene_name,
            "timestamp": samples[cur]["timestamp"],
            "image_paths": [str(index[cur]["cams"][c]) for c in CAMS],
            "future_dt": (np.diff([samples[t]["timestamp"] for t in chain]) / 1e6).tolist(),
            # camera -> ego extrinsic (translation, quaternion wxyz) and intrinsic K, for drawing on images
            "calib": {c: {k: calibs[index[cur]["calib"][c]][k] for k in ("translation", "rotation", "camera_intrinsic")}
                      for c in CAMS},
        })

    gt = torch.tensor(np.stack(trajs), dtype=torch.float32)  # (N, num_steps, 2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": args.version, "cams": CAMS, "frame": "ego (x forward, y left), meters",
                "trajectory": gt, "samples": records}, out)

    scenes_used = sorted({r["scene"] for r in records})
    final = gt[:, -1]
    print(f"Saved {len(records)} samples from {len(scenes_used)} scenes ({scenes_used[0]}..{scenes_used[-1]}) -> {out}")
    print(f"trajectory {tuple(gt.shape)} | final-step x [{final[:, 0].min():.1f}, {final[:, 0].max():.1f}] m, "
          f"y [{final[:, 1].min():.1f}, {final[:, 1].max():.1f}] m | stationary (<0.5 m) {(final.norm(dim=1) < 0.5).sum().item()}")


if __name__ == "__main__":
    main()
