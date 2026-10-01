"""Feed nuScenes keyframes (6 surround cameras + prompt) to Qwen3.5-2B and save last-layer latents.

Dataset path comes from NUSCENES_ROOT (env var or ./.env, which is gitignored).

    python scripts/nuscenes_latent.py --num-samples 10
"""

import argparse
import time
from pathlib import Path

import torch

from vla_arch.backbones import Qwen35_2B
from vla_arch.data.nuscenes import CAMS, DRIVING_PROMPT as PROMPT
from vla_arch.data.nuscenes import keyframe_index, load_cams, load_json, nuscenes_root, scene_sample_tokens


def load_samples(root: Path, version: str, n: int, scene_idx: int) -> list[dict]:
    """First n keyframes of one scene, each with its 6 camera image paths."""
    scene = load_json(root, version, "scene")[scene_idx]
    samples = {s["token"]: s for s in load_json(root, version, "sample")}
    tokens = scene_sample_tokens(samples, scene)[:n]

    print(f"Indexing {version}/sample_data.json (large, takes a bit)...")
    index = keyframe_index(root, version, set(tokens))
    return [
        {"sample_token": t, "scene": scene["name"], "timestamp": samples[t]["timestamp"],
         "images": [index[t]["cams"][c] for c in CAMS]}
        for t in tokens
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--scene", type=int, default=0, help="scene index in scene.json")
    ap.add_argument("--image-width", type=int, default=800, help="resize cameras (1600x900 native) to cut tokens")
    ap.add_argument("--gt", help="encode the samples of this GT file (from nuscenes_gt.py) instead of --scene")
    ap.add_argument("--last-token-only", action="store_true", help="skip saving all-token latents (~17 MB/sample)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="outputs/nuscenes_latents.pt")
    args = ap.parse_args()

    if args.gt:
        samples = [{**s, "images": s["image_paths"]} for s in torch.load(args.gt)["samples"]]
    else:
        samples = load_samples(nuscenes_root(), args.version, args.num_samples, args.scene)
    model = Qwen35_2B()

    results = []
    for b in range(0, len(samples), args.batch_size):
        batch = samples[b : b + args.batch_size]
        t0 = time.time()
        with torch.no_grad():
            hidden, mask = model(images=[load_cams(s["images"], args.image_width) for s in batch], texts=[PROMPT] * len(batch))
        torch.cuda.synchronize()

        for j, s in enumerate(batch):
            h = hidden[j, mask[j].bool()]  # drop (left) padding -> (T, 2048)
            rec = {
                "sample_token": s["sample_token"],
                "scene": s["scene"],
                "timestamp": s["timestamp"],
                "cams": CAMS,
                "image_paths": [str(p) for p in s["images"]],
                "last_token": h[-1].float().cpu(),  # (2048,) final-position summary vector
            }
            if not args.last_token_only:
                rec["last_hidden_state"] = h.float().cpu()  # every token's last-layer latent
            results.append(rec)
        print(f"[{len(results)}/{len(samples)}] batch of {len(batch)}  latent {tuple(hidden.shape)}  "
              f"{time.time() - t0:.2f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt": PROMPT, "model": "Qwen/Qwen3.5-2B", "samples": results}, out)
    print(f"Saved {len(results)} samples -> {out}  | peak vram {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
