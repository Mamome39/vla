"""Draw predicted vs GT trajectories for test samples: projected on CAM_FRONT + bird's-eye view,
captioned with Qwen3.5-2B's text answer to the same 6-camera prompt the latents were extracted with.

Captions are cached by sample token in outputs/captions.json, so the backbone only runs for new samples.

    python scripts/visualize_test.py --run-dir outputs/head_run_flow
"""

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from vla_arch.data.nuscenes import CAMS, DRIVING_PROMPT, load_cams

CAM = "CAM_FRONT"
PRED_COLORS = ["red", "orange", "magenta", "cyan", "yellow"]


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def project(traj_xy, calib):
    """Ego-frame ground points (N, 2) -> pixel coords (M, 2) of points in front of the camera."""
    pts = np.concatenate([traj_xy, np.zeros((len(traj_xy), 1))], axis=1)  # z = 0: ground
    cam = (pts - np.asarray(calib["translation"])) @ quat_to_rot(calib["rotation"])  # ego -> camera
    cam = cam[cam[:, 2] > 0.5]
    uv = cam @ np.asarray(calib["camera_intrinsic"]).T
    return uv[:, :2] / uv[:, 2:3]


def get_captions(samples, cache_path, max_new_tokens):
    """Qwen3.5-2B answers to DRIVING_PROMPT for each sample, reusing cached ones (same max_new_tokens)."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get("max_new_tokens") != max_new_tokens:
        cache = {"max_new_tokens": max_new_tokens, "captions": {}}
    captions = cache["captions"]
    todo = [s for s in samples if s["sample_token"] not in captions]
    if todo:
        from vla_arch.backbones import Qwen35_2B

        model = Qwen35_2B()
        for b in range(0, len(todo), 5):
            batch = todo[b : b + 5]
            texts = model.chat(images=[load_cams(s["image_paths"]) for s in batch], texts=[DRIVING_PROMPT] * len(batch),
                               max_new_tokens=max_new_tokens)
            for s, t in zip(batch, texts):
                captions[s["sample_token"]] = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
            print(f"captioned {min(b + 5, len(todo))}/{len(todo)}")
        cache_path.write_text(json.dumps(cache, indent=2))
    return [captions[s["sample_token"]] for s in samples]


def format_caption(text, width=78):
    """Strip markdown, keep paragraph/bullet line breaks, wrap each line."""
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^#+\s*", "", line.strip()).replace("**", "")
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        indent = "  " if line.startswith(("-", "*", "•")) or re.match(r"^\d+\.", line) else ""
        lines += textwrap.wrap(line, width, subsequent_indent=indent + "  " if indent else "")
    return "\n".join(lines)


def draw_fitted_text(ax, text, max_size=7.5, min_size=4.0):
    """Monospace text in `ax`, wrapped to its width, shrinking the font until it fits its height."""
    fig = ax.figure
    box = ax.get_position()
    w_pt, h_pt = box.width * fig.get_figwidth() * 72, box.height * fig.get_figheight() * 72
    size = max_size
    while True:
        body = format_caption(text, width=int(0.85 * w_pt / (0.6 * size)))  # monospace char ~0.6 em, 15% margin
        if body.count("\n") * size * 1.25 <= h_pt or size <= min_size:
            break
        size -= 0.25
    ax.text(0, 1, body, transform=ax.transAxes, va="top", ha="left", fontsize=size, family="monospace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/head_run_mlp", help="output dir of train_head.py")
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--caption-tokens", type=int, default=512, help="max new tokens per caption")
    ap.add_argument("--caption-cache", default="outputs/captions.json")
    args = ap.parse_args()

    d = torch.load(Path(args.run_dir) / "test_preds.pt")
    pred, gt, samples = d["pred"].numpy(), d["gt"].numpy(), d["samples"]
    if pred.ndim == 3:  # older runs saved (N, 6, 2)
        pred = pred[:, None]
    k = pred.shape[1]
    picks = np.linspace(0, len(samples) - 1, args.num).round().astype(int)  # spread over the test set
    out = Path(args.run_dir) / "vis"
    out.mkdir(parents=True, exist_ok=True)

    captions = [None] * len(picks) if args.no_captions else \
        get_captions([samples[i] for i in picks], Path(args.caption_cache), args.caption_tokens)

    ncols = 2 if args.no_captions else 3  # camera | BEV | caption
    fig, axes = plt.subplots(args.num, ncols, figsize=(20 if ncols == 3 else 14, 5.2 * args.num),
                             gridspec_kw={"width_ratios": [2.2, 0.9, 1.9][:ncols]})
    for row, i in enumerate(picks):
        s = samples[i]
        calib = s["calib"][CAM]
        # Prepend the ego origin so lines start at the car.
        origin = np.zeros((1, 2))
        preds = [np.vstack([origin, pred[i, j]]) for j in range(k)]
        g = np.vstack([origin, gt[i]])
        err = np.linalg.norm(pred[i] - gt[i][None], axis=-1)  # (K, 6)
        best = err.mean(1).argmin()
        lines = [(g, "lime", "GT")] + [(p, PRED_COLORS[j % len(PRED_COLORS)], f"pred {j}" + (" (best)" if k > 1 and j == best else ""))
                                       for j, p in enumerate(preds)]

        ax = axes[row, 0]
        img = Image.open(s["image_paths"][CAMS.index(CAM)])
        ax.imshow(img)
        for traj, color, label in lines:
            uv = project(traj, calib)
            ax.plot(uv[:, 0], uv[:, 1], "-o", color=color, lw=2.5 if label == "GT" else 1.8, ms=5 if label == "GT" else 3,
                    label=label)
        ax.set_xlim(0, img.width)
        ax.set_ylim(img.height, 0)
        title = f"ADE {err[0].mean():.2f} m  FDE {err[0, -1]:.2f} m" if k == 1 else \
            f"minADE {err.mean(1).min():.2f} m  minFDE {err[:, -1].min():.2f} m  (best of {k})"
        ax.set_title(f"{s['scene']}  {s['sample_token'][:8]}  |  {title}", fontsize=10)
        ax.axis("off")
        ax.legend(loc="upper right", fontsize=7)

        ax = axes[row, 1]  # BEV: x forward is up, y left is left
        for traj, color, label in lines:
            ax.plot(-traj[:, 1], traj[:, 0], "-o", color="green" if label == "GT" else color, ms=3)
        ax.plot(0, 0, "ks", ms=8)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-2, 30)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("← left   lateral (m)   right →", fontsize=8)
        ax.set_ylabel("forward (m)", fontsize=8)


    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.12, hspace=0.12)

    for row, caption in enumerate(captions):
        if caption:
            ax = axes[row, 2]
            ax.axis("off")
            draw_fitted_text(ax, "Qwen3.5-2B (same 6-cam prompt as the latents):\n\n" + caption)
    fig.savefig(out / "test_10.png", dpi=80)
    print(f"Saved {out / 'test_10.png'}  (test indices {picks.tolist()})")


if __name__ == "__main__":
    main()
