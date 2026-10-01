"""Train an action head (MLP or flow matching) on frozen Qwen3.5-2B last-token latents -> 6-step ego trajectory.

Split is chronological (samples are ordered scene by scene): first --train, next --val, rest test.
Checkpoint = best val ADE. Predictions are (N, K, 6, 2): K=1 for mlp, K trajectories (one per noise) for flow.

    python scripts/train_head.py --head mlp
    python scripts/train_head.py --head flow
"""

import argparse
import json
from pathlib import Path

import torch

from vla_arch.heads import FlowMatchingActionHead, MLPActionHead


def metrics(pred, gt):
    """pred (N, K, 6, 2), gt (N, 6, 2). L2 (m) at 1s/2s/3s (steps 2/4/6 at 2 Hz), ADE, FDE averaged over
    the K trajectories; minADE/minFDE take the best of the K per sample."""
    err = (pred - gt[:, None]).norm(dim=-1)  # (N, K, 6)
    return {"L2_1s": err[..., 1].mean().item(), "L2_2s": err[..., 3].mean().item(), "L2_3s": err[..., 5].mean().item(),
            "ADE": err.mean().item(), "FDE": err[..., -1].mean().item(),
            "minADE": err.mean(-1).min(1).values.mean().item(), "minFDE": err[..., -1].min(1).values.mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="outputs/nuscenes_gt.pt")
    ap.add_argument("--latents", default="outputs/nuscenes_latents_gt200.pt")
    ap.add_argument("--train", type=int, default=100)
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--head", choices=["mlp", "flow"], default="mlp")
    ap.add_argument("--epochs", type=int, help="default: 1000 for mlp, 5000 for flow (noisier loss)")
    ap.add_argument("--num-samples", type=int, default=3, help="flow: trajectories per input (different noise)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", help="default: outputs/head_run_<head>")
    args = ap.parse_args()
    args.epochs = args.epochs or {"mlp": 1000, "flow": 5000}[args.head]
    args.out_dir = args.out_dir or f"outputs/head_run_{args.head}"
    torch.manual_seed(args.seed)

    gt_file, lat_file = torch.load(args.gt), torch.load(args.latents)
    assert [s["sample_token"] for s in gt_file["samples"]] == [s["sample_token"] for s in lat_file["samples"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.stack([s["last_token"] for s in lat_file["samples"]]).to(device)  # (N, 2048)
    y = gt_file["trajectory"].to(device)  # (N, 6, 2)

    tr, va = slice(0, args.train), slice(args.train, args.train + args.val)
    te = slice(args.train + args.val, len(x))
    scenes = [s["scene"] for s in gt_file["samples"]]
    for name, sl in [("train", tr), ("val", va), ("test", te)]:
        print(f"{name:<5} {len(x[sl]):>4} samples  {len(set(scenes[sl]))} scenes")

    if args.head == "mlp":
        head = MLPActionHead().to(device)
    else:
        head = FlowMatchingActionHead(num_samples=args.num_samples).to(device)
        head.fit_normalization(y[tr])

    def predict(latent):
        # Flow head samples from noise; fix the seed so evaluation is repeatable.
        head.eval()
        with torch.no_grad():
            if args.head == "flow":
                return head(latent, generator=torch.Generator(device).manual_seed(0))  # (N, K, 6, 2)
            return head(latent)[:, None]  # (N, 1, 6, 2)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = (float("inf"), None, -1)
    for epoch in range(args.epochs):
        head.train()
        loss = head.loss(x[tr], y[tr])  # full batch
        opt.zero_grad()
        loss.backward()
        opt.step()

        val_ade = metrics(predict(x[va]), y[va])["ADE"]
        if val_ade < best[0]:
            best = (val_ade, {k: v.clone() for k, v in head.state_dict().items()}, epoch)
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  train loss {loss.item():.3f}  val ADE {val_ade:.3f}")

    head.load_state_dict(best[1])
    pred = predict(x)
    mean_traj = y[tr].mean(0)[None, None].expand(len(y), 1, -1, -1)  # baseline: average train trajectory

    results = {
        "head_type": args.head,
        "num_trajectories": pred.shape[1],
        "best_epoch": best[2],
        "split": {"train": args.train, "val": args.val, "test": len(x[te])},
        "head": {n: metrics(pred[sl], y[sl]) for n, sl in [("train", tr), ("val", va), ("test", te)]},
        "baseline_train_mean": {n: metrics(mean_traj[sl], y[sl]) for n, sl in [("val", va), ("test", te)]},
    }
    print(f"\n[{args.head}] best val epoch {best[2]}")
    cols = ["L2_1s", "L2_2s", "L2_3s", "ADE", "FDE", "minADE", "minFDE"]
    print(f"\n[{args.head}] K={pred.shape[1]} trajectories per sample")
    print(f"{'':<26}" + "".join(f"{c:>8}" for c in cols))
    rows = [("head / train", results["head"]["train"]), ("head / val", results["head"]["val"]),
            ("head / test", results["head"]["test"]), ("mean-traj baseline / test", results["baseline_train_mean"]["test"])]
    for name, m in rows:
        print(f"{name:<26}" + "".join(f"{m[c]:>8.2f}" for c in cols))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out / "head.pt")
    torch.save({"pred": pred[te].cpu(), "gt": y[te].cpu(), "samples": gt_file["samples"][te]}, out / "test_preds.pt")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved head, test predictions and metrics -> {out}/")


if __name__ == "__main__":
    main()
