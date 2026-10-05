"""M3 — Train TrackNetV2 on a leave-one-video-out fold (same folds as YOLO).

The held-out video (--test-video) is never seen in training; train/val come from the other two,
so the numbers are directly comparable to the YOLO lovo_test{N} run.

Run:  python scripts/03_train_tracknet.py --test-video 3
      python scripts/03_train_tracknet.py --test-video 3 --fraction 0.05 --epochs 2   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import time

import torch
from torch.utils.data import DataLoader

from common import TRACKNET_DIR, MODELS_DIR, default_workers
from tracknet import TrackNetV2, TrackNetDataset, focal_wbce, evaluate_tracknet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-video", default="3", choices=["1", "2", "3", "all"],
                    help="held-out video (LOVO fold); 'all' = deployment model on all 3 videos")
    ap.add_argument("--res", type=int, default=288)
    ap.add_argument("--sigma", type=float, default=3.0, help="Gaussian heatmap sigma (output px)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--thresh", type=float, default=0.5, help="heatmap peak threshold for 'ball present'")
    ap.add_argument("--tol-px", type=float, default=4.0, help="TP distance tolerance in output px")
    ap.add_argument("--cache", action="store_true", help="cache resized frames in RAM (~2GB, faster)")
    ap.add_argument("--workers", type=int, default=None,
                    help="dataloader workers (default: 0 on Windows, up to 8 on Linux)")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set (fast smoke test)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    workers = args.workers if args.workers is not None else default_workers()

    device = args.device if torch.cuda.is_available() else "cpu"
    csv_path = TRACKNET_DIR / "triples.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path} — run scripts/01_prepare_data.py first")

    tr = TrackNetDataset(csv_path, args.test_video, "train", args.res, args.sigma, args.cache)
    va = TrackNetDataset(csv_path, args.test_video, "val", args.res, args.sigma, args.cache)
    if args.fraction < 1.0:
        keep = max(1, int(len(tr.rows) * args.fraction))
        tr.rows = tr.rows[:keep]
        va.rows = va.rows[: max(1, int(len(va.rows) * args.fraction))]
    print(f"[data] fold test-video={args.test_video}  train={len(tr)}  val={len(va)}  "
          f"res={args.res} sigma={args.sigma}  workers={workers}")

    # workers=0 on Windows (spawn hangs/segfaults with this stack); parallel on Linux.
    # Note: with the RAM frame cache (--cache) and workers>0 each worker builds its own copy,
    # so on Linux prefer either --cache with --workers 0, or workers>0 without --cache.
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=workers, pin_memory=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=workers, pin_memory=True)

    model = TrackNetV2(in_ch=9).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    tol_norm = args.tol_px / args.res

    out_dir = MODELS_DIR / ("tracknet_all" if args.test_video == "all"
                            else f"tracknet_test{args.test_video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "history.csv"
    hist_f = open(hist_path, "w", newline="", encoding="utf-8")
    hist_w = csv.writer(hist_f)
    hist_w.writerow(["epoch", "train_loss", "val_loss", "val_precision", "val_recall",
                     "val_f1", "val_pos_err_px640"])

    best_f1 = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run = 0.0
        for x, y, _ in tl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                pred = model(x)
            loss = focal_wbce(pred.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += float(loss)
        sched.step()
        train_loss = run / max(1, len(tl))
        m = evaluate_tracknet(model, vl, device, args.thresh, tol_norm)
        hist_w.writerow([ep, f"{train_loss:.5f}", f"{m['loss']:.5f}", f"{m['precision']:.4f}",
                         f"{m['recall']:.4f}", f"{m['f1']:.4f}", f"{m['pos_err_px640']:.2f}"])
        hist_f.flush()
        print(f"[ep {ep:03d}] train_loss={train_loss:.4f}  val_loss={m['loss']:.4f}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  "
              f"posErr={m['pos_err_px640']:.1f}px  ({time.time()-t0:.0f}s)")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save({"model": model.state_dict(), "res": args.res, "sigma": args.sigma,
                        "test_video": args.test_video, "epoch": ep, "val_f1": best_f1},
                       out_dir / "best.pt")

    hist_f.close()
    print(f"[done] best val F1={best_f1:.3f}  weights -> {out_dir/'best.pt'}  history -> {hist_path}")


if __name__ == "__main__":
    main()
