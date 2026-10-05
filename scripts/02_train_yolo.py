"""M2 — Train a YOLOv11 shuttlecock detector on a chosen split.

Splits (data/splits/*.yaml, produced by 01_prepare_data.py):
  random      : Roboflow's original split (leaky control — inflated metrics)
  lovo_test1  : train on videos {2,3}, test on video 1   (honest generalization)
  lovo_test2  : train on videos {1,3}, test on video 2
  lovo_test3  : train on videos {1,2}, test on video 3   (headline fold)

Augmentation is configured in configs/yolo_smallobj.yaml. Blur robustness (fast shuttle ->
motion blur) comes from Ultralytics' default Albumentations Blur/MedianBlur, applied
automatically because albumentations is installed.

Run:  python scripts/02_train_yolo.py --split lovo_test3
      python scripts/02_train_yolo.py --split random --epochs 60
      python scripts/02_train_yolo.py --split lovo_test3 --fraction 0.03 --epochs 1   # fast smoke test
"""
from __future__ import annotations

import argparse
import yaml

from common import CONFIGS_DIR, SPLITS_DIR, MODELS_DIR, default_workers, default_plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="lovo_test3",
                    help="random | lovo_test1 | lovo_test2 | lovo_test3")
    ap.add_argument("--config", default=str(CONFIGS_DIR / "yolo_smallobj.yaml"))
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--batch", type=int, default=None, help="override config batch")
    ap.add_argument("--workers", type=int, default=None,
                    help="dataloader workers (default: 0 on Windows, up to 8 on Linux)")
    ap.add_argument("--cache", default=None, help="ultralytics cache: false|ram|disk (default from config)")
    ap.add_argument("--plots", dest="plots", action="store_true", default=None,
                    help="force Ultralytics plots on (default: off on Windows, on elsewhere)")
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set (for fast smoke tests)")
    ap.add_argument("--device", default="0", help="cuda id, or 'cpu'")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.cache is not None:
        cfg["cache"] = args.cache
    # workers/plots pick a Windows-safe or Linux-fast default unless explicitly overridden.
    cfg["workers"] = args.workers if args.workers is not None else default_workers()
    plots = args.plots if args.plots is not None else default_plots()

    data_yaml = SPLITS_DIR / f"{args.split}.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"missing {data_yaml} — run scripts/01_prepare_data.py first")

    from ultralytics import YOLO

    model = YOLO(cfg.pop("model"))
    run_name = f"yolo_{args.split}"
    print(f"[cfg] split={args.split} workers={cfg['workers']} cache={cfg['cache']} "
          f"plots={plots} batch={cfg['batch']} epochs={cfg['epochs']}")

    # plots: Ultralytics' internal plotting (plot_images/plot_labels) segfaults under
    # Anaconda+numpy-1.x on Windows, so it defaults off there; on Linux it's kept on. Either way
    # per-epoch metrics go to results.csv and 04_eval_compare.py builds our own comparison charts.
    # Everything left in cfg is a valid train() kwarg (imgsz/epochs/batch/aug knobs/etc.)
    model.train(
        data=str(data_yaml),
        project=str(MODELS_DIR),
        name=run_name,
        exist_ok=True,
        device=args.device,
        fraction=args.fraction,
        plots=plots,
        **cfg,
    )

    # Honest number: evaluate the best weights on the held-out TEST split.
    print(f"\n[eval] validating best weights on TEST split of {args.split} ...")
    metrics = model.val(data=str(data_yaml), split="test", device=args.device, plots=plots)
    print(f"[eval] {args.split}  mAP50={metrics.box.map50:.4f}  "
          f"mAP50-95={metrics.box.map:.4f}  "
          f"P={metrics.box.mp:.4f}  R={metrics.box.mr:.4f}")
    print(f"[done] weights -> {MODELS_DIR / run_name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
