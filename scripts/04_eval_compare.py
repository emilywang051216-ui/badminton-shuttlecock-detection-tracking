"""M2/M3 — Evaluate trained models and build the comparison story.

Discovers whatever is trained under models/ and produces:
  outputs/eval_yolo.csv        — YOLO mAP50 / mAP50-95 / P / R per split (random + LOVO folds)
  outputs/eval_tracknet.csv    — TrackNet P / R / F1 / positioning-error per LOVO fold
  outputs/gap_chart.png        — the headline: leaky-random mAP50 vs honest cross-video mAP50
  outputs/yolo_curves.png      — YOLO mAP50 training curves (from each run's results.csv)
  outputs/tracknet_curves.png  — TrackNet val-F1 curves (from each run's history.csv)

Safe to run before training finishes — it reports only what exists. Uses plain matplotlib
(Ultralytics' own plotting segfaults in this env; ours is the render path proven to work).

Run:  python scripts/04_eval_compare.py
"""
from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import MODELS_DIR, SPLITS_DIR, OUTPUTS_DIR, TRACKNET_DIR

LOVO_SPLITS = ["lovo_test1", "lovo_test2", "lovo_test3"]


# --------------------------------------------------------------------------- #
# YOLO
# --------------------------------------------------------------------------- #
def eval_yolo():
    rows = []
    splits = ["random"] + LOVO_SPLITS
    weights = {s: MODELS_DIR / f"yolo_{s}" / "weights" / "best.pt" for s in splits}
    present = {s: w for s, w in weights.items() if w.exists()}
    if not present:
        print("[yolo] no trained models found (models/yolo_*/weights/best.pt) — skipping")
        return rows
    from ultralytics import YOLO
    for s, w in present.items():
        data_yaml = SPLITS_DIR / f"{s}.yaml"
        print(f"[yolo] evaluating {s} on its TEST split ...")
        m = YOLO(str(w)).val(data=str(data_yaml), split="test", plots=False, verbose=False)
        rows.append({"model": "YOLOv11", "split": s,
                     "mAP50": round(float(m.box.map50), 4),
                     "mAP50_95": round(float(m.box.map), 4),
                     "precision": round(float(m.box.mp), 4),
                     "recall": round(float(m.box.mr), 4)})
    _write_csv(OUTPUTS_DIR / "eval_yolo.csv", rows)
    return rows


# --------------------------------------------------------------------------- #
# TrackNet
# --------------------------------------------------------------------------- #
def eval_tracknet():
    rows = []
    import torch
    from torch.utils.data import DataLoader
    from tracknet import TrackNetV2, TrackNetDataset, evaluate_tracknet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_path = TRACKNET_DIR / "triples.csv"
    any_model = False
    for v in ["1", "2", "3"]:
        w = MODELS_DIR / f"tracknet_test{v}" / "best.pt"
        if not w.exists():
            continue
        any_model = True
        ckpt = torch.load(str(w), map_location=device, weights_only=False)
        model = TrackNetV2(in_ch=9).to(device)
        model.load_state_dict(ckpt["model"])
        ds = TrackNetDataset(csv_path, v, "test", ckpt["res"], ckpt["sigma"])
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        print(f"[tracknet] evaluating fold test-video={v} ({len(ds)} frames) ...")
        m = evaluate_tracknet(model, dl, device, tol_norm=4 / ckpt["res"])
        rows.append({"model": "TrackNetV2", "split": f"lovo_test{v}",
                     "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
                     "f1": round(m["f1"], 4), "pos_err_px640": round(m["pos_err_px640"], 2)})
    if not any_model:
        print("[tracknet] no trained models found (models/tracknet_test*/best.pt) — skipping")
    else:
        _write_csv(OUTPUTS_DIR / "eval_tracknet.csv", rows)
    return rows


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def gap_chart(yolo_rows):
    """The headline: leaky random split vs honest cross-video (LOVO) mAP50."""
    by = {r["split"]: r["mAP50"] for r in yolo_rows}
    lovo_vals = [by[s] for s in LOVO_SPLITS if s in by]
    if "random" not in by or not lovo_vals:
        print("[chart] need both random + >=1 LOVO YOLO model for the gap chart — skipping")
        return
    lovo_avg = sum(lovo_vals) / len(lovo_vals)
    labels = ["random\n(leaky)"] + [s.replace("lovo_", "") for s in LOVO_SPLITS if s in by] + ["LOVO\navg"]
    vals = [by["random"]] + lovo_vals + [lovo_avg]
    colors = ["#c0392b"] + ["#2980b9"] * len(lovo_vals) + ["#16a085"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("mAP@0.5 (test)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("Why the split matters: same-video leakage inflates mAP\n"
                 "(random) vs true generalization to an unseen video (LOVO)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "gap_chart.png", dpi=130)
    plt.close(fig)
    print(f"[chart] gap_chart.png  (random={by['random']:.3f} vs LOVO-avg={lovo_avg:.3f})")


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def yolo_curves():
    runs = [(s, MODELS_DIR / f"yolo_{s}" / "results.csv")
            for s in ["random"] + LOVO_SPLITS]
    runs = [(s, p) for s, p in runs if p.exists()]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s, p in runs:
        rows = _read_csv(p)
        col = next((c for c in rows[0] if "mAP50(B)" in c and "50-95" not in c), None)
        if col is None:
            continue
        ep = [float(r["epoch"]) for r in rows]
        y = [float(r[col]) for r in rows]
        ax.plot(ep, y, label=s, linewidth=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("val mAP@0.5"); ax.legend()
    ax.set_title("YOLOv11 training curves")
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "yolo_curves.png", dpi=130); plt.close(fig)
    print("[chart] yolo_curves.png")


def tracknet_curves():
    runs = [(v, MODELS_DIR / f"tracknet_test{v}" / "history.csv") for v in ["1", "2", "3"]]
    runs = [(v, p) for v, p in runs if p.exists()]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for v, p in runs:
        rows = _read_csv(p)
        ep = [float(r["epoch"]) for r in rows]
        y = [float(r["val_f1"]) for r in rows]
        ax.plot(ep, y, label=f"test-video {v}", linewidth=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("val F1"); ax.legend()
    ax.set_title("TrackNetV2 validation F1")
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "tracknet_curves.png", dpi=130); plt.close(fig)
    print("[chart] tracknet_curves.png")


# --------------------------------------------------------------------------- #
def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[csv] wrote {path.name}")


def _print_table(title, rows):
    if not rows:
        return
    print(f"\n=== {title} ===")
    cols = list(rows[0].keys())
    print("  ".join(f"{c:>12}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>12}" for c in cols))


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    yolo_rows = eval_yolo()
    tn_rows = eval_tracknet()
    gap_chart(yolo_rows)
    yolo_curves()
    tracknet_curves()
    _print_table("YOLOv11 (per split)", yolo_rows)
    _print_table("TrackNetV2 (per LOVO fold)", tn_rows)
    if not yolo_rows and not tn_rows:
        print("\n[info] nothing trained yet — run 02_train_yolo.py / 03_train_tracknet.py first.")


if __name__ == "__main__":
    main()
