"""M1 — Data preparation.

Does four things:
  1. Extract the Roboflow zip into data/raw/ (idempotent).
  2. Write dataset yamls for two evaluation regimes:
       - random.yaml     : Roboflow's original split (leaky control — all 3 videos in every split)
       - lovo_testN.yaml : leave-one-video-out (train on 2 videos, test on the held-out one)
     LOVO is the honest "does it generalize to a new clip" measure; random is the inflated control.
  3. Build data/tracknet/triples.csv — the 3-consecutive-frame windows TrackNet needs, each tagged
     with the ball location and the LOVO fold it belongs to.
  4. QC: draw ground-truth boxes back onto 24 random frames so we can eyeball label alignment,
     and assert the LOVO splits are video-disjoint.

Run:  python scripts/01_prepare_data.py
"""
from __future__ import annotations

import csv
import json
import zipfile
from collections import defaultdict

import cv2

from common import (
    ZIP_PATH, RAW_DIR, SPLITS_DIR, TRACKNET_DIR, OUTPUTS_DIR,
    VIDEOS, CLASS_NAMES, RAW_SPLITS, iter_raw_frames,
)

VAL_TAIL_FRAC = 0.12   # last 12% (contiguous) of each training video -> validation, to limit
                       # near-duplicate leakage between train and val within LOVO folds.


# --------------------------------------------------------------------------- #
# 1. Extract
# --------------------------------------------------------------------------- #
def extract_zip():
    already = (RAW_DIR / "train" / "images").exists()
    n_imgs = len(list((RAW_DIR / "train" / "images").glob("*.jpg"))) if already else 0
    if already and n_imgs > 5000:
        print(f"[extract] data/raw already populated ({n_imgs} train imgs) — skipping.")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[extract] unzipping {ZIP_PATH.name} -> {RAW_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW_DIR)
    print("[extract] done.")


# --------------------------------------------------------------------------- #
# 2. Video-level (LOVO) splits + random control
# --------------------------------------------------------------------------- #
def build_index():
    """Group every frame by video; return {video: [frame dicts sorted by frame idx]}."""
    by_video = defaultdict(list)
    for rec in iter_raw_frames():
        by_video[rec["video"]].append(rec)
    for v in by_video:
        by_video[v].sort(key=lambda r: r["frame"])
    return by_video


def _write_list(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(str(r["image"].resolve()) + "\n")


def _write_yaml(path, train_txt, val_txt, test_txt):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"train: {train_txt.resolve().as_posix()}\n")
        f.write(f"val: {val_txt.resolve().as_posix()}\n")
        f.write(f"test: {test_txt.resolve().as_posix()}\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")


def write_random_yaml():
    """Roboflow's original leaky split — point straight at the raw folders."""
    path = SPLITS_DIR / "random.yaml"
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"train: {(RAW_DIR / 'train' / 'images').resolve().as_posix()}\n")
        f.write(f"val: {(RAW_DIR / 'valid' / 'images').resolve().as_posix()}\n")
        f.write(f"test: {(RAW_DIR / 'test' / 'images').resolve().as_posix()}\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")
    print(f"[split] wrote {path.name} (leaky control: all 3 videos in every split)")


def write_lovo_yamls(by_video):
    """For each held-out video: train/val from the other two, test = held-out video."""
    summary = {}
    for held in VIDEOS:
        train_recs, val_recs = [], []
        for v in VIDEOS:
            if v == held:
                continue
            frames = by_video[v]
            cut = int(len(frames) * (1 - VAL_TAIL_FRAC))
            train_recs += frames[:cut]      # earlier (contiguous) frames -> train
            val_recs += frames[cut:]        # tail (contiguous) frames    -> val
        test_recs = by_video[held]

        train_txt = SPLITS_DIR / f"lovo_test{held}_train.txt"
        val_txt = SPLITS_DIR / f"lovo_test{held}_val.txt"
        test_txt = SPLITS_DIR / f"lovo_test{held}_test.txt"
        _write_list(train_txt, train_recs)
        _write_list(val_txt, val_recs)
        _write_list(test_txt, test_recs)
        _write_yaml(SPLITS_DIR / f"lovo_test{held}.yaml", train_txt, val_txt, test_txt)

        summary[held] = dict(train=len(train_recs), val=len(val_recs), test=len(test_recs))
        print(f"[split] lovo_test{held}: train={len(train_recs)} val={len(val_recs)} "
              f"test={len(test_recs)}  (test video = {held})")
    return summary


def write_all_yaml(by_video):
    """DEPLOYMENT split: train on ALL 3 videos (max diversity), tail of each video as val for
    early stopping. No held-out test video — this model is for running on a genuinely new clip,
    not for measuring generalization (that's what the LOVO folds are for)."""
    train_recs, val_recs = [], []
    for v in VIDEOS:
        frames = by_video[v]
        cut = int(len(frames) * (1 - VAL_TAIL_FRAC))
        train_recs += frames[:cut]
        val_recs += frames[cut:]
    train_txt = SPLITS_DIR / "all_train.txt"
    val_txt = SPLITS_DIR / "all_val.txt"
    _write_list(train_txt, train_recs)
    _write_list(val_txt, val_recs)
    _write_yaml(SPLITS_DIR / "all.yaml", train_txt, val_txt, val_txt)  # test = val (no held-out)
    print(f"[split] all (deployment): train={len(train_recs)} val={len(val_recs)} "
          f"(all 3 videos; test=val)")
    return dict(train=len(train_recs), val=len(val_recs))


# --------------------------------------------------------------------------- #
# 3. TrackNet triples
# --------------------------------------------------------------------------- #
def build_tracknet_triples(by_video):
    """A sample = 3 consecutive frames (prev, curr, next). Label = ball (cx,cy) of curr frame.

    Fold assignment mirrors LOVO: a triple whose video == held-out video is 'test' for fold N.
    We store the fold membership as three flags so the loader can filter per experiment.
    """
    TRACKNET_DIR.mkdir(parents=True, exist_ok=True)
    out = TRACKNET_DIR / "triples.csv"
    n_total = 0
    per_video = defaultdict(int)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "prev", "curr", "next",
                    "cx", "cy", "has_ball",
                    "test_fold1", "test_fold2", "test_fold3"])
        for v in VIDEOS:
            frames = by_video[v]
            idx = {r["frame"]: r for r in frames}
            for r in frames:
                fi = r["frame"]
                if (fi - 1) not in idx or (fi + 1) not in idx:
                    continue  # need both neighbors
                prev, nxt = idx[fi - 1], idx[fi + 1]
                w.writerow([
                    v, fi,
                    prev["image"].resolve().as_posix(),
                    r["image"].resolve().as_posix(),
                    nxt["image"].resolve().as_posix(),
                    "" if r["cx"] is None else f"{r['cx']:.6f}",
                    "" if r["cy"] is None else f"{r['cy']:.6f}",
                    int(r["has_ball"]),
                    int(v == "1"), int(v == "2"), int(v == "3"),
                ])
                n_total += 1
                per_video[v] += 1
    print(f"[tracknet] wrote {out.name}: {n_total} triples  "
          f"(per video: {dict(per_video)})")
    return n_total


# --------------------------------------------------------------------------- #
# 4. QC
# --------------------------------------------------------------------------- #
def qc_draw_boxes(by_video, n=24):
    qc_dir = OUTPUTS_DIR / "qc_labels"
    qc_dir.mkdir(parents=True, exist_ok=True)
    # spread samples across videos, only ball-present frames
    picked = []
    for v in VIDEOS:
        ball_frames = [r for r in by_video[v] if r["has_ball"]]
        step = max(1, len(ball_frames) // (n // len(VIDEOS)))
        picked += ball_frames[::step][: n // len(VIDEOS)]
    for r in picked:
        img = cv2.imread(str(r["image"]))
        if img is None:
            continue
        H, W = img.shape[:2]
        cx, cy, bw, bh = r["cx"] * W, r["cy"] * H, r["w"] * W, r["h"] * H
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        # draw an enlarged marker so the tiny (~5x9px) box is visible
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.circle(img, (int(cx), int(cy)), 14, (0, 255, 0), 2)
        cv2.imwrite(str(qc_dir / f"v{r['video']}_f{r['frame']}.jpg"), img)
    print(f"[qc] wrote {len(picked)} annotated frames -> {qc_dir} (eyeball the green circles hit the ball)")


def assert_disjoint(summary):
    # LOVO test video must not appear in its own train/val — guaranteed by construction, but verify
    # by re-reading one fold's train list and confirming no held-out-video frames leaked in.
    for held in VIDEOS:
        train_txt = SPLITS_DIR / f"lovo_test{held}_train.txt"
        bad = 0
        for line in train_txt.read_text().splitlines():
            if f"video_label_{held}_" in line:
                bad += 1
        status = "OK" if bad == 0 else f"LEAK({bad})"
        print(f"[verify] lovo_test{held}: held-out frames in train list = {bad}  -> {status}")
        assert bad == 0, f"LEAK: video {held} frames found in its own training list"


def main():
    extract_zip()
    by_video = build_index()
    total = sum(len(v) for v in by_video.values())
    print(f"[index] {total} frames  (per video: "
          f"{ {v: len(by_video[v]) for v in VIDEOS} })")

    write_random_yaml()
    summary = write_lovo_yamls(by_video)
    all_summary = write_all_yaml(by_video)
    n_triples = build_tracknet_triples(by_video)
    qc_draw_boxes(by_video)
    assert_disjoint(summary)

    (OUTPUTS_DIR / "prep_summary.json").write_text(json.dumps(
        {"total_frames": total,
         "per_video": {v: len(by_video[v]) for v in VIDEOS},
         "lovo": summary, "all": all_summary, "tracknet_triples": n_triples}, indent=2))
    print("\n[done] data preparation complete. Summary -> outputs/prep_summary.json")


if __name__ == "__main__":
    main()
