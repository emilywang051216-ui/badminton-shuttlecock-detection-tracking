"""TrackNetV2 — shared library (model, loss, dataset, heatmap utils).

Imported by 03_train_tracknet.py (training), 04_eval_compare.py (metrics) and
05_infer_video.py (inference). Kept framework-light: plain PyTorch, no Ultralytics.

TrackNet idea: feed 3 consecutive frames (9 channels) and regress a Gaussian heatmap whose
peak is the shuttlecock in the *current* (middle) frame. Temporal context is what lets it beat
a single-frame detector on a tiny, fast, motion-blurred object.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

VAL_TAIL_FRAC = 0.12  # mirrors 01_prepare_data.py so TrackNet and YOLO use the same folds


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


def _triple_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class TrackNetV2(nn.Module):
    """Compact VGG16/U-Net-style encoder-decoder. Input 9ch (3 RGB frames) -> 1ch heatmap."""

    def __init__(self, in_ch: int = 9):
        super().__init__()
        self.enc1 = _double_conv(in_ch, 64)     # -> 64
        self.enc2 = _double_conv(64, 128)       # -> 128
        self.enc3 = _triple_conv(128, 256)      # -> 256
        self.enc4 = _triple_conv(256, 512)      # bottleneck -> 512
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = _triple_conv(512 + 256, 256)
        self.dec2 = _double_conv(256 + 128, 128)
        self.dec1 = _double_conv(128 + 64, 64)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)                 # (B,64,H,W)
        e2 = self.enc2(self.pool(e1))     # (B,128,H/2,W/2)
        e3 = self.enc3(self.pool(e2))     # (B,256,H/4,W/4)
        e4 = self.enc4(self.pool(e3))     # (B,512,H/8,W/8)
        d3 = self.dec3(torch.cat([self.up(e4), e3], 1))   # H/4
        d2 = self.dec2(torch.cat([self.up(d3), e2], 1))   # H/2
        d1 = self.dec1(torch.cat([self.up(d2), e1], 1))   # H
        return torch.sigmoid(self.head(d1))               # (B,1,H,W) in [0,1]


def focal_wbce(pred, target, eps: float = 1e-6):
    """TrackNetV2 weighted focal BCE on a soft Gaussian target. Handles the heavy
    background/foreground imbalance without a hand-tuned pos_weight."""
    p = pred.clamp(eps, 1 - eps)
    loss = -((1 - p) ** 2 * target * torch.log(p)
             + p ** 2 * (1 - target) * torch.log(1 - p))
    return loss.mean()


# --------------------------------------------------------------------------- #
# Heatmap helpers (shared by data, eval, inference)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_tracknet(model, loader, device, thresh=0.5, tol_norm=4 / 288):
    """Detection precision/recall/F1 + mean positioning error (px@640) + avg loss on a loader.

    A prediction is a TP if the GT frame has a ball and the decoded peak is within tol_norm of it;
    FP if it fires far from GT or on a ball-absent frame; FN if a ball is missed. Shared by
    training (early-stopping metric) and 04_eval_compare (final report) so they never diverge.
    """
    model.eval()
    tp = fp = fn = tn = 0
    err_sum = err_n = 0.0
    loss_sum = n_batches = 0
    for x, y, gt in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            pred = model(x)
        loss_sum += float(focal_wbce(pred.float(), y)); n_batches += 1
        hm = pred.squeeze(1).float().cpu().numpy()
        for b in range(hm.shape[0]):
            gtx, gty, has = gt[b].tolist()
            dec = heatmap_to_xy(hm[b], thresh)
            if has > 0.5:
                if dec is None:
                    fn += 1
                else:
                    d = float(np.hypot(dec[0] - gtx, dec[1] - gty))
                    if d <= tol_norm:
                        tp += 1; err_sum += d * 640; err_n += 1
                    else:
                        fp += 1
            else:
                if dec is None:
                    tn += 1
                else:
                    fp += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"loss": loss_sum / max(1, n_batches), "precision": prec, "recall": rec, "f1": f1,
            "pos_err_px640": err_sum / err_n if err_n else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def make_heatmap(cx, cy, res: int, sigma: float) -> np.ndarray:
    """Gaussian heatmap (res,res), peak 1.0 at normalized (cx,cy). All-zeros if no ball."""
    hm = np.zeros((res, res), np.float32)
    if cx is None or cy is None:
        return hm
    x, y = cx * res, cy * res
    ax = np.arange(res, dtype=np.float32)
    gx = np.exp(-((ax - x) ** 2) / (2 * sigma ** 2))
    gy = np.exp(-((ax - y) ** 2) / (2 * sigma ** 2))
    return np.outer(gy, gx).astype(np.float32)


def heatmap_to_xy(hm: np.ndarray, thresh: float = 0.5):
    """Decode a heatmap -> (cx, cy, peak) normalized, or None if peak below threshold."""
    peak = float(hm.max())
    if peak < thresh:
        return None
    y, x = np.unravel_index(int(hm.argmax()), hm.shape)
    res = hm.shape[0]
    return x / res, y / res, peak


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def load_triples(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def split_triples(rows, test_video: str, split: str):
    """Subset of triples for a fold. LOVO (test_video in {1,2,3}): test = held-out video;
    train/val = the other two videos with a contiguous tail as val. Deployment
    (test_video == 'all'): train/val from ALL videos' tails, no held-out test (test = val)."""
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video"]].append(r)
    for v in by_video:
        by_video[v].sort(key=lambda r: int(r["frame"]))

    if test_video == "all":
        out = []
        for v, frames in by_video.items():
            cut = int(len(frames) * (1 - VAL_TAIL_FRAC))
            out += frames[:cut] if split == "train" else frames[cut:]   # test falls to tail (=val)
        return out

    if split == "test":
        return by_video[test_video]

    out = []
    for v, frames in by_video.items():
        if v == test_video:
            continue
        cut = int(len(frames) * (1 - VAL_TAIL_FRAC))
        out += frames[:cut] if split == "train" else frames[cut:]
    return out


class TrackNetDataset(Dataset):
    def __init__(self, csv_path, test_video, split, res=288, sigma=3.0, cache=False):
        self.rows = split_triples(load_triples(csv_path), test_video, split)
        self.res = res
        self.sigma = sigma
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.rows)

    def _read(self, path):
        if self.cache is not None and path in self.cache:
            return self.cache[path]
        img = cv2.imread(path)                                   # BGR uint8
        img = cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_AREA)
        if self.cache is not None:
            self.cache[path] = img
        return img

    def __getitem__(self, i):
        r = self.rows[i]
        frames = [self._read(r["prev"]), self._read(r["curr"]), self._read(r["next"])]
        x = np.concatenate(frames, axis=2).astype(np.float32) / 255.0   # (res,res,9)
        x = torch.from_numpy(x).permute(2, 0, 1)                        # (9,res,res)
        cx = float(r["cx"]) if r["cx"] else None
        cy = float(r["cy"]) if r["cy"] else None
        hm = make_heatmap(cx, cy, self.res, self.sigma)                # (res,res)
        y = torch.from_numpy(hm).unsqueeze(0)                          # (1,res,res)
        has_ball = int(r["has_ball"])
        gt = torch.tensor([cx if cx is not None else -1.0,
                           cy if cy is not None else -1.0,
                           float(has_ball)])
        return x, y, gt
