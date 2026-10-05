"""M4 — Inference / demo pipeline: detect -> track -> trajectory -> landing heatmap -> mp4.

Works with either detector, on a real video or a replayed dataset video:
  # YOLO on the held-out video 3 (never seen in lovo_test3 training):
  python scripts/05_infer_video.py --model yolo --source dataset:3
  # TrackNet on a real clip:
  python scripts/05_infer_video.py --model tracknet --source path/to/match.mp4

Outputs (in outputs/):
  demo_<model>.mp4   — frames with the ball marked + a fading flight-trail
  heatmap_<model>.png — landing/position heatmap accumulated over the whole clip

Temporal-consistency gating (a shuttle moves smoothly) rejects physically-impossible jumps,
which doubles as a false-positive filter — the practical lever for robustness on new clips.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from common import OUTPUTS_DIR, MODELS_DIR, video_frame_paths


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
def frame_source(source: str, max_frames: int):
    """Yield BGR frames from a video file or a replayed dataset video (source='dataset:N')."""
    if source.startswith("dataset:"):
        paths = video_frame_paths(source.split(":", 1)[1])
        if max_frames:
            paths = paths[:max_frames]
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                yield img
    else:
        cap = cv2.VideoCapture(source)
        n = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield img
            n += 1
            if max_frames and n >= max_frames:
                break
        cap.release()


# --------------------------------------------------------------------------- #
# Detectors — each returns (cx, cy, score) normalized to [0,1], or None
# --------------------------------------------------------------------------- #
class YoloDetector:
    def __init__(self, weights, conf, device):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device

    def __call__(self, frame):
        r = self.model.predict(frame, conf=self.conf, device=self.device,
                               verbose=False, imgsz=640)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        # single class -> take the highest-confidence box
        confs = r.boxes.conf.cpu().numpy()
        i = int(confs.argmax())
        x1, y1, x2, y2 = r.boxes.xyxyn[i].cpu().numpy()
        return (x1 + x2) / 2, (y1 + y2) / 2, float(confs[i])


class TrackNetDetector:
    """Rolling 3-frame window -> heatmap peak. Emits a detection for the middle frame,
    so the reported position lags the newest frame by one (fine for offline demo)."""

    def __init__(self, weights, thresh, device):
        import torch
        from tracknet import TrackNetV2, heatmap_to_xy
        self.torch = torch
        self.heatmap_to_xy = heatmap_to_xy
        ckpt = torch.load(str(weights), map_location=device, weights_only=False)
        self.res = ckpt["res"]
        self.thresh = thresh
        self.device = device
        self.model = TrackNetV2(in_ch=9).to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.buf = deque(maxlen=3)

    def __call__(self, frame):
        small = cv2.resize(frame, (self.res, self.res), interpolation=cv2.INTER_AREA)
        self.buf.append(small)
        if len(self.buf) < 3:
            return None
        x = np.concatenate(list(self.buf), axis=2).astype(np.float32) / 255.0
        t = self.torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with self.torch.no_grad(), self.torch.amp.autocast("cuda"):
            hm = self.model(t)
        dec = self.heatmap_to_xy(hm.squeeze().float().cpu().numpy(), self.thresh)
        return dec  # (cx, cy, peak) or None


# --------------------------------------------------------------------------- #
# Tracker with temporal-consistency gating
# --------------------------------------------------------------------------- #
class GatedTracker:
    def __init__(self, gate_px=120, trail=40, w=640, h=640):
        self.gate = gate_px / max(w, h)   # normalized gate
        self.trail = deque(maxlen=trail)
        self.last = None
        self.vel = (0.0, 0.0)
        self.misses = 0

    def update(self, det):
        """Return accepted (cx,cy) or None. Gates on distance from the predicted position."""
        if det is None:
            self.misses += 1
            if self.misses > 5:
                self.last = None; self.vel = (0.0, 0.0)
            return None
        cx, cy = det[0], det[1]
        if self.last is not None:
            px, py = self.last[0] + self.vel[0], self.last[1] + self.vel[1]
            if np.hypot(cx - px, cy - py) > self.gate + 0.02:
                self.misses += 1                 # physically implausible jump -> reject
                if self.misses > 5:
                    self.last = None; self.vel = (0.0, 0.0)
                return None
            self.vel = (cx - self.last[0], cy - self.last[1])
        self.last = (cx, cy)
        self.misses = 0
        self.trail.append((cx, cy))
        return (cx, cy)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def draw_overlay(frame, tracker, W, H):
    pts = list(tracker.trail)
    for i in range(1, len(pts)):
        a = int(255 * i / len(pts))                       # fade older segments
        p0 = (int(pts[i - 1][0] * W), int(pts[i - 1][1] * H))
        p1 = (int(pts[i][0] * W), int(pts[i][1] * H))
        cv2.line(frame, p0, p1, (0, a, 255), 2)
    if tracker.last is not None:
        c = (int(tracker.last[0] * W), int(tracker.last[1] * H))
        cv2.circle(frame, c, 7, (0, 255, 0), 2)
    return frame


def render_heatmap(accum, bg):
    a = accum.copy()
    if a.max() > 0:
        a = a / a.max()
    a = np.power(a, 0.5)                                    # gamma for visibility
    cm = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bg, 0.55, cm, 0.65, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["yolo", "tracknet"], default="yolo")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--source", default="dataset:3", help="video path | dir | 'dataset:N'")
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--thresh", type=float, default=0.5, help="TrackNet heatmap threshold")
    ap.add_argument("--gate-px", type=float, default=120, help="max plausible ball jump per frame (px@640)")
    ap.add_argument("--trail", type=int, default=40, help="trajectory trail length (frames)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames (0 = all)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.weights is None:
        args.weights = (MODELS_DIR / "yolo_lovo_test3/weights/best.pt" if args.model == "yolo"
                        else MODELS_DIR / "tracknet_test3/best.pt")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(OUTPUTS_DIR / f"demo_{args.model}.mp4")
    # heatmap name follows the output video so different clips/models never clobber each other
    heat_path = str(Path(out_path).with_name(Path(out_path).stem + "_heatmap.png"))

    det = (YoloDetector(args.weights, args.conf, args.device) if args.model == "yolo"
           else TrackNetDetector(args.weights, args.thresh, args.device))

    writer = None
    tracker = None
    accum = None
    bg = None
    W = H = None
    n_frames = n_det = 0

    for frame in frame_source(args.source, args.max_frames):
        if writer is None:
            H, W = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (W, H))
            tracker = GatedTracker(args.gate_px, args.trail, W, H)
            accum = np.zeros((H, W), np.float32)
            bg = frame.copy()
        n_frames += 1
        acc = tracker.update(det(frame))
        if acc is not None:
            n_det += 1
            px, py = int(acc[0] * W), int(acc[1] * H)
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(accum, (px, py), 6, 1.0, -1)     # splat for a smooth heatmap
        writer.write(draw_overlay(frame, tracker, W, H))

    if writer is None:
        raise SystemExit(f"no frames read from source '{args.source}'")
    writer.release()
    accum = cv2.GaussianBlur(accum, (0, 0), 8)
    cv2.imwrite(heat_path, render_heatmap(accum, bg))
    rate = 100 * n_det / n_frames if n_frames else 0
    print(f"[done] {args.model}: {n_frames} frames, {n_det} accepted detections ({rate:.1f}%)")
    print(f"       video   -> {out_path}")
    print(f"       heatmap -> {heat_path}")


if __name__ == "__main__":
    main()
