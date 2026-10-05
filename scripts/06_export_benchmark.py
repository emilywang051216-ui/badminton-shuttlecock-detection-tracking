"""M5 — Export the YOLO detector to ONNX and benchmark inference speed.

This is the honest "acceleration" story for the slides: the fast path is NOT hand-written
C++/Rust, it's exporting to the ONNX Runtime (a C++ inference engine) and using FP16 — the
model's heavy compute already runs as compiled CUDA kernels under PyTorch.

Reports FPS for: PyTorch FP32, PyTorch FP16, ONNX Runtime — averaged over real frames.

Run:  python scripts/06_export_benchmark.py --weights models/yolo_lovo_test3/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from common import MODELS_DIR, OUTPUTS_DIR, video_frame_paths


def load_frames(source, n):
    if source.startswith("dataset:"):
        import cv2
        paths = video_frame_paths(source.split(":", 1)[1])[:n]
        return [cv2.imread(str(p)) for p in paths]
    import cv2
    cap = cv2.VideoCapture(source)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def bench(model, frames, device, half=False, warmup=10):
    # warmup (kernels/caches)
    for f in frames[:warmup]:
        model.predict(f, device=device, half=half, verbose=False, imgsz=640)
    t0 = time.time()
    for f in frames:
        model.predict(f, device=device, half=half, verbose=False, imgsz=640)
    dt = time.time() - t0
    return len(frames) / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(MODELS_DIR / "yolo_lovo_test3/weights/best.pt"))
    ap.add_argument("--source", default="dataset:3")
    ap.add_argument("--n", type=int, default=200, help="frames to benchmark over")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from ultralytics import YOLO
    frames = load_frames(args.source, args.n)
    if not frames:
        raise SystemExit(f"no frames from '{args.source}'")
    print(f"[bench] {len(frames)} frames @640, device={args.device}")

    results = []

    pt = YOLO(args.weights)
    fps_fp32 = bench(pt, frames, args.device, half=False)
    results.append(("PyTorch FP32", fps_fp32))
    print(f"[bench] PyTorch FP32 : {fps_fp32:6.1f} FPS")

    fps_fp16 = bench(pt, frames, args.device, half=True)
    results.append(("PyTorch FP16", fps_fp16))
    print(f"[bench] PyTorch FP16 : {fps_fp16:6.1f} FPS")

    # Export to ONNX (auto-installs onnx/onnxslim if missing) and benchmark via ONNX Runtime.
    # onnxruntime-gpu's CUDA provider needs its own CUDA/cuDNN on PATH and often fails to bind
    # here; fall back to the CPU provider so we still get the artifact + a portability number.
    try:
        onnx_path = pt.export(format="onnx", imgsz=640, simplify=True)
        onnx_model = YOLO(str(onnx_path))
        for onnx_dev in [args.device, "cpu"]:
            try:
                fps_onnx = bench(onnx_model, frames, onnx_dev)
                results.append((f"ONNX Runtime ({onnx_dev})", fps_onnx))
                print(f"[bench] ONNX Runtime ({onnx_dev}) : {fps_onnx:6.1f} FPS  ({onnx_path})")
                break
            except Exception as e:  # noqa: BLE001
                print(f"[bench] ONNX on {onnx_dev} failed ({str(e)[:80]}...), trying next provider")
    except Exception as e:  # noqa: BLE001
        print(f"[bench] ONNX export skipped: {e}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["backend", "fps", "speedup_vs_fp32"])
        for name, fps in results:
            w.writerow([name, f"{fps:.1f}", f"{fps / fps_fp32:.2f}x"])
    print(f"[done] benchmark.csv written; realtime headroom vs 30fps video: "
          f"{max(f for _, f in results)/30:.1f}x")


if __name__ == "__main__":
    main()
