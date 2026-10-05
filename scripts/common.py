"""Shared paths and helpers used across the pipeline scripts."""
from __future__ import annotations

import os
import platform
import re
from pathlib import Path

# --- Platform-aware defaults ---------------------------------------------
# The Windows laptop needed several workarounds (Anaconda + numpy 1.x); a clean Linux venv
# (e.g. a Linux GPU box) does not, and should use the faster settings. These helpers pick the
# right default per OS so the same scripts run well on both without editing.
IS_WINDOWS = platform.system() == "Windows"


def default_workers() -> int:
    """0 on Windows (spawn DataLoader workers hang/segfault with this stack); parallel elsewhere.

    On a SLURM node os.cpu_count() reports the whole machine, not the job's allocation, so honor
    SLURM_CPUS_PER_TASK when set to avoid oversubscribing the cores the scheduler gave us."""
    if IS_WINDOWS:
        return 0
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    n = int(slurm) if slurm else (os.cpu_count() or 4)
    return max(1, min(8, n))


def default_plots() -> bool:
    """Ultralytics' matplotlib plotting segfaults under Anaconda+numpy-1.x on Windows; it is
    fine on a clean Linux venv, where the nicer training plots are worth keeping."""
    return not IS_WINDOWS

# --- Project layout -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "Shuttlecock.v1i.yolov11.zip"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # Roboflow's original train/valid/test (leaky random split)
SPLITS_DIR = DATA_DIR / "splits"          # generated video-level split yamls + txt file lists
TRACKNET_DIR = DATA_DIR / "tracknet"      # generated triple manifest for TrackNet

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

RAW_SPLITS = ("train", "valid", "test")   # Roboflow's folder names
VIDEOS = ("1", "2", "3")                   # the three broadcast sources
CLASS_NAMES = ["Shuttlecock"]

# Filename pattern: video_label_{video}_{frame}_jpg.rf.{hash}.jpg
_FRAME_RE = re.compile(r"video_label_(\d+)_(\d+)_jpg")


def parse_video_frame(name: str):
    """Return (video_id:str, frame_idx:int) parsed from an image/label filename, or None."""
    m = _FRAME_RE.search(name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def read_label(label_path: Path):
    """Read a YOLO label file. Returns (has_ball, cx, cy, w, h) with normalized coords.

    Frames with no shuttlecock (empty label) return (False, None, None, None, None).
    Only the first box is used (this dataset has at most one).
    """
    if not label_path.exists():
        return False, None, None, None, None
    txt = label_path.read_text().strip()
    if not txt:
        return False, None, None, None, None
    parts = txt.splitlines()[0].split()
    _cls, cx, cy, w, h = parts[:5]
    return True, float(cx), float(cy), float(w), float(h)


def video_frame_paths(video_id: str):
    """Sorted list of image Paths for one dataset video (frames scattered across raw splits).

    Lets us replay a held-out video as a 'clip' for the demo without re-encoding an mp4.
    """
    frames = []
    for rec in iter_raw_frames():
        if rec["video"] == video_id:
            frames.append((rec["frame"], rec["image"]))
    frames.sort(key=lambda t: t[0])
    return [p for _, p in frames]


def iter_raw_frames():
    """Yield dicts for every frame in data/raw across all Roboflow splits.

    Each dict: {video, frame, raw_split, image, label, has_ball, cx, cy, w, h}
    Coordinates are normalized (0-1); cx/cy are None for ball-absent frames.
    """
    for raw_split in RAW_SPLITS:
        img_dir = RAW_DIR / raw_split / "images"
        lbl_dir = RAW_DIR / raw_split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            vf = parse_video_frame(img.name)
            if vf is None:
                continue
            video, frame = vf
            label = lbl_dir / (img.stem + ".txt")
            has_ball, cx, cy, w, h = read_label(label)
            yield {
                "video": video, "frame": frame, "raw_split": raw_split,
                "image": img, "label": label,
                "has_ball": has_ball, "cx": cx, "cy": cy, "w": w, "h": h,
            }
