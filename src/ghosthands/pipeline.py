from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .backends import HandSegmentationBackend
from .render import RenderStyle, render_glow, sample_style

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class SampleResult:
    name: str
    status: str
    frame_count: int
    hand_frame_ratio: float
    flags: list[str]


def find_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)


def process_video(source_path: Path, output_root: Path, backend: HandSegmentationBackend, seed: int, crf: int = 15) -> SampleResult:
    sample_name = source_path.stem
    sample_dir = output_root / "samples" / sample_name
    frame_dir = sample_dir / "masks"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    frame_dir.mkdir(parents=True)

    capture = cv2.VideoCapture(str(source_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not width or not height:
        raise RuntimeError(f"Cannot read dimensions from {source_path}")
    output_width = width + (width % 2)
    output_height = height + (height % 2)

    local_seed = int(hashlib.sha256(f"{seed}:{source_path.name}".encode()).hexdigest()[:8], 16)
    style: RenderStyle = sample_style(local_seed)
    temp_target = sample_dir / "target_intermediate.mp4"
    writer = cv2.VideoWriter(str(temp_target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the temporary video writer")

    frame_index = 0
    detected_frames = 0
    mask_areas: list[float] = []
    hands_per_frame: list[int] = []
    tracked_instances = backend.prepare_video(source_path)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            instances = tracked_instances.get(frame_index, []) if tracked_instances is not None else backend.segment(frame)
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            for instance in instances:
                combined_mask = cv2.bitwise_or(combined_mask, instance.mask)
            if instances:
                detected_frames += 1
            mask_areas.append(float(np.count_nonzero(combined_mask)) / float(width * height))
            hands_per_frame.append(len(instances))
            output_mask = _pad_to_even(combined_mask)
            cv2.imwrite(str(frame_dir / f"{frame_index:06d}.png"), output_mask)
            writer.write(_pad_to_even(render_glow(frame, combined_mask, style)))
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames decoded from {source_path}")
    target_path = sample_dir / "target.mp4"
    source_output = sample_dir / "source.mp4"
    _normalize_video(source_path, source_output, fps, crf)
    _normalize_video(temp_target, target_path, fps, crf)
    temp_target.unlink(missing_ok=True)

    area_deltas = np.abs(np.diff(mask_areas)) if len(mask_areas) > 1 else np.array([])
    flags: list[str] = []
    hand_ratio = detected_frames / frame_index
    if hand_ratio < 0.35:
        flags.append("few_hand_frames")
    if area_deltas.size and float(np.quantile(area_deltas, 0.95)) > 0.12:
        flags.append("mask_area_jump")
    if float(np.mean(mask_areas)) > 0.35:
        flags.append("mask_area_too_large")
    if max(hands_per_frame, default=0) > 2:
        flags.append("many_hands")
    status = "needs_review" if flags else "ready"
    quality = {
        "status": status,
        "flags": flags,
        "frame_count": frame_index,
        "hand_frame_ratio": hand_ratio,
        "mask_area_fraction": {"mean": float(np.mean(mask_areas)), "p95_frame_delta": float(np.quantile(area_deltas, 0.95)) if area_deltas.size else 0.0},
    }
    metadata = {
        "source": str(source_path.resolve()),
        "backend": backend.name,
        "source_frame_size": [width, height],
        "frame_size": [output_width, output_height],
        "fps": fps,
        "frame_count": frame_index,
        "seed": local_seed,
        "style": style.to_dict(),
        "mask_format": "8-bit PNG; union of all visible hands",
        "backend_metadata": backend.metadata(),
    }
    (sample_dir / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return SampleResult(sample_name, status, frame_index, hand_ratio, flags)


def _normalize_video(input_path: Path, output_path: Path, fps: float, crf: int) -> None:
    command = ["ffmpeg", "-y", "-i", str(input_path), "-map", "0:v:0", "-an", "-r", f"{fps:.6f}", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=black", "-c:v", "libx264", "-profile:v", "main", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p", str(output_path)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed for {input_path}: {completed.stderr[-1200:]}")


def _pad_to_even(frame: np.ndarray) -> np.ndarray:
    """Keep H.264 output broadly playable while preserving input pixels exactly."""
    height, width = frame.shape[:2]
    return cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT, value=0)
