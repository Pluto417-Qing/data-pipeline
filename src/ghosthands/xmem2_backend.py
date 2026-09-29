from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from .backends import HandInstance, HandSegmentationBackend, MediaPipeBackend


class XMem2Backend(HandSegmentationBackend):
    """XMem++ video tracker seeded by sparse MediaPipe hand masks.

    XMem++ is deliberately invoked through its maintained command-line entry
    point.  That keeps its older inference stack outside this package while
    preserving this pipeline's normal backend interface.
    """

    name = "xmem2"

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        settings_path = project_root / "config" / "xmem2.yaml"
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        self._project_root = project_root
        self._root = project_root / str(settings.get("xmem2_root", "models/xmem2"))
        self._python = project_root / str(settings.get("python_executable", ".venv-sam2/bin/python"))
        self._checkpoint = project_root / str(settings.get("checkpoint", "models/xmem2/saves/XMem.pth"))
        self._seed_stride = max(1, int(settings.get("seed_stride", 30)))
        self._max_hands = max(1, int(settings.get("max_hands", 2)))
        self._prompter = MediaPipeBackend(max_hands=self._max_hands)
        self._temporary_dirs: list[tempfile.TemporaryDirectory[str]] = []
        self._last_metadata: dict[str, object] = {}

    def segment(self, bgr_frame: np.ndarray) -> list[HandInstance]:
        raise RuntimeError("XMem++ requires video context. Use process with --branch xmem2 on a video file.")

    def prepare_video(self, source_path: Path) -> dict[int, list[HandInstance]]:
        self._validate_installation()
        workspace = tempfile.TemporaryDirectory(prefix="ghosthands-xmem2-")
        self._temporary_dirs.append(workspace)
        workspace_path = Path(workspace.name)
        seeds_path = workspace_path / "seeds"
        output_path = workspace_path / "predictions"
        seed_frames, object_ids = self._write_sparse_seeds(source_path, seeds_path)
        if not seed_frames:
            raise RuntimeError(
                f"XMem++ found no MediaPipe hand masks in {source_path.name}; cannot create a reference mask. "
                "Use a clip with a visible hand or select another segmentation branch."
            )

        command = [
            str(self._python),
            "process_video.py",
            "--video", str(source_path.resolve()),
            "--masks", str(seeds_path),
            "--output", str(output_path),
        ]
        completed = subprocess.run(command, cwd=self._root, capture_output=True, text=True)
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout)[-1600:]
            raise RuntimeError(f"XMem++ inference failed for {source_path.name}: {details}")

        tracked = self._read_predictions(output_path, object_ids)
        if not tracked:
            raise RuntimeError("XMem++ completed but produced no readable PNG masks.")
        self._last_metadata = {
            "xmem2_root": str(self._root.relative_to(self._project_root)),
            "checkpoint": str(self._checkpoint.relative_to(self._project_root)),
            "seed_stride": self._seed_stride,
            "seed_frames": seed_frames,
            "object_count": len(object_ids),
        }
        return tracked

    def _validate_installation(self) -> None:
        script = self._root / "process_video.py"
        if not script.is_file():
            raise RuntimeError(f"XMem++ source is missing: {script}. Run scripts/setup-xmem2.sh first.")
        if not self._python.is_file():
            raise RuntimeError(f"XMem++ Python interpreter is missing: {self._python}. Run scripts/setup-xmem2.sh first.")
        if not self._checkpoint.is_file():
            raise RuntimeError(f"XMem++ checkpoint is missing: {self._checkpoint}. Run scripts/setup-xmem2.sh first.")

    def _write_sparse_seeds(self, source_path: Path, seeds_path: Path) -> tuple[list[int], set[int]]:
        from PIL import Image

        seeds_path.mkdir(parents=True)
        capture = cv2.VideoCapture(str(source_path))
        object_centers: dict[int, np.ndarray] = {}
        object_handedness: dict[int, str] = {}
        seed_frames: list[int] = []
        object_ids: set[int] = set()
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                found = self._prompter.segment(frame)
                assigned = self._assign_objects(found, object_centers, object_handedness)
                has_new_object = any(object_id not in object_ids for object_id, _ in assigned)
                should_seed = bool(assigned) and (frame_index % self._seed_stride == 0 or has_new_object)
                if should_seed:
                    labels = np.zeros(frame.shape[:2], dtype=np.uint8)
                    for object_id, instance in assigned:
                        labels[instance.mask > 0] = object_id
                        object_ids.add(object_id)
                    image = Image.fromarray(labels, mode="P")
                    image.putpalette(_davis_palette())
                    image.save(seeds_path / f"frame_{frame_index:06d}.png")
                    seed_frames.append(frame_index)
                frame_index += 1
        finally:
            capture.release()
        return seed_frames, object_ids

    def _assign_objects(
        self,
        found: list[HandInstance],
        centers: dict[int, np.ndarray],
        handedness: dict[int, str],
    ) -> list[tuple[int, HandInstance]]:
        assigned: list[tuple[int, HandInstance]] = []
        used: set[int] = set()
        for instance in found:
            center = instance.landmarks.mean(axis=0).astype(np.float32)
            label = instance.handedness.lower()
            candidates = [object_id for object_id in centers if object_id not in used]
            same_hand = [object_id for object_id in candidates if handedness.get(object_id) == label]
            pool = same_hand or candidates
            object_id = min(pool, key=lambda item: float(np.linalg.norm(center - centers[item]))) if pool else None
            # A hand with a new handedness normally needs a new object ID.  A
            # close fallback retains continuity when MediaPipe flips a label.
            if object_id is not None and not same_hand and float(np.linalg.norm(center - centers[object_id])) > 96.0:
                object_id = None
            if object_id is None or (len(centers) < self._max_hands and object_id in used):
                unused_ids = [item for item in range(1, self._max_hands + 1) if item not in centers]
                if not unused_ids:
                    continue
                object_id = unused_ids[0]
            centers[object_id] = center
            handedness[object_id] = label
            used.add(object_id)
            assigned.append((object_id, instance))
        return assigned

    @staticmethod
    def _read_predictions(output_path: Path, object_ids: set[int]) -> dict[int, list[HandInstance]]:
        by_frame: dict[int, list[Path]] = {}
        for path in output_path.rglob("*.png"):
            match = re.search(r"\d+", path.stem)
            if match:
                by_frame.setdefault(int(match.group()), []).append(path)
        tracked: dict[int, list[HandInstance]] = {}
        for frame_index, paths in by_frame.items():
            # XMem++ may also emit overlays. Prefer a path explicitly named as a mask.
            path = next((item for item in paths if "mask" in str(item.parent).lower()), paths[0])
            from PIL import Image

            labels = np.asarray(Image.open(path).convert("P"), dtype=np.uint8)
            instances: list[HandInstance] = []
            for object_id in sorted(object_ids):
                mask = np.where(labels == object_id, 255, 0).astype(np.uint8)
                if np.any(mask):
                    instances.append(HandInstance(mask, np.empty((0, 2), np.int32), f"XMem2-{object_id}", 1.0))
            if not instances and len(object_ids) == 1 and np.any(labels):
                instances.append(HandInstance(np.where(labels > 0, 255, 0).astype(np.uint8), np.empty((0, 2), np.int32), "XMem2-1", 1.0))
            tracked[frame_index] = instances
        return tracked

    def metadata(self) -> dict[str, object]:
        return self._last_metadata

    def close(self) -> None:
        self._prompter.close()
        for workspace in self._temporary_dirs:
            workspace.cleanup()
        self._temporary_dirs.clear()


def _davis_palette() -> list[int]:
    """Return the standard DAVIS-style indexed palette expected by XMem++."""
    palette = [0] * (256 * 3)
    for index in range(256):
        value = index
        bit = 0
        while value:
            palette[index * 3] |= ((value >> 0) & 1) << (7 - bit)
            palette[index * 3 + 1] |= ((value >> 1) & 1) << (7 - bit)
            palette[index * 3 + 2] |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
            bit += 1
    return palette
