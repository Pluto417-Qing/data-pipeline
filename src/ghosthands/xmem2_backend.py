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
        # More frequent corrections keep the permanent memory from carrying a
        # coarse landmark outline through a pose change or partial occlusion.
        self._seed_stride = max(1, int(settings.get("seed_stride", 10)))
        self._max_hands = max(1, int(settings.get("max_hands", 2)))
        self._edge_refinement = bool(settings.get("edge_refinement", True))
        self._edge_margin = max(4, int(settings.get("edge_margin", 20)))
        self._edge_iterations = max(1, int(settings.get("edge_iterations", 2)))
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

        capture = cv2.VideoCapture(str(source_path))
        try:
            source_shape = (
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            )
        finally:
            capture.release()
        if not all(source_shape):
            raise RuntimeError(f"Cannot read dimensions from {source_path}")

        tracked = self._read_predictions(output_path, object_ids, source_shape)
        if not tracked:
            raise RuntimeError("XMem++ completed but produced no readable PNG masks.")
        if self._edge_refinement:
            self._refine_prediction_edges(source_path, tracked)
        self._last_metadata = {
            "xmem2_root": str(self._root.relative_to(self._project_root)),
            "checkpoint": str(self._checkpoint.relative_to(self._project_root)),
            "seed_stride": self._seed_stride,
            "seed_frames": seed_frames,
            "object_count": len(object_ids),
            "edge_refinement": self._edge_refinement,
            "edge_margin": self._edge_margin if self._edge_refinement else None,
        }
        return tracked

    def _refine_prediction_edges(self, source_path: Path, tracked: dict[int, list[HandInstance]]) -> None:
        """Snap XMem contours to pixels without allowing a full-frame leak.

        XMem's output is a strong temporal prior, but its inference-scale mask
        becomes stair-stepped when restored to the source resolution. GrabCut
        is therefore run only in a narrow dilated band around each prediction:
        eroded XMem pixels remain definite foreground and all pixels outside
        the band remain definite background.
        """
        capture = cv2.VideoCapture(str(source_path))
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                instances = tracked.get(frame_index)
                if instances:
                    tracked[frame_index] = [
                        HandInstance(
                            self._refine_mask_edge(frame, instance.mask),
                            instance.landmarks,
                            instance.handedness,
                            instance.confidence,
                        )
                        for instance in instances
                    ]
                frame_index += 1
        finally:
            capture.release()

    def _refine_mask_edge(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return mask

        height, width = mask.shape
        ys, xs = np.where(mask > 0)
        margin = self._edge_margin
        x0, x1 = max(0, int(xs.min()) - margin), min(width, int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(height, int(ys.max()) + margin + 1)
        mask_crop = mask[y0:y1, x0:x1]
        frame_crop = frame[y0:y1, x0:x1]

        # Keep a conservative XMem interior locked as hand. The uncertainty
        # band is deliberately narrow, so similar-coloured background cannot
        # be absorbed far away from the tracked contour.
        core_kernel = np.ones((5, 5), np.uint8)
        band_kernel = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
        core = cv2.erode(mask_crop, core_kernel)
        if not np.any(core):
            core = mask_crop
        envelope = cv2.dilate(mask_crop, band_kernel)
        grabcut_mask = np.full(mask_crop.shape, cv2.GC_BGD, dtype=np.uint8)
        grabcut_mask[envelope > 0] = cv2.GC_PR_BGD
        grabcut_mask[mask_crop > 0] = cv2.GC_PR_FGD
        grabcut_mask[core > 0] = cv2.GC_FGD
        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                frame_crop,
                grabcut_mask,
                None,
                background_model,
                foreground_model,
                self._edge_iterations,
                cv2.GC_INIT_WITH_MASK,
            )
            refined_crop = np.where(
                (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
        except cv2.error:
            refined_crop = mask_crop.copy()

        refined_crop = cv2.bitwise_and(refined_crop, envelope)
        # Remove islands that GrabCut may create, while retaining every region
        # connected to the definite XMem core (important for separated fingers).
        count, labels, _, _ = cv2.connectedComponentsWithStats(refined_crop)
        keep = np.unique(labels[core > 0])
        cleaned = np.zeros_like(refined_crop)
        for label in keep:
            if label:
                cleaned[labels == label] = 255
        if count <= 1 or not np.any(cleaned):
            cleaned = mask_crop.copy()
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        refined = np.zeros_like(mask)
        refined[y0:y1, x0:x1] = cleaned
        return refined

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
    def _read_predictions(
        output_path: Path,
        object_ids: set[int],
        source_shape: tuple[int, int],
    ) -> dict[int, list[HandInstance]]:
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

            # XMem++ maps prediction indices back to the palette of the input
            # seed masks and writes an RGB PNG. Converting that image to a new
            # palette changes IDs 1/2 into arbitrary palette indices, making
            # every object appear absent. Decode DAVIS palette colors directly.
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
            palette = np.asarray(_davis_palette(), dtype=np.uint8).reshape(256, 3)
            for object_id in object_ids:
                labels[np.all(rgb == palette[object_id], axis=2)] = object_id

            # XMem++ uses a 480-pixel minimum side by default. Restore the
            # original frame size before combining the mask with OpenCV frames.
            if labels.shape != source_shape:
                labels = cv2.resize(labels, (source_shape[1], source_shape[0]), interpolation=cv2.INTER_NEAREST)
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
