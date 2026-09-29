from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class HandInstance:
    mask: np.ndarray
    landmarks: np.ndarray
    handedness: str
    confidence: float


class HandSegmentationBackend(ABC):
    name: str

    @abstractmethod
    def segment(self, bgr_frame: np.ndarray) -> list[HandInstance]:
        """Return one full-resolution binary mask for every visible hand."""

    def close(self) -> None:
        pass

    def metadata(self) -> dict[str, object]:
        """Return backend-specific, JSON-serialisable run details."""
        return {}

    def prepare_video(self, source_path: Path) -> dict[int, list[HandInstance]] | None:
        """Optionally return tracked instances for a whole video before rendering.

        Frame-based backends return ``None`` and keep using ``segment``. Video
        trackers such as SAM 2 override this method so temporal state is never
        recreated for every frame.
        """
        return None


class SkinToneBackend(HandSegmentationBackend):
    """Dependency-free research baseline for checking the renderer and dataset flow."""

    name = "skin"

    def segment(self, bgr_frame: np.ndarray) -> list[HandInstance]:
        ycrcb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        ycrcb_mask = cv2.inRange(ycrcb, np.array((0, 133, 77), np.uint8), np.array((255, 173, 127), np.uint8))
        hsv_mask = cv2.inRange(hsv, np.array((0, 20, 45), np.uint8), np.array((30, 255, 255), np.uint8))
        mask = cv2.bitwise_and(ycrcb_mask, hsv_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        instances: list[HandInstance] = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 350:
                continue
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            instances.append(HandInstance(component, np.empty((0, 2), np.int32), "Unknown", 0.0))
        return instances



class MediaPipeBackend(HandSegmentationBackend):
    """Fast baseline: converts 21 hand landmarks to a conservative hand mask."""

    name = "mediapipe"

    def __init__(self, max_hands: int = 4, detection_confidence: float = 0.55) -> None:
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This MediaPipe version needs a separate Hand Landmarker model file, which is unavailable on this machine. "
                "Use --backend skin for the local rendering baseline, or add a local MediaPipe task model adapter."
            )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.55,
        )

    def segment(self, bgr_frame: np.ndarray) -> list[HandInstance]:
        height, width = bgr_frame.shape[:2]
        result = self._hands.process(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return []

        instances: list[HandInstance] = []
        handedness_items: Iterable = result.multi_handedness or []
        for index, landmark_set in enumerate(result.multi_hand_landmarks):
            points = np.array(
                [[int(point.x * width), int(point.y * height)] for point in landmark_set.landmark],
                dtype=np.int32,
            )
            mask = self._landmarks_to_mask(points, width, height)
            classification = list(handedness_items)[index].classification[0] if index < len(result.multi_handedness or []) else None
            instances.append(
                HandInstance(
                    mask=mask,
                    landmarks=points,
                    handedness=classification.label if classification else "Unknown",
                    confidence=float(classification.score) if classification else 0.0,
                )
            )
        return instances

    @staticmethod
    def _landmarks_to_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        # Palm hull keeps the palm solid while finger chains preserve finger separation.
        palm_indices = np.array([0, 1, 5, 9, 13, 17], dtype=np.int32)
        palm = cv2.convexHull(points[palm_indices])
        cv2.fillConvexPoly(mask, palm, 255)

        finger_chains = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
        palm_width = max(12, int(np.linalg.norm(points[5] - points[17]) * 0.36))
        for chain in finger_chains:
            chain_points = points[list(chain)]
            cv2.polylines(mask, [chain_points], False, 255, palm_width, cv2.LINE_AA)
            for point in chain_points:
                cv2.circle(mask, tuple(point), max(4, palm_width // 2), 255, -1, cv2.LINE_AA)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    def close(self) -> None:
        self._hands.close()


class Sam2Backend(HandSegmentationBackend):
    name = "sam2"

    def __init__(self, max_hands: int = 2) -> None:
        import yaml
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as error:
            raise RuntimeError(
                "SAM 2 is not installed. Install torch and the local models/sam2 package before selecting --branch sam2."
            ) from error

        project_root = Path(__file__).resolve().parents[2]
        settings_path = project_root / "config" / "sam2.yaml"
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        sam2_root = project_root / str(settings.get("sam2_root", "models/sam2"))
        model_config_name = str(settings.get("model_config", "configs/sam2.1/sam2.1_hiera_s.yaml"))
        config_value = Path(model_config_name)
        # SAM 2 registers ``sam2/configs`` as its Hydra package.  The builder
        # therefore needs the logical ``configs/...`` name, while this check
        # must use the on-disk package location.
        model_config = config_value if config_value.is_absolute() else sam2_root / "sam2" / config_value
        checkpoint = project_root / str(settings.get("checkpoint", "models/sam2/checkpoints/sam2.1_hiera_small.pt"))
        if not model_config.is_file():
            raise RuntimeError(f"SAM 2 config is missing: {model_config}")
        if not checkpoint.is_file():
            raise RuntimeError(f"SAM 2 checkpoint is missing: {checkpoint}")
        self._torch = torch
        # Landmark masks are an inexpensive way to prompt SAM 2, but they are
        # not accurate enough to be the final output.  Periodic prompts give
        # SAM 2 chances to recover when a hand is initially absent or becomes
        # heavily occluded.
        self._prompt_mode = str(settings.get("prompt_mode", "keyframes")).lower()
        self._prompt_stride = max(1, int(settings.get("prompt_stride", 12)))
        self._prompt_type = str(settings.get("prompt_type", "box_points")).lower()
        if self._prompt_mode not in {"initial", "keyframes"}:
            raise RuntimeError("sam2.prompt_mode must be 'initial' or 'keyframes'.")
        if self._prompt_type not in {"mask", "box_points"}:
            raise RuntimeError("sam2.prompt_type must be 'mask' or 'box_points'.")
        self._max_hands = max_hands
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if config_value.is_absolute():
            raise RuntimeError("SAM 2 model_config must be a logical configs/... name, not an absolute path.")
        self._predictor = build_sam2_video_predictor(model_config_name, str(checkpoint), device=self._device)
        self._prompter = MediaPipeBackend(max_hands=max_hands)

    def segment(self, bgr_frame: np.ndarray) -> list[HandInstance]:
        raise RuntimeError("SAM 2 requires video context. Use process with --branch sam2 on a video file.")

    def prepare_video(self, source_path: Path) -> dict[int, list[HandInstance]]:
        """Use MediaPipe masks as initial or periodic correction prompts for SAM 2."""
        prompt_frames = self._collect_prompts(source_path)
        if not prompt_frames:
            return {}

        with self._torch.inference_mode():
            state = self._predictor.init_state(video_path=str(source_path))
            for frame_index, object_id, prompt in prompt_frames:
                if self._prompt_type == "mask":
                    self._predictor.add_new_mask(state, frame_index, object_id, prompt.mask > 0)
                else:
                    points = prompt.landmarks.astype(np.float32)
                    # A small box gives SAM 2 a clean spatial prior; all 21
                    # detected landmarks are positive evidence for the hand.
                    x0, y0 = points.min(axis=0)
                    x1, y1 = points.max(axis=0)
                    margin = max(10.0, float(np.linalg.norm(points[5] - points[17]) * 0.35))
                    video_height, video_width = prompt.mask.shape
                    box = np.array([
                        max(0.0, x0 - margin), max(0.0, y0 - margin),
                        min(float(video_width - 1), x1 + margin), min(float(video_height - 1), y1 + margin),
                    ], dtype=np.float32)
                    self._predictor.add_new_points_or_box(
                        state,
                        frame_index,
                        object_id,
                        points=points,
                        labels=np.ones(len(points), dtype=np.int32),
                        box=box,
                        normalize_coords=False,
                    )
            tracked: dict[int, list[HandInstance]] = {}
            for frame_index, object_ids, mask_logits in self._predictor.propagate_in_video(state):
                frame_instances: list[HandInstance] = []
                for object_id, logits in zip(object_ids, mask_logits):
                    mask = (logits > 0.0).squeeze().detach().cpu().numpy().astype(np.uint8) * 255
                    if mask.ndim != 2:
                        continue
                    frame_instances.append(HandInstance(mask, np.empty((0, 2), np.int32), f"SAM2-{object_id}", 1.0))
                tracked[int(frame_index)] = frame_instances
            self._predictor.reset_state(state)
        return tracked

    def _collect_prompts(self, source_path: Path) -> list[tuple[int, int, HandInstance]]:
        """Associate hand detections across sparse key frames for SAM 2 prompts."""
        capture = cv2.VideoCapture(str(source_path))
        prompts: list[tuple[int, int, HandInstance]] = []
        object_for_handedness: dict[str, int] = {}
        latest_centers: dict[int, np.ndarray] = {}
        next_object_id = 1
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                found = self._prompter.segment(frame)
                should_correct = self._prompt_mode == "keyframes" and frame_index % self._prompt_stride == 0
                for prompt in found:
                    center = prompt.landmarks.mean(axis=0).astype(np.float32)
                    handedness = prompt.handedness.lower()
                    object_id = object_for_handedness.get(handedness)
                    if object_id is None and latest_centers:
                        object_id = min(latest_centers, key=lambda item: float(np.linalg.norm(center - latest_centers[item])))
                    if object_id is None:
                        if next_object_id > self._max_hands:
                            continue
                        object_id = next_object_id
                        next_object_id += 1
                    object_for_handedness[handedness] = object_id
                    latest_centers[object_id] = center
                    # Always create a prompt for a newly observed hand.  Later
                    # prompts are optional corrections and preserve the legacy
                    # first-prompt route when prompt_mode=initial.
                    is_new_object = not any(item[1] == object_id for item in prompts)
                    if is_new_object or should_correct:
                        prompts.append((frame_index, object_id, prompt))
                frame_index += 1
        finally:
            capture.release()
        return prompts

    def close(self) -> None:
        self._prompter.close()


def create_backend(name: str) -> HandSegmentationBackend:
    # Keep the optional XMem++ dependency isolated: importing the normal CLI
    # must not require its local checkout or its inference-only packages.
    from .xmem2_backend import XMem2Backend

    backends = {"skin": SkinToneBackend, "mediapipe": MediaPipeBackend, "sam2": Sam2Backend, "xmem2": XMem2Backend}
    try:
        return backends[name.lower()]()
    except KeyError as error:
        raise ValueError(f"Unknown backend '{name}'. Available: {', '.join(backends)}") from error
