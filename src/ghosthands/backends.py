from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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

    def __init__(self, *_: object, **__: object) -> None:
        raise RuntimeError(
            "The SAM 2 comparison adapter is not configured yet. Install SAM 2, provide its checkpoint "
            "and config, then register the adapter; the pipeline will not silently substitute MediaPipe."
        )

    def segment(self, bgr_frame: np.ndarray) -> list[HandInstance]:
        raise AssertionError("unreachable")


def create_backend(name: str) -> HandSegmentationBackend:
    backends = {"skin": SkinToneBackend, "mediapipe": MediaPipeBackend, "sam2": Sam2Backend}
    try:
        return backends[name.lower()]()
    except KeyError as error:
        raise ValueError(f"Unknown backend '{name}'. Available: {', '.join(backends)}") from error
