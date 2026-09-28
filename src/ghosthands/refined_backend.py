from __future__ import annotations

import cv2
import numpy as np

from .backends import HandInstance, HandSegmentationBackend, MediaPipeBackend


class RefinedMediaPipeBackend(HandSegmentationBackend):
    """Separate pixel-aware branch: landmark mask, then constrained GrabCut."""

    name = "mediapipe_grabcut"

    def __init__(self) -> None:
        self._landmark_backend = MediaPipeBackend()

    def segment(self, bgr_frame: np.ndarray) -> list[HandInstance]:
        initial_instances = self._landmark_backend.segment(bgr_frame)
        return [
            HandInstance(
                mask=self._refine(bgr_frame, instance.mask),
                landmarks=instance.landmarks,
                handedness=instance.handedness,
                confidence=instance.confidence,
            )
            for instance in initial_instances
        ]

    @staticmethod
    def _refine(frame: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
        height, width = initial_mask.shape
        if not np.any(initial_mask):
            return initial_mask

        # Work only in a padded hand crop. Running GrabCut on an entire 720p
        # frame for each hand is needlessly expensive and does not add evidence.
        ys, xs = np.where(initial_mask > 0)
        margin = 28
        x0, x1 = max(0, int(xs.min()) - margin), min(width, int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(height, int(ys.max()) + margin + 1)
        frame_crop = frame[y0:y1, x0:x1]
        mask_crop = initial_mask[y0:y1, x0:x1]

        # Known hand pixels and a bounded nearby background prevent leakage to
        # piano keys or other similarly coloured objects.
        core = cv2.erode(mask_crop, np.ones((9, 9), np.uint8))
        if not np.any(core):
            core = mask_crop
        envelope = cv2.dilate(mask_crop, np.ones((31, 31), np.uint8))
        grabcut_mask = np.full(mask_crop.shape, cv2.GC_BGD, dtype=np.uint8)
        grabcut_mask[envelope > 0] = cv2.GC_PR_BGD
        grabcut_mask[mask_crop > 0] = cv2.GC_PR_FGD
        grabcut_mask[core > 0] = cv2.GC_FGD

        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(frame_crop, grabcut_mask, None, background_model, foreground_model, 2, cv2.GC_INIT_WITH_MASK)
            foreground = np.where(
                (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
        except cv2.error:
            foreground = mask_crop.copy()

        foreground = cv2.bitwise_and(foreground, envelope)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        refined = np.zeros_like(initial_mask)
        refined[y0:y1, x0:x1] = foreground
        return refined

    def close(self) -> None:
        self._landmark_backend.close()
