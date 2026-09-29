from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .backends import HandInstance, HandSegmentationBackend, MediaPipeBackend


class ManoKinematicBackend(HandSegmentationBackend):
    """Low-cost pre-MANO route based on tracked 21-point hand kinematics.

    A licensed MANO model file is intentionally not bundled with this project.
    This backend therefore keeps the part of the MANO route that matters for a
    quick review: two independent hand states, a complete landmark topology,
    and temporal interpolation through short detector dropouts.  It emits a
    full hand silhouette from that topology and records whether a state was
    observed or predicted.  Replacing ``_mask_from_landmarks`` with a MANO
    mesh renderer is a contained follow-up once MANO assets are available.
    """

    name = "mano_kinematic"

    def __init__(self, max_gap: int = 18, recovery_detection_confidence: float = 0.35) -> None:
        self._detector = MediaPipeBackend(max_hands=2)
        self._recovery_detector = MediaPipeBackend(
            max_hands=2, detection_confidence=recovery_detection_confidence
        )
        self._max_gap = max_gap
        self._recovery_detection_confidence = recovery_detection_confidence
        self._recovery_attempt_frames = 0
        self._recovery_hand_count = 0
        self._run_metadata: dict[str, object] = {
            "representation": "21-point kinematic hand proxy; not a licensed MANO mesh",
            "max_predicted_gap_frames": max_gap,
            "single_hand_recovery": {
                "enabled": True,
                "detection_confidence": recovery_detection_confidence,
            },
        }

    def segment(self, bgr_frame: np.ndarray) -> list[HandInstance]:
        raise RuntimeError("mano_kinematic requires video context. Process a video file instead of a single frame.")

    def prepare_video(self, source_path: Path) -> dict[int, list[HandInstance]]:
        observed = self._read_and_associate(source_path)
        total_frames = len(observed["Left"])
        completed = {side: self._complete_track(track) for side, track in observed.items()}
        tracked: dict[int, list[HandInstance]] = {}
        counts = {"observed": 0, "interpolated": 0, "extrapolated": 0}
        for frame_index in range(total_frames):
            instances: list[HandInstance] = []
            for side in ("Left", "Right"):
                points, state = completed[side][frame_index]
                if points is None:
                    continue
                counts[state] += 1
                height, width = self._frame_size
                mask = self._mask_from_landmarks(points, width, height)
                confidence = 0.95 if state == "observed" else (0.55 if state == "interpolated" else 0.35)
                instances.append(HandInstance(mask, points.astype(np.int32), f"{side}:{state}", confidence))
            tracked[frame_index] = instances
        self._run_metadata.update({
            "frame_count": total_frames,
            "state_counts": counts,
            "hand_tracks": {side: int(sum(point is not None for point in track)) for side, track in observed.items()},
            "single_hand_recovery": {
                "enabled": True,
                "detection_confidence": self._recovery_detection_confidence,
                "attempted_frames": self._recovery_attempt_frames,
                "recovered_hands": self._recovery_hand_count,
            },
        })
        return tracked

    def _read_and_associate(self, source_path: Path) -> dict[str, list[np.ndarray | None]]:
        self._recovery_attempt_frames = 0
        self._recovery_hand_count = 0
        capture = cv2.VideoCapture(str(source_path))
        tracks: dict[str, list[np.ndarray | None]] = {"Left": [], "Right": []}
        latest: dict[str, np.ndarray | None] = {"Left": None, "Right": None}
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self._frame_size = frame.shape[:2]
                detections = self._detector.segment(frame)
                if len(detections) < 2:
                    self._recovery_attempt_frames += 1
                    primary_count = len(detections)
                    detections = self._merge_recovery_detections(
                        detections, self._recovery_detector.segment(frame)
                    )
                    self._recovery_hand_count += len(detections) - primary_count
                assigned: dict[str, HandInstance] = {}
                # Respect stable handedness when possible.  If both detections
                # receive the same label during overlap, assign the duplicate
                # to the unused nearest track rather than merging two hands.
                for detection in detections:
                    preferred = detection.handedness.title()
                    candidates = [preferred] if preferred in tracks and preferred not in assigned else []
                    candidates += [side for side in tracks if side not in assigned and side not in candidates]
                    if not candidates:
                        continue
                    center = detection.landmarks.mean(axis=0).astype(np.float32)
                    side = min(
                        candidates,
                        key=lambda candidate: 0.0 if candidate == preferred and latest[candidate] is None
                        else float(np.linalg.norm(center - latest[candidate].mean(axis=0))) if latest[candidate] is not None
                        else 1e9,
                    )
                    assigned[side] = detection
                for side in tracks:
                    points = assigned[side].landmarks.astype(np.float32) if side in assigned else None
                    tracks[side].append(points)
                    if points is not None:
                        latest[side] = points
        finally:
            capture.release()
        return tracks

    @staticmethod
    def _merge_recovery_detections(
        primary: list[HandInstance], recovery: list[HandInstance]
    ) -> list[HandInstance]:
        """Add only a genuinely missing hand from the lower-threshold pass."""
        merged = list(primary)
        for candidate in recovery:
            center = candidate.landmarks.mean(axis=0)
            is_duplicate = False
            for existing in merged:
                existing_center = existing.landmarks.mean(axis=0)
                palm_width = float(np.linalg.norm(existing.landmarks[5] - existing.landmarks[17]))
                if np.linalg.norm(center - existing_center) < max(24.0, 0.4 * palm_width):
                    is_duplicate = True
                    break
            if not is_duplicate and len(merged) < 2:
                merged.append(candidate)
        return merged

    def _complete_track(self, track: list[np.ndarray | None]) -> list[tuple[np.ndarray | None, str]]:
        result: list[tuple[np.ndarray | None, str]] = [(point, "observed") if point is not None else (None, "missing") for point in track]
        known = [index for index, point in enumerate(track) if point is not None]
        for left, right in zip(known, known[1:]):
            gap = right - left - 1
            if not gap or gap > self._max_gap:
                continue
            start, end = track[left], track[right]
            assert start is not None and end is not None
            for offset in range(1, gap + 1):
                fraction = offset / (gap + 1)
                result[left + offset] = (start * (1.0 - fraction) + end * fraction, "interpolated")
        # At a clip boundary, use constant-velocity extrapolation only for a
        # very short outage.  Never invent a hand before it has been observed.
        if len(known) >= 2:
            first, second = known[0], known[1]
            if first > 0 and first <= self._max_gap:
                velocity = (track[second] - track[first]) / max(1, second - first)  # type: ignore[operator]
                for index in range(first - 1, -1, -1):
                    result[index] = (track[first] - velocity * (first - index), "extrapolated")  # type: ignore[operator]
            penultimate, last = known[-2], known[-1]
            tail = len(track) - last - 1
            if tail and tail <= self._max_gap:
                velocity = (track[last] - track[penultimate]) / max(1, last - penultimate)  # type: ignore[operator]
                for index in range(last + 1, len(track)):
                    result[index] = (track[last] + velocity * (index - last), "extrapolated")  # type: ignore[operator]
        return result

    @staticmethod
    def _mask_from_landmarks(points: np.ndarray, width: int, height: int) -> np.ndarray:
        # The landmark topology is the same 21-joint topology to which MANO is
        # fitted.  This produces a complete silhouette even where the detector
        # did not observe a pixel during a short occlusion.
        return MediaPipeBackend._landmarks_to_mask(np.rint(points).astype(np.int32), width, height)

    def metadata(self) -> dict[str, object]:
        return self._run_metadata

    def close(self) -> None:
        self._detector.close()
        self._recovery_detector.close()
