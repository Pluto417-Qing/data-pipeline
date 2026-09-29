from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .backends import HandInstance
from .mano_kinematic_backend import ManoKinematicBackend


class ManoMeshBackend(ManoKinematicBackend):
    """Fit the licensed MANO mesh to tracked MediaPipe landmarks in a video.

    This intentionally uses a weak-perspective camera: a monocular RGB frame
    cannot determine absolute depth.  The output is nevertheless a genuine
    778-vertex MANO mesh projected with the recovered hand pose, rather than
    a landmark-drawn silhouette.  Missing landmark frames inherit a short
    temporally completed track before fitting.
    """

    name = "mano"

    def __init__(
        self,
        max_gap: int = 18,
        fit_steps: int = 25,
        edge_margin: int = 12,
        edge_iterations: int = 1,
    ) -> None:
        super().__init__(max_gap=max_gap)
        import smplx

        root = Path(__file__).resolve().parents[2]
        model_path = root / "models" / "mano" / "MANO_RIGHT.pkl"
        if not model_path.is_file():
            raise RuntimeError(f"MANO model is missing: {model_path}")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = smplx.create(
            str(model_path), model_type="mano", is_rhand=True,
            use_pca=False, flat_hand_mean=False, batch_size=1,
        ).to(self._device).eval()
        self._faces = self._model.faces.astype(np.int32)
        self._fit_steps = fit_steps
        # MANO provides the stable geometric prior; GrabCut is allowed to
        # adjust only a narrow band around it.  This recovers image-space
        # skin contours without letting nearby objects become foreground.
        self._edge_margin = max(4, edge_margin)
        self._edge_iterations = max(1, edge_iterations)
        self._run_metadata.update({
            "representation": "licensed MANO_RIGHT mesh fitted to 21-point tracks",
            "mesh_vertices": 778,
            "mesh_faces": int(len(self._faces)),
            "camera": "weak-perspective",
            "fit_steps": fit_steps,
            "device": str(self._device),
            "edge_refinement": "constrained_grabcut",
            "edge_margin": self._edge_margin,
            "finger_valley_separation": "landmark-guided distal valleys",
        })

    def prepare_video(self, source_path: Path) -> dict[int, list[HandInstance]]:
        frames = self._read_frames(source_path)
        observed = self._read_and_associate(source_path)
        total_frames = len(observed["Left"])
        if len(frames) != total_frames:
            raise RuntimeError(
                f"Frame-count mismatch while processing {source_path.name}: decoded {len(frames)}, tracked {total_frames}"
            )
        completed = {side: self._complete_track(track) for side, track in observed.items()}
        meshes = {side: self._fit_track(completed[side], side) for side in ("Left", "Right")}
        tracked: dict[int, list[HandInstance]] = {}
        counts = {"observed": 0, "interpolated": 0, "extrapolated": 0}
        for frame_index in range(total_frames):
            instances: list[HandInstance] = []
            for side in ("Left", "Right"):
                points, state = completed[side][frame_index]
                vertices = meshes[side][frame_index]
                if points is None or vertices is None:
                    continue
                counts[state] += 1
                mask = self._refine_mesh_edge(frames[frame_index], self._rasterize_mesh(vertices))
                mask = self._separate_finger_valleys(mask, points)
                confidence = 0.95 if state == "observed" else (0.55 if state == "interpolated" else 0.35)
                instances.append(HandInstance(mask, np.rint(points).astype(np.int32), f"{side}:{state}", confidence))
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

    def _read_frames(self, source_path: Path) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(source_path))
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        if not frames:
            raise RuntimeError(f"Could not decode frames from {source_path}")
        return frames

    def _refine_mesh_edge(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Align a MANO silhouette to pixels while retaining its topology.

        The eroded mesh remains definite foreground.  Only the thin dilated
        perimeter can change, so the refinement cannot chase the colourful
        paper object or the distant blue background.
        """
        if not np.any(mask):
            return mask
        height, width = mask.shape
        ys, xs = np.where(mask > 0)
        margin = self._edge_margin
        x0, x1 = max(0, int(xs.min()) - margin), min(width, int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(height, int(ys.max()) + margin + 1)
        frame_crop = frame[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]

        core = cv2.erode(mask_crop, np.ones((5, 5), np.uint8))
        # Very thin fingers can disappear under erosion; their mesh pixels
        # stay probable foreground rather than being discarded.
        envelope = cv2.dilate(mask_crop, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
        grabcut_mask = np.full(mask_crop.shape, cv2.GC_BGD, dtype=np.uint8)
        grabcut_mask[envelope > 0] = cv2.GC_PR_BGD
        grabcut_mask[mask_crop > 0] = cv2.GC_PR_FGD
        grabcut_mask[core > 0] = cv2.GC_FGD
        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                frame_crop, grabcut_mask, None, background_model, foreground_model,
                self._edge_iterations, cv2.GC_INIT_WITH_MASK,
            )
            refined_crop = np.where(
                (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
        except cv2.error:
            refined_crop = mask_crop.copy()

        refined_crop = cv2.bitwise_and(refined_crop, envelope)
        # Keep every component linked to a guaranteed mesh interior. This
        # suppresses islands that GrabCut may find in the nearby scene.
        count, labels, _, _ = cv2.connectedComponentsWithStats(refined_crop)
        if np.any(core):
            keep = np.unique(labels[core > 0])
            cleaned = np.zeros_like(refined_crop)
            for label in keep:
                if label:
                    cleaned[labels == label] = 255
            refined_crop = cleaned if np.any(cleaned) else mask_crop.copy()
        if count <= 1:
            refined_crop = mask_crop.copy()
        refined_crop = cv2.morphologyEx(refined_crop, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        initial_area = int(np.count_nonzero(mask_crop))
        changed = int(np.count_nonzero(cv2.bitwise_xor(refined_crop, mask_crop)))
        refined_area = int(np.count_nonzero(refined_crop))
        # A pixel-only classifier is not allowed to override the mesh when it
        # substantially changes the hand.  This specifically rejects paper
        # touching a fingertip, which GrabCut may otherwise merge with skin.
        if (
            not initial_area
            or refined_area < 0.93 * initial_area
            or refined_area > 1.08 * initial_area
            or changed > 0.11 * initial_area
        ):
            refined_crop = mask_crop.copy()
        refined = np.zeros_like(mask)
        refined[y0:y1, x0:x1] = refined_crop
        return refined

    @staticmethod
    def _separate_finger_valleys(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Restore distal finger gaps using only the observed hand topology.

        MANO's projected surface can bridge tightly posed fingers.  For each
        non-thumb neighbouring pair, the midpoint of its PIP, DIP and tip
        landmarks gives a reliable valley direction.  A very thin cut starts
        at the PIP level, deliberately leaving the palm intact.
        """
        result = mask.copy()
        palm_width = float(np.linalg.norm(points[5] - points[17]))
        thickness = max(2, min(5, int(round(palm_width * 0.035))))
        # Index-middle, middle-ring, ring-pinky. Thumb contact is commonly
        # occluded by the grasped object, so it is intentionally excluded.
        finger_pairs = ((5, 9), (9, 13), (13, 17))
        for left_base, right_base in finger_pairs:
            left = points[left_base:left_base + 4]
            right = points[right_base:right_base + 4]
            if np.linalg.norm(left[3] - right[3]) < 2.5 * thickness:
                continue
            valley = np.rint((left[1:] + right[1:]) * 0.5).astype(np.int32)
            # A cut must be backed by foreground along its distal end; this
            # avoids drawing a stray mark when MediaPipe loses one finger.
            tip_x, tip_y = valley[-1]
            if not (0 <= tip_x < result.shape[1] and 0 <= tip_y < result.shape[0]) or result[tip_y, tip_x] == 0:
                continue
            cv2.polylines(result, [valley], False, 0, thickness, cv2.LINE_8)
        return result

    def _fit_track(self, track: list[tuple[np.ndarray | None, str]], side: str) -> list[np.ndarray | None]:
        fitted: list[np.ndarray | None] = []
        previous: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        for points, _ in track:
            if points is None:
                fitted.append(None)
                continue
            vertices, previous = self._fit_frame(points, side, previous)
            fitted.append(vertices)
        return fitted

    def _fit_frame(
        self,
        points: np.ndarray,
        side: str,
        previous: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Fit the right MANO model to mirrored left-hand landmarks.  Mirror
        # the final projection back afterwards, avoiding an unlicensed or
        # absent left-model substitute.
        target = points.astype(np.float32).copy()
        wrist_x = float(target[0, 0])
        if side == "Left":
            target[:, 0] = 2.0 * wrist_x - target[:, 0]
        palm_size = max(24.0, float(np.linalg.norm(target[5] - target[17])))
        target_relative = torch.as_tensor((target - target[0]) / palm_size, device=self._device)
        if previous is None:
            orient = torch.zeros((1, 3), device=self._device, requires_grad=True)
            pose = torch.zeros((1, 45), device=self._device, requires_grad=True)
            log_scale = torch.zeros((1, 1), device=self._device, requires_grad=True)
        else:
            orient, pose, log_scale = [item.detach().clone().requires_grad_(True) for item in previous]
        optimizer = torch.optim.Adam((orient, pose, log_scale), lr=0.045)
        for _ in range(self._fit_steps):
            output = self._model(global_orient=orient, hand_pose=pose, return_verts=True)
            joints = self._mano_21(output.joints[0], output.vertices[0])
            relative = (joints[:, :2] - joints[0, :2]) * torch.exp(log_scale)
            loss = torch.nn.functional.smooth_l1_loss(relative, target_relative, beta=0.04)
            # Keep single-view solutions within the natural MANO pose space.
            loss = loss + 0.0015 * pose.square().mean() + 0.0005 * orient.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            output = self._model(global_orient=orient, hand_pose=pose, return_verts=True)
            joints = self._mano_21(output.joints[0], output.vertices[0])
            scale = torch.exp(log_scale)[0, 0] * palm_size
            vertices = (output.vertices[0, :, :2] - joints[0, :2]) * scale + torch.as_tensor(target[0], device=self._device)
            result = vertices.detach().cpu().numpy()
        if side == "Left":
            result[:, 0] = 2.0 * wrist_x - result[:, 0]
        return result, (orient.detach(), pose.detach(), log_scale.detach())

    @staticmethod
    def _mano_21(joints: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
        # MANO's 16 regressed joints are wrist, index, middle, pinky, ring,
        # thumb (three joints each).  The five fingertip landmarks come from
        # standard MANO mesh vertices.  Reorder into MediaPipe's 0..20 order.
        tips = vertices[torch.as_tensor([744, 320, 443, 554, 671], device=vertices.device)]
        return torch.cat((
            joints[0:1], joints[13:16], tips[0:1],
            joints[1:4], tips[1:2],
            joints[4:7], tips[2:3],
            joints[10:13], tips[3:4],
            joints[7:10], tips[4:5],
        ), dim=0)

    def _rasterize_mesh(self, vertices: np.ndarray) -> np.ndarray:
        height, width = self._frame_size
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.rint(vertices).astype(np.int32)
        for triangle in self._faces:
            cv2.fillConvexPoly(mask, points[triangle], 255, cv2.LINE_AA)
        # A failed monocular pose can occasionally fold a few adjacent faces
        # away from the hand.  The physical MANO surface is connected, so the
        # largest projected component is the valid silhouette.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return mask
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == largest, 255, 0).astype(np.uint8)
