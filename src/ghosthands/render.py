from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RenderStyle:
    inner_alpha: float
    rim_width: int
    rim_alpha: float
    glow_radius: int
    glow_alpha: float
    cyan_bgr: tuple[int, int, int] = (255, 190, 65)
    rim_bgr: tuple[int, int, int] = (255, 245, 205)

    def to_dict(self) -> dict:
        return asdict(self)


def sample_style(seed: int) -> RenderStyle:
    rng = np.random.default_rng(seed)
    return RenderStyle(
        inner_alpha=float(rng.uniform(0.27, 0.42)),
        rim_width=int(rng.integers(2, 5)),
        rim_alpha=float(rng.uniform(0.72, 0.9)),
        glow_radius=int(rng.integers(13, 29)) | 1,
        glow_alpha=float(rng.uniform(0.30, 0.52)),
    )


def render_glow(bgr_frame: np.ndarray, combined_mask: np.ndarray, style: RenderStyle) -> np.ndarray:
    """Composite a hand-attached cyan tint, rim, and soft glow over a source frame."""
    frame = bgr_frame.astype(np.float32)
    mask = (combined_mask.astype(np.float32) / 255.0)[..., None]
    cyan = np.full_like(frame, style.cyan_bgr, dtype=np.float32)
    # Screen blend keeps skin texture and highlights visible through the tint.
    screened = 255.0 - (255.0 - frame) * (255.0 - cyan) / 255.0
    frame = frame * (1.0 - mask * style.inner_alpha) + screened * (mask * style.inner_alpha)

    kernel_size = max(3, style.rim_width * 2 + 1)
    dilated = cv2.dilate(combined_mask, np.ones((kernel_size, kernel_size), np.uint8))
    eroded = cv2.erode(combined_mask, np.ones((kernel_size, kernel_size), np.uint8))
    rim = cv2.subtract(dilated, eroded).astype(np.float32) / 255.0
    glow = cv2.GaussianBlur(dilated, (style.glow_radius, style.glow_radius), 0).astype(np.float32) / 255.0

    frame = frame * (1.0 - glow[..., None] * style.glow_alpha) + cyan * (glow[..., None] * style.glow_alpha)
    rim_color = np.full_like(frame, style.rim_bgr, dtype=np.float32)
    frame = frame * (1.0 - rim[..., None] * style.rim_alpha) + rim_color * (rim[..., None] * style.rim_alpha)
    return np.clip(frame, 0, 255).astype(np.uint8)
