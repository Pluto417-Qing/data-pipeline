from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .backends import HandSegmentationBackend, create_backend
from .refined_backend import RefinedMediaPipeBackend


@dataclass(frozen=True)
class ProcessingBranch:
    """A named, reproducible route from frames to hand masks."""

    name: str
    description: str

    def create_backend(self) -> HandSegmentationBackend:
        if self.name == "classic":
            # Preserve the original pipeline exactly: landmarks -> geometric mask.
            return create_backend("mediapipe")
        if self.name == "refined":
            return RefinedMediaPipeBackend()
        if self.name == "sam2":
            return create_backend("sam2")
        if self.name == "xmem2":
            return create_backend("xmem2")
        raise ValueError(f"Unknown processing branch '{self.name}'")


BRANCHES = {
    "classic": ProcessingBranch("classic", "Original MediaPipe landmark-to-mask pipeline."),
    "refined": ProcessingBranch("refined", "MediaPipe initial mask followed by constrained GrabCut boundary refinement."),
    "sam2": ProcessingBranch("sam2", "MediaPipe prompt masks propagated through SAM 2 video tracking."),
    "xmem2": ProcessingBranch("xmem2", "Sparse MediaPipe hand masks propagated by XMem++ permanent-memory video segmentation."),
}


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"


def default_branch_name() -> str:
    path = _config_path()
    if not path.exists():
        return "classic"
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    name = str(config.get("active_branch", "classic")).lower()
    if name not in BRANCHES:
        raise ValueError(f"config/pipeline.yaml has unknown active_branch '{name}'. Available: {', '.join(BRANCHES)}")
    return name


def resolve_branch(name: str | None) -> ProcessingBranch:
    selected = (name or default_branch_name()).lower()
    try:
        return BRANCHES[selected]
    except KeyError as error:
        raise ValueError(f"Unknown processing branch '{selected}'. Available: {', '.join(BRANCHES)}") from error


def add_branch_metadata(sample_dir: Path, branch: ProcessingBranch) -> None:
    """Record the selected route without changing legacy pipeline.py behavior."""
    path = sample_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["processing_branch"] = branch.name
    metadata["processing_branch_description"] = branch.description
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
