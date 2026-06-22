"""
Perception schema.

The dataclasses below are the model-agnostic contract between the VLM
(see impl.VLM) and the rest of the pipeline. All box coords are normalized
0..1 so resolution/FPS changes don't matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: "Box") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    """A generic raw detection: a label + box + score, pre-translation."""
    label: str          # raw model label (English/variant ok; mapped later)
    box: Box
    score: float


@dataclass
class PerceptionFrame:
    """Everything the model saw in one sampled frame, at time `t` seconds."""
    t: float
    vehicle_box: Optional[Box] = None
    vehicle_type: Optional[str] = None         # raw type label
    vehicle_type_score: float = 0.0
    plate_box: Optional[Box] = None
    plate_score: float = 0.0
    persons: List[Detection] = field(default_factory=list)     # raw "person" boxes
    seatbelts: List[Detection] = field(default_factory=list)   # raw "seat_belt" boxes
    driver_actions: List[Detection] = field(default_factory=list)  # raw action labels
    objects: List[Detection] = field(default_factory=list)     # raw in-cabin objects


class PerceptionModel(Protocol):
    """Any backend implementing this protocol can be plugged into pipeline."""
    def analyze_frame(self, frame_bgr, t: float) -> PerceptionFrame: ...
