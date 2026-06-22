"""
Frame sampling. Yields (frame_bgr, t_seconds) at the configured sample rate.

`iter_frames` adapts under a wall-clock budget: if inference is slower than
the video's playback time, the step grows so we still cover the whole clip
at a coarser FPS instead of getting killed partway through.
"""

from __future__ import annotations

import math
import time
from typing import Iterator, Optional, Tuple

from . import config


# Re-check the budget / projected throughput every N emitted frames.
_ADAPT_EVERY = 10


def _safe_fps(raw: float) -> float:
    """OpenCV returns 0 or NaN for some containers; fall back to 30 fps."""
    if raw is None or raw <= 0 or math.isnan(raw) or math.isinf(raw):
        return 30.0
    return float(raw)


def iter_frames(video_path: str,
                sample_fps: float = config.SAMPLE_FPS,
                max_frames: int = config.MAX_FRAMES,
                time_budget_s: Optional[float] = None,
                ) -> Iterator[Tuple[object, float]]:
    """Decode `video_path` and yield ~`sample_fps` frames/sec of video.

    If `time_budget_s` is set, the sample step is widened on the fly when
    wall-clock projection says we won't finish — degrading temporal
    resolution gracefully rather than truncating the video.
    """
    import cv2  # imported lazily so config import never pulls OpenCV in.

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    src_fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_duration_s = total_frames / src_fps if total_frames > 0 else 0.0

    base_step = max(1, int(round(src_fps / max(sample_fps, 1e-6))))
    step = base_step

    start_t = time.monotonic()
    idx, emitted, last_adapt = 0, 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield frame, idx / src_fps
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break

                # --- adaptive degradation ---
                if (time_budget_s
                        and total_duration_s > 0
                        and emitted - last_adapt >= _ADAPT_EVERY):
                    last_adapt = emitted
                    elapsed = time.monotonic() - start_t
                    remaining_budget = time_budget_s - elapsed
                    if remaining_budget <= 0:
                        break                              # out of time, stop early
                    cur_t = idx / src_fps
                    remaining_video_s = total_duration_s - cur_t
                    if remaining_video_s <= 0:
                        continue
                    time_per_emit = elapsed / emitted
                    affordable = int(remaining_budget / max(time_per_emit, 1e-6))
                    needed = int(remaining_video_s * src_fps / step)
                    if affordable > 0 and needed > affordable:
                        # Widen the step to fit the remaining budget.
                        step = max(step + 1, int(step * needed / affordable))
            idx += 1
    finally:
        cap.release()
