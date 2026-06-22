"""
Orchestrator. Wires perception -> reasoning -> aggregation -> JSON.

`run_inference()` is the single entry point main.py calls. It returns the
final results dict (already validated).
"""

from __future__ import annotations

from typing import List

from . import aggregation, config, video_io
from .impl import ColorClassifier, PlateOCR, VLM
from .perception import Box, PerceptionFrame
from .reasoning import derive_events, detect_sequence_events
from .validate import validate_results


def _crop(frame_bgr, box: Box):
    """Crop a normalized box out of a BGR frame."""
    if frame_bgr is None or box is None:
        return None
    h, w = frame_bgr.shape[:2]
    x1, y1 = int(box.x1 * w), int(box.y1 * h)
    x2, y2 = int(box.x2 * w), int(box.y2 * h)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return frame_bgr[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None


def run_inference(
    video_path: str,
    weights_path: str = config.WEIGHTS_PATH,
    *,
    ocr: PlateOCR,
    color: ColorClassifier,
    video_id: str = "video.mp4",
) -> dict:
    model = VLM(weights_path)

    # --- perception pass: build the per-frame world view ---
    # Rolling-best crops kept in O(1) memory. Caching every raw frame
    # (hundreds of full-res frames per video) blows the 16 GB cap on real
    # inputs — we only ever need the single best plate/vehicle crop, so
    # crop greedily as we go and throw the raw frame away.
    frames: List[PerceptionFrame] = []
    best_plate_crop, best_plate_score = None, -1.0
    best_vehicle_crop, best_vehicle_area = None, -1.0

    for frame_bgr, t in video_io.iter_frames(video_path, time_budget_s=config.TIME_BUDGET_S):
        pf = model.analyze_frame(frame_bgr, t)
        frames.append(pf)

        if pf.plate_box is not None and pf.plate_score > best_plate_score:
            crop = _crop(frame_bgr, pf.plate_box)
            if crop is not None:
                best_plate_crop = crop
                best_plate_score = pf.plate_score

        if pf.vehicle_box is not None and pf.vehicle_box.area > best_vehicle_area:
            crop = _crop(frame_bgr, pf.vehicle_box)
            if crop is not None:
                best_vehicle_crop = crop
                best_vehicle_area = pf.vehicle_box.area

    if not frames:
        raise RuntimeError("No frames decoded from the input video.")

    # --- reasoning -> raw events -> consolidated tespitler ---
    tespitler = aggregation.consolidate_events(derive_events(frames))
    tespitler += detect_sequence_events(frames)        # slalom etc. (pre-consolidated)
    tespitler.sort(key=lambda d: d["zaman_saniye"])

    # --- vehicle summary (one main vehicle) ---
    tip, tip_conf = aggregation.vote_vehicle_type(frames)

    plate_str, plate_conf = "", 0.0
    if best_plate_crop is not None:
        plate_str, plate_conf = ocr.read(best_plate_crop)

    renk, renk_conf = "gri", 0.0
    if best_vehicle_crop is not None:
        renk, renk_conf = color.predict(best_vehicle_crop)

    arac_bilgisi = {
        "tip": tip,
        "plaka": plate_str,
        "renk": renk,
        "confidence_score": config.combine_vehicle_confidence(tip_conf, plate_conf, renk_conf),
    }

    results = {
        "video_id": video_id,
        "arac_bilgisi": arac_bilgisi,
        "tespitler": tespitler,
    }
    validate_results(results)                # fail loudly before writing
    return results
