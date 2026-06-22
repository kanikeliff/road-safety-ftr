"""
Aggregation layer.

Two jobs:
  1. consolidate_events: collapse noisy per-frame RawEvents into the discrete
     `tespitler` the judge expects (one timestamp per event), using the
     per-category policy in config.CONSOLIDATION.
  2. vehicle summary: the scene has ONE main vehicle, so type/plate/color are
     decided ONCE over the whole video — vote the type, pick the best crops to
     hand to the OCR/color modules.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from . import config
from .perception import PerceptionFrame
from .reasoning import RawEvent


# --------------------------------------------------------------------------
# Temporal consolidation -> tespitler entries.
# --------------------------------------------------------------------------
def consolidate_events(raw_events: List[RawEvent]) -> List[dict]:
    by_label: Dict[Tuple[str, str], List[RawEvent]] = defaultdict(list)
    for e in raw_events:
        by_label[(e.kategori, e.etiket)].append(e)

    tespitler: List[dict] = []
    for (kategori, etiket), hits in by_label.items():
        # Validation gate: never emit a label outside the allowed set.
        if kategori not in config.ALLOWED_KATEGORI:
            continue
        if etiket not in config.ALLOWED_LABELS_BY_KATEGORI[kategori]:
            continue

        policy = config.CONSOLIDATION[kategori]
        hits.sort(key=lambda e: e.t)

        # Split into runs separated by a gap larger than merge_gap_s.
        runs: List[List[RawEvent]] = []
        if policy["collapse"]:
            runs = [hits]                      # whole video = one event per label
        else:
            cur = [hits[0]]
            for e in hits[1:]:
                if e.t - cur[-1].t <= policy["merge_gap_s"]:
                    cur.append(e)
                else:
                    runs.append(cur)
                    cur = [e]
            runs.append(cur)

        for run in runs:
            if len(run) < policy["min_frames"]:
                continue                        # too brief -> likely false positive
            if policy["pick"] == "onset":
                chosen = run[0]
            else:                               # peak confidence
                chosen = max(run, key=lambda e: e.score)
            tespitler.append({
                "zaman_saniye": round(chosen.t, 2),
                "kategori": kategori,
                "etiket": etiket,
                "confidence_score": round(chosen.score, config.CONF_DECIMALS),
            })

    tespitler.sort(key=lambda d: d["zaman_saniye"])
    return tespitler


# --------------------------------------------------------------------------
# Vehicle summary helpers.
# --------------------------------------------------------------------------
def vote_vehicle_type(frames: List[PerceptionFrame]) -> Tuple[str, float]:
    """Confidence-weighted vote across frames, mapped to a competition type."""
    score_sum: Dict[str, float] = defaultdict(float)
    best_conf: Dict[str, float] = defaultdict(float)
    for f in frames:
        if not f.vehicle_type:
            continue
        etiket = config.VEHICLE_TYPE_MAP.get(f.vehicle_type.lower())
        if etiket not in config.ALLOWED_VEHICLE_TYPES:
            continue
        score_sum[etiket] += f.vehicle_type_score
        best_conf[etiket] = max(best_conf[etiket], f.vehicle_type_score)

    if not score_sum:
        # Fallback keeps output valid; choose a neutral common type.
        return "sedan", 0.0
    winner = max(score_sum, key=score_sum.get)
    return winner, round(best_conf[winner], 3)


def select_best_plate_frame(frames: List[PerceptionFrame]) -> Optional[PerceptionFrame]:
    """Frame with the highest-confidence plate box (most readable crop)."""
    cands = [f for f in frames if f.plate_box is not None]
    return max(cands, key=lambda f: f.plate_score) if cands else None


def select_best_vehicle_frame(frames: List[PerceptionFrame]) -> Optional[PerceptionFrame]:
    """Frame with the largest vehicle box (best crop for color)."""
    cands = [f for f in frames if f.vehicle_box is not None]
    return max(cands, key=lambda f: f.vehicle_box.area) if cands else None
