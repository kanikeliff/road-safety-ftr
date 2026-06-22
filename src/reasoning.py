"""
Reasoning layer.

Turns raw perception (boxes + raw labels) into competition labels. This is
where the THREE derived labels live — the ones no detector outputs directly:

  * emniyet_kemeri_ihlali : driver present AND no seatbelt on the driver box.
  * on_koltuk / arka_koltuk_* : passenger seat from person-box geometry.
  * slalom : oscillation of the vehicle centroid over time.

Direct labels (phone, yawn, laptop, ...) are just renamed via config maps.
Output is a flat list of RawEvent — one per (frame, label) hit. Collapsing
these noisy per-frame hits into discrete events is aggregation.py's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import config
from .perception import Box, Detection, PerceptionFrame


@dataclass
class RawEvent:
    t: float
    kategori: str
    etiket: str
    score: float


# --------------------------------------------------------------------------
# Driver identification: the front-left occupant. The fine-tuned model may
# instead tag the driver directly — if so, replace this heuristic with that.
# --------------------------------------------------------------------------
def _pick_driver(persons: List[Detection]) -> Optional[Detection]:
    if not persons:
        return None
    # Front row = smaller cy (higher in frame); driver = leftmost of front row.
    front_cy = min(p.box.cy for p in persons)
    front_row = [p for p in persons if p.box.cy <= front_cy + 0.12]
    return min(front_row, key=lambda p: p.box.cx)


# --------------------------------------------------------------------------
# Passenger seat assignment from a person box (driver already removed).
# HEURISTIC — tune the splits, or bypass entirely if the model emits seats.
#   front/back  : by vertical position (cy)
#   left/right  : by horizontal position (cx)
# --------------------------------------------------------------------------
FRONT_BACK_SPLIT = 0.52   # cy above this = front row, below = back row
LEFT_RIGHT_SPLIT = 0.50   # cx left of this = left side


def _seat_label(box: Box) -> str:
    is_back = box.cy > FRONT_BACK_SPLIT
    is_left = box.cx < LEFT_RIGHT_SPLIT
    if not is_back:
        return "on_koltuk"                       # any front passenger
    return "arka_koltuk_1" if is_left else "arka_koltuk_2"


def _driver_belted(driver: Detection, belts: List[Detection]) -> bool:
    return any(b.box.iou(driver.box) >= config.SEATBELT_IOU for b in belts)


def derive_events(frames: List[PerceptionFrame]) -> List[RawEvent]:
    events: List[RawEvent] = []

    for f in frames:
        driver = _pick_driver(f.persons)

        # --- direct driver actions (phone, yawn, ...) ---
        for d in f.driver_actions:
            etiket = config.ACTION_LABEL_MAP.get(d.label.lower())
            if etiket and etiket in config.ALLOWED_ACTIONS and d.score >= config.MIN_CONFIDENCE:
                events.append(RawEvent(f.t, config.KATEGORI_SOFOR, etiket, d.score))

        # --- derived: seatbelt violation ---
        if driver is not None and not _driver_belted(driver, f.seatbelts):
            # confidence ~ how sure we are the driver is there
            events.append(RawEvent(f.t, config.KATEGORI_SOFOR,
                                   "emniyet_kemeri_ihlali", round(driver.score, 3)))

        # --- in-cabin objects (laptop -> bilgisayar, can -> teknocan) ---
        for o in f.objects:
            etiket = config.OBJECT_LABEL_MAP.get(o.label.lower())
            if etiket and etiket in config.ALLOWED_OBJECTS and o.score >= config.MIN_CONFIDENCE:
                events.append(RawEvent(f.t, config.KATEGORI_NESNE, etiket, o.score))

        # --- derived: passenger seats (everyone who isn't the driver) ---
        for p in f.persons:
            if p is driver:
                continue
            if p.score < config.MIN_CONFIDENCE:
                continue
            events.append(RawEvent(f.t, config.KATEGORI_YOLCU, _seat_label(p.box), p.score))

    return events


def detect_sequence_events(frames: List[PerceptionFrame]) -> List[dict]:
    """Sequence-level (already-consolidated) events that must NOT pass through
    the per-frame consolidator. Currently: slalom. Returns final tespit dicts."""
    out: List[dict] = []
    slalom = _detect_slalom(frames)
    if slalom is not None:
        out.append({
            "zaman_saniye": round(slalom.t, 2),
            "kategori": slalom.kategori,
            "etiket": slalom.etiket,
            "confidence_score": round(slalom.score, config.CONF_DECIMALS),
        })
    return out


def _detect_slalom(frames: List[PerceptionFrame]) -> Optional[RawEvent]:
    """Count alternating lateral swings of the vehicle centroid above a
    minimum amplitude. >= SLALOM_MIN_SWINGS direction changes => slalom."""
    track = [(f.t, f.vehicle_box.cx) for f in frames if f.vehicle_box is not None]
    if len(track) < 5:
        return None

    # Find turning points (local extrema) in cx.
    extrema = []  # (t, cx)
    for i in range(1, len(track) - 1):
        prev_cx, cur_cx, next_cx = track[i - 1][1], track[i][1], track[i + 1][1]
        if (cur_cx - prev_cx) * (next_cx - cur_cx) < 0:   # direction flips here
            extrema.append(track[i])

    # Count swings whose peak-to-peak amplitude clears the threshold.
    swings, peak_t, peak_amp = 0, None, 0.0
    for j in range(1, len(extrema)):
        amp = abs(extrema[j][1] - extrema[j - 1][1])
        if amp >= config.SLALOM_MIN_AMPLITUDE:
            swings += 1
            if amp > peak_amp:
                peak_amp, peak_t = amp, extrema[j][0]

    if swings >= config.SLALOM_MIN_SWINGS and peak_t is not None:
        conf = round(min(0.95, 0.6 + 0.1 * swings), 3)
        return RawEvent(peak_t, config.KATEGORI_SOFOR, "slalom", conf)
    return None
