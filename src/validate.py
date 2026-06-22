"""
Final safety net: validate results.json against the spec before it ships.
Catches the exact things the judge fails on — wrong keys, non-ASCII labels,
out-of-vocabulary values, bad confidence ranges.
Raises ValueError on the first problem.
"""

from __future__ import annotations

from typing import Any, Dict

from . import config
from .impl import PLATE_UNREADABLE


def _check_conf(value: Any, where: str) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{where}: confidence_score must be a number, got {value!r}")
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{where}: confidence_score {value} out of [0,1]")


def validate_results(r: Dict[str, Any]) -> None:
    if "video_id" not in r:
        raise ValueError("missing 'video_id'")

    # --- arac_bilgisi ---
    av = r.get("arac_bilgisi")
    if not isinstance(av, dict):
        raise ValueError("missing/invalid 'arac_bilgisi'")
    for key in ("tip", "plaka", "renk", "confidence_score"):
        if key not in av:
            raise ValueError(f"arac_bilgisi missing key '{key}'")
    if av["tip"] not in config.ALLOWED_VEHICLE_TYPES:
        raise ValueError(f"arac_bilgisi.tip '{av['tip']}' not allowed")
    if av["renk"] not in config.ALLOWED_COLORS:
        raise ValueError(f"arac_bilgisi.renk '{av['renk']}' not allowed")
    if av["plaka"] != PLATE_UNREADABLE and not config.is_ascii_lower(av["tip"]):
        raise ValueError("arac_bilgisi.tip must be ascii lowercase")
    _check_conf(av["confidence_score"], "arac_bilgisi")

    # --- tespitler ---
    dets = r.get("tespitler")
    if not isinstance(dets, list):
        raise ValueError("missing/invalid 'tespitler' (must be a list)")
    for i, d in enumerate(dets):
        where = f"tespitler[{i}]"
        for key in ("zaman_saniye", "kategori", "etiket", "confidence_score"):
            if key not in d:
                raise ValueError(f"{where} missing key '{key}'")
        kat, eti = d["kategori"], d["etiket"]
        if kat not in config.ALLOWED_KATEGORI:
            raise ValueError(f"{where}: kategori '{kat}' not allowed")
        if eti not in config.ALLOWED_LABELS_BY_KATEGORI[kat]:
            raise ValueError(f"{where}: etiket '{eti}' not valid for '{kat}'")
        if not config.is_ascii_lower(eti):
            raise ValueError(f"{where}: etiket '{eti}' must be ascii lowercase")
        if not isinstance(d["zaman_saniye"], (int, float)):
            raise ValueError(f"{where}: zaman_saniye must be numeric")
        _check_conf(d["confidence_score"], where)
