"""
Central configuration: paths, sampling, label taxonomy, and validation.

Everything the automated judge is strict about lives here:
  - exact JSON key names
  - ASCII-safe, lowercase label values
  - the closed set of allowed labels per category

If a label is not in these sets, it must NOT appear in results.json.
The mapping below is the single boundary where raw model output is
translated into competition labels. Keep all renaming here.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# 1. Competition paths (fixed by the spec). Do not change these defaults.
#    For local runs, override them with environment variables instead of
#    editing this file, e.g.
#      RS_INPUT_PATH=video.mp4 RS_OUTPUT_PATH=data/output/results.json \
#      RS_WEIGHTS_PATH=weights/ python3 main.py
# --------------------------------------------------------------------------
INPUT_PATH = os.environ.get("RS_INPUT_PATH", "/app/data/input/video.mp4")
OUTPUT_PATH = os.environ.get("RS_OUTPUT_PATH", "/app/data/output/results.json")
# VLM is loaded via `from_pretrained`, which expects a DIRECTORY
# containing config.json + safetensors + processor files — not a single
# checkpoint file. Keep this as a directory path.
WEIGHTS_PATH = os.environ.get("RS_WEIGHTS_PATH", "/app/weights/")

# --------------------------------------------------------------------------
# 2. Frame sampling.
#    Tesla T4 + 10 min wall-clock budget => we cannot run every frame.
#    SAMPLE_FPS is the temporal resolution of every detection's timestamp.
#    Lower = faster but coarser event timing. Tune against the budget.
# --------------------------------------------------------------------------
SAMPLE_FPS = 3.0          # frames analysed per second of video
MAX_FRAMES = 0            # 0 = no cap; set >0 to hard-limit for safety

# Wall-clock budget for the full pipeline. Container is killed at 10 min;
# leave headroom for OCR/color + JSON write. iter_frames widens its step
# on the fly if projected throughput won't finish under this budget.
TIME_BUDGET_S = 9 * 60

# --------------------------------------------------------------------------
# 3. Allowed label sets (the closed vocabulary the judge accepts).
# --------------------------------------------------------------------------
KATEGORI_SOFOR = "sofor_eylemi"
KATEGORI_NESNE = "nesneler"
KATEGORI_YOLCU = "yolcular"

ALLOWED_KATEGORI = {KATEGORI_SOFOR, KATEGORI_NESNE, KATEGORI_YOLCU}

ALLOWED_ACTIONS = {
    "arkaya_bakma", "esneme", "sigara_icme", "su_icme",
    "telefonla_konusma", "slalom", "etrafa_bakinma", "emniyet_kemeri_ihlali",
}
ALLOWED_OBJECTS = {"teknocan", "bilgisayar"}
ALLOWED_PASSENGERS = {"arka_koltuk_1", "arka_koltuk_2", "on_koltuk"}

ALLOWED_LABELS_BY_KATEGORI = {
    KATEGORI_SOFOR: ALLOWED_ACTIONS,
    KATEGORI_NESNE: ALLOWED_OBJECTS,
    KATEGORI_YOLCU: ALLOWED_PASSENGERS,
}

ALLOWED_VEHICLE_TYPES = {
    "sedan", "suv", "hatchback", "pickup", "minibus", "panelvan", "kamyon",
}
ALLOWED_COLORS = {
    "beyaz", "siyah", "gri", "kirmizi", "mavi",
    "sari", "yesil", "turuncu", "kahverengi",
}

# --------------------------------------------------------------------------
# 4. LABEL_MAP: raw model output -> competition label.
#    The fine-tuned VLM may emit English / verbose / variant tokens.
#    This is the ONLY place renaming happens. Add aliases freely.
#    Values MUST be members of the ALLOWED_* sets above.
# --------------------------------------------------------------------------
ACTION_LABEL_MAP = {
    "looking_back": "arkaya_bakma",
    "yawning": "esneme",
    "yawn": "esneme",
    "smoking": "sigara_icme",
    "cigarette": "sigara_icme",
    "drinking": "su_icme",
    "drinking_water": "su_icme",
    "phone": "telefonla_konusma",
    "using_phone": "telefonla_konusma",
    "phone_call": "telefonla_konusma",
    "looking_around": "etrafa_bakinma",
    "distracted": "etrafa_bakinma",
    # slalom + emniyet_kemeri_ihlali are DERIVED in reasoning.py, not mapped.
}
OBJECT_LABEL_MAP = {
    "laptop": "bilgisayar",
    "computer": "bilgisayar",
    "teknocan": "teknocan",
    "can": "teknocan",
}
VEHICLE_TYPE_MAP = {
    "sport_utility_vehicle": "suv",
    "suv": "suv",
    "sedan": "sedan",
    "hatchback": "hatchback",
    "pickup": "pickup",
    "pickup_truck": "pickup",
    "minibus": "minibus",
    "van": "panelvan",
    "panel_van": "panelvan",
    "panelvan": "panelvan",
    "truck": "kamyon",
    "kamyon": "kamyon",
}

# --------------------------------------------------------------------------
# 5. Detection / reasoning thresholds.
# --------------------------------------------------------------------------
MIN_CONFIDENCE = 0.40        # drop raw detections below this
SEATBELT_IOU = 0.10          # min overlap of a belt box with the driver box
SEATBELT_MIN_FRAMES = 5      # sustained frames of "no belt" before flagging
SLALOM_MIN_SWINGS = 3        # lateral direction changes to call it slalom
SLALOM_MIN_AMPLITUDE = 0.04  # min lateral swing as fraction of frame width

# --------------------------------------------------------------------------
# 6. Temporal consolidation policy, per category.
#    merge_gap_s : detections of the same label within this gap = one event.
#    min_frames  : require this many hits before emitting (false-positive guard).
#    collapse    : if True, the whole video yields at most ONE event per label
#                  (used for passengers — presence, not a momentary action).
#    pick        : "peak" -> timestamp of max-confidence hit; "onset" -> first.
# --------------------------------------------------------------------------
CONSOLIDATION = {
    KATEGORI_SOFOR: {"merge_gap_s": 2.0, "min_frames": 3, "collapse": False, "pick": "peak"},
    KATEGORI_NESNE: {"merge_gap_s": 3.0, "min_frames": 2, "collapse": False, "pick": "peak"},
    KATEGORI_YOLCU: {"merge_gap_s": 1e9, "min_frames": 3, "collapse": True,  "pick": "peak"},
}

# Confidence rounding in the final JSON.
CONF_DECIMALS = 2


def combine_vehicle_confidence(type_conf: float, plate_conf: float, color_conf: float) -> float:
    """One combined score for tip+plaka+renk, per spec section 3.

    Mean is the safe default (a single weak sub-estimate shouldn't zero the
    whole vehicle). Swap to a product if you want to punish any weak link.
    """
    return round((type_conf + plate_conf + color_conf) / 3.0, CONF_DECIMALS)


def is_ascii_lower(value: str) -> bool:
    return value.isascii() and value == value.lower()
