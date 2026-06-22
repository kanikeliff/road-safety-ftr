"""
============================================================================
 FILL THIS FILE. Nothing else needs to change for a standard submission.
============================================================================

Three classes for you to implement. Until then each method raises
NotImplementedError — the pipeline fails loud rather than shipping
fabricated detections.

  VLM              : your vision-language model wrapper
  PlateOCR         : plate-bbox crop  -> (plate_string, confidence)
  ColorClassifier  : vehicle-bbox crop -> (color, confidence)

Helpers in this file (you don't have to touch these — use them):
  normalize_plate  : enforces the Turkish plate regex; returns
                     PLATE_UNREADABLE if the OCR string is malformed.
  PLATE_UNREADABLE : the sentinel the spec mandates ("tespit edilemedi").
"""

from __future__ import annotations

import re
from typing import Tuple

from .perception import PerceptionFrame

# --------------------------------------------------------------------------
# Plate normalization helpers — use these in PlateOCR.read().
# --------------------------------------------------------------------------
# Spec plate regex (allows optional single spaces around the letter group).
PLATE_RE = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"((\s?[a-zA-Z]\s?)(\d{4,5})"
    r"|(\s?[a-zA-Z]{2}\s?)(\d{3,4})"
    r"|(\s?[a-zA-Z]{3}\s?)(\d{2,3}))$"
)
PLATE_UNREADABLE = "tespit edilemedi"


def normalize_plate(raw: str) -> str:
    """Uppercase, strip spaces, validate. Returns compact plate or the
    spec's 'tespit edilemedi' sentinel if it fails the regex."""
    if not raw:
        return PLATE_UNREADABLE
    compact = re.sub(r"\s+", "", raw).upper()
    return compact if PLATE_RE.match(compact) else PLATE_UNREADABLE


# --------------------------------------------------------------------------
# 1) VLM — the only module that talks to the vision-language model.
# --------------------------------------------------------------------------
class VLM:
    """Vision-language model wrapper.

    Implementation notes:
      * `weights_path` is a DIRECTORY containing config.json + safetensors +
        processor files, loaded via AutoProcessor.from_pretrained /
        AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True).
        NOT a single .pt checkpoint.
      * Move the model to cuda and call .eval() in __init__.
      * In analyze_frame: run your fine-tuned task prompts on `frame_bgr`,
        parse the structured output, and pack everything into a
        PerceptionFrame (see src/perception.py for the schema).
      * Keep raw labels — translation to competition labels happens later
        in reasoning.py via config.*_LABEL_MAP.
    """

    def __init__(self, weights_path: str, device: str = "cuda"):
        self.weights_path = weights_path
        self.device = device
        raise NotImplementedError(
            "Load AutoProcessor + AutoModelForCausalLM from weights_path, "
            ".to(device), .eval()."
        )

    def analyze_frame(self, frame_bgr, t: float) -> PerceptionFrame:
        """Run the VLM on `frame_bgr` and return a populated PerceptionFrame."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# 2) PlateOCR — read a plate string from a cropped plate image.
# --------------------------------------------------------------------------
class PlateOCR:
    """Plate OCR. Pass your raw OCR string through `normalize_plate` so a
    malformed read becomes PLATE_UNREADABLE instead of leaking into JSON.
    """

    def read(self, plate_crop) -> Tuple[str, float]:
        raise NotImplementedError


# --------------------------------------------------------------------------
# 3) ColorClassifier — predict a vehicle color from a vehicle crop.
# --------------------------------------------------------------------------
class ColorClassifier:
    """Vehicle color classifier. Return one of config.ALLOWED_COLORS or
    fall back to 'gri' so the result stays inside the closed vocabulary.
    """

    def predict(self, vehicle_crop) -> Tuple[str, float]:
        raise NotImplementedError
