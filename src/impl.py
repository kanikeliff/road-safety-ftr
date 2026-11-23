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

import os
import re
from typing import List, Tuple
from unittest import mock

import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import dynamic_module_utils

from .perception import Box, Detection, PerceptionFrame

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


def _imports_without_flash_attn(get_imports):
    """Wrap transformers' `get_imports` so Florence's remote code doesn't
    demand `flash_attn`.

    modeling_florence2.py imports flash_attn only behind an
    `is_flash_attn_2_available()` guard, but transformers' static import
    check still lists it as required and refuses to load when the package
    is missing. The model is loaded with eager attention, so it's unused.
    """
    def _filtered(filename):
        imports = get_imports(filename)
        if str(filename).endswith("modeling_florence2.py"):
            imports = [imp for imp in imports if imp != "flash_attn"]
        return imports

    return _filtered


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
        self.device = device if torch.cuda.is_available() else "cpu"

        # Competition/Docker path is expected as /app/weights/.
        # Locally, config.WEIGHTS_PATH can temporarily point to "weights/".
        self.model_dir = os.path.join(
            self.weights_path,
            "vlm",
            "florence_finetuned",
        )

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Fine-tuned Florence model directory not found: {self.model_dir}"
            )

        # Use fp16 on CUDA for faster inference and lower memory use.
        # On CPU, keep fp32 because half precision is not reliably supported.
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Slow BartTokenizer needs merges.txt; fall back to the fast tokenizer
        # (tokenizer.json) when the saved checkpoint doesn't include it.
        use_fast = not os.path.isfile(os.path.join(self.model_dir, "merges.txt"))
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            use_fast=use_fast,
            local_files_only=True,
        )

        # attn_implementation="eager": standard attention, so flash_attn is
        # never needed (it isn't available on macOS/CPU).
        with mock.patch.object(
            dynamic_module_utils, "get_imports",
            _imports_without_flash_attn(dynamic_module_utils.get_imports),
        ):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                local_files_only=True,
                attn_implementation="eager",
            ).to(self.device)

        self.model.eval()

    def _frame_bgr_to_pil(self, frame_bgr) -> Image.Image:
        """OpenCV reads frames in BGR. Florence expects a PIL RGB image."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def _loc_tokens_to_box(self, locs: List[int]) -> Box:
        """Convert Florence <loc_0>..<loc_999> coordinates to normalized Box.

        PerceptionFrame expects normalized 0..1 coordinates, not pixel coords.
        """
        x1, y1, x2, y2 = locs

        return Box(
            x1=max(0.0, min(1.0, x1 / 999.0)),
            y1=max(0.0, min(1.0, y1 / 999.0)),
            x2=max(0.0, min(1.0, x2 / 999.0)),
            y2=max(0.0, min(1.0, y2 / 999.0)),
        )

    def _parse_generated_text(self, generated_text: str) -> List[Tuple[str, Box]]:
        """Parse fine-tuned Florence output.

        Expected examples after fine-tuning:
          telefon_gorunur<loc_326><loc_423><loc_402><loc_775>
          kemer_gorunur<loc_535><loc_700><loc_694><loc_998>

        Returns:
          [(label, normalized_box), ...]
        """
        pattern = r"([a-zA-Z_]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
        matches = re.findall(pattern, generated_text)

        parsed: List[Tuple[str, Box]] = []

        for label, x1, y1, x2, y2 in matches:
            label = label.lower().strip()
            box = self._loc_tokens_to_box([int(x1), int(y1), int(x2), int(y2)])

            # Ignore malformed boxes. This prevents downstream geometry issues.
            if box.x2 <= box.x1 or box.y2 <= box.y1:
                continue

            parsed.append((label, box))

        return parsed

    def analyze_frame(self, frame_bgr, t: float) -> PerceptionFrame:
        """Run the VLM on `frame_bgr` and return a populated PerceptionFrame.

        Current fine-tuned model focuses on:
          * telefon_gorunur -> driver_actions / phone
          * kemer_gorunur   -> seatbelts / seatbelt

        The reasoning layer later maps:
          phone -> telefonla_konusma

        Seatbelt violation is not emitted directly by the model. It is derived
        in reasoning.py when a driver/person exists and no belt overlaps it.
        """
        image = self._frame_bgr_to_pil(frame_bgr)
        task_prompt = "<OD>"

        inputs = self.processor(
            text=task_prompt,
            images=image,
            return_tensors="pt",
        )

        inputs = {
            "input_ids": inputs["input_ids"].to(self.device),
            "pixel_values": inputs["pixel_values"].to(self.device, dtype=self.dtype),
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        parsed_detections = self._parse_generated_text(generated_text)

        pf = PerceptionFrame(t=t)

        # The fine-tuned labels are Turkish/ASCII internal labels.
        # We convert them back to raw labels expected by config.*_LABEL_MAP.
        for label, box in parsed_detections:
            # Score is not directly produced by generation.
            # Use a fixed confidence for now; can be refined later with
            # beam scores or a separate confidence heuristic.
            score = 0.80

            if label == "telefon_gorunur":
                pf.driver_actions.append(
                    Detection(label="phone", box=box, score=score)
                )

            elif label == "kemer_gorunur":
                pf.seatbelts.append(
                    Detection(label="seatbelt", box=box, score=score)
                )

            # Future labels can be routed here:
            # elif label == "sigara":
            #     pf.driver_actions.append(
            #         Detection(label="cigarette", box=box, score=score)
            #     )
            # elif label == "bilgisayar":
            #     pf.objects.append(
            #         Detection(label="laptop", box=box, score=score)
            #     )

        return pf


# --------------------------------------------------------------------------
# 2) PlateOCR — read a plate string from a cropped plate image.
# --------------------------------------------------------------------------
class PlateOCR:
    """Plate OCR. Pass your raw OCR string through `normalize_plate` so a
    malformed read becomes PLATE_UNREADABLE instead of leaking into JSON.
    """

    def read(self, plate_crop) -> Tuple[str, float]:
        # MVP fallback: OCR is not integrated yet.
        # Return the spec-approved unreadable sentinel so results.json remains valid.
        return PLATE_UNREADABLE, 0.0


# --------------------------------------------------------------------------
# 3) ColorClassifier — predict a vehicle color from a vehicle crop.
# --------------------------------------------------------------------------
class ColorClassifier:
    """Vehicle color classifier. Return one of config.ALLOWED_COLORS or
    fall back to 'gri' so the result stays inside the closed vocabulary.
    """

    def predict(self, vehicle_crop) -> Tuple[str, float]:
        # MVP fallback: color classifier is not integrated yet.
        # 'gri' is in the allowed color vocabulary and keeps validation safe.
        return "gri", 0.0