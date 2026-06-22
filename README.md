# Akıllı Yol Güvenliği — FTR İnferans Sistemi

A template for the competition. One video in
(`/app/data/input/video.mp4`), one `results.json` out
(`/app/data/output/results.json`), conforming exactly to the spec schema.

The entire perception/reasoning/aggregation/validation pipeline is wired
up. To make it run, fork the repo and fill in **one file**.

## What you fill in: `src/impl.py`

Three classes, none of them long:

| Class             | What it does                                          |
| ----------------- | ----------------------------------------------------- |
| `VLM`             | Loads your fine-tuned vision-language model, runs it on a frame, returns a `PerceptionFrame` (schema in `src/perception.py`). |
| `PlateOCR`        | Crop in → `(plate_string, confidence)`. Helpers `normalize_plate` and `PLATE_UNREADABLE` are in the same file. |
| `ColorClassifier` | Vehicle crop in → `(color, confidence)`. Color must be in `config.ALLOWED_COLORS`. |

Drop your fine-tuned weights at `weights/vlm/` (a `from_pretrained`
directory: `config.json` + `*.safetensors` + processor files). That's it
— `python3 main.py` reads the video, runs the pipeline, writes the JSON.

Until you implement the stubs, the pipeline raises NotImplementedError
loudly — no fabricated detections leak out.

## Architecture

```
video ─► video_io ─► perception (VLM) ─► reasoning ─► aggregation ─► results.json
                       raw boxes/labels    derived     consolidate    + validate
```

- **impl.py** — *fill this in.* VLM + PlateOCR + ColorClassifier + plate helper.
- **perception.py** — the dataclass schema your `VLM.analyze_frame` returns.
- **reasoning.py** — derives the labels no detector emits directly:
  `emniyet_kemeri_ihlali` (driver present + no belt), `on_koltuk` /
  `arka_koltuk_*` (seat from person-box geometry), `slalom` (centroid
  trajectory). Direct labels (phone, yawn, laptop…) are renamed via maps.
- **aggregation.py** — collapses noisy per-frame hits into discrete
  `tespitler` events, votes the single vehicle's type, picks best crops
  for OCR + color.
- **video_io.py** — frame sampling with a wall-clock budget that
  auto-degrades FPS if inference is projected to overrun.
- **config.py** — paths, sampling, time budget, the closed label
  vocabulary, `LABEL_MAP`, thresholds, consolidation policy.
- **validate.py** — final schema/label/ASCII/confidence gate before writing.
- **pipeline.py** — orchestrator that wires it all together.

## Run

```bash
python3 main.py               # reads /app/data/input, writes /app/data/output
```

## Before submission

1. Fill in `src/impl.py`.
2. Drop the VLM `from_pretrained` directory at `weights/vlm/`.
3. Confirm GPU usage (model `.to("cuda")`, `torch.cuda.is_available()`).
4. Tune `SAMPLE_FPS` / `TIME_BUDGET_S` in `config.py` against a real clip
   — the pipeline auto-degrades FPS if it's projected to overrun.
5. Round out `LABEL_MAP` in `config.py` once you see the raw tokens your
   fine-tuned VLM actually emits.

## Decision points worth revisiting

- **Sampling rate vs. event timing** (`SAMPLE_FPS`).
- **Consolidation policy** per category (`config.CONSOLIDATION`) — how
  aggressively to merge repeated hits into one event.
- **Seat geometry splits** (`reasoning.FRONT_BACK_SPLIT/LEFT_RIGHT_SPLIT`) —
  ideally replaced by the model emitting seat labels directly.
- **Confidence combination** for `arac_bilgisi` (mean vs. product).
