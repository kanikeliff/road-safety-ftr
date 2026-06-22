"""
Container entry point.

    python3 main.py
  -> reads  /app/data/input/video.mp4
  -> writes /app/data/output/results.json
"""

import json
import os
import sys

# Make `src` importable whether run as `python3 main.py` or from elsewhere.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src import config
from src.impl import ColorClassifier, PlateOCR
from src.pipeline import run_inference


def main() -> None:
    input_path = config.INPUT_PATH
    out_path = config.OUTPUT_PATH

    if not os.path.exists(input_path):
        print(f"Hata: Girdi videosu bulunamadi -> {input_path}")
        sys.exit(1)

    print("Yol Guvenligi Yapay Zeka Cikarim Islemi Baslatildi...")
    try:
        results = run_inference(
            video_path=input_path,
            weights_path=config.WEIGHTS_PATH,
            ocr=PlateOCR(),
            color=ColorClassifier(),
            video_id=os.path.basename(input_path),
        )
    except Exception as e:                     # never crash the container
        print(f"Model calistirilirken bir hata olustu: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Islem tamamlandi. Cikti kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
