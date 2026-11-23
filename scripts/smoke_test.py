"""
Local smoke test: load the VLM once and run it on ONE frame.

Faster than main.py for checking that the model loads and produces sane
detections before running the whole video.

    RS_WEIGHTS_PATH=weights/ python3 scripts/smoke_test.py --video video.mp4 --t 1.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.impl import VLM  # noqa: E402


def _fmt(box) -> str:
    return "-" if box is None else f"({box.x1:.2f},{box.y1:.2f})-({box.x2:.2f},{box.y2:.2f})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=config.INPUT_PATH)
    ap.add_argument("--t", type=float, default=1.0, help="timestamp in seconds")
    ap.add_argument("--weights", default=config.WEIGHTS_PATH)
    args = ap.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[FAIL] cannot open video: {args.video}")
        return 1
    cap.set(cv2.CAP_PROP_POS_MSEC, args.t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"[FAIL] no frame at t={args.t}s")
        return 1
    print(f"[ OK ] frame {frame.shape[1]}x{frame.shape[0]} at t={args.t}s")

    t0 = time.monotonic()
    vlm = VLM(args.weights)
    print(f"[ OK ] model loaded in {time.monotonic() - t0:.1f}s on "
          f"{getattr(vlm, 'device', '?')} ({getattr(vlm, 'dtype', '?')})")

    t0 = time.monotonic()
    pf = vlm.analyze_frame(frame, args.t)
    print(f"[ OK ] inference in {time.monotonic() - t0:.1f}s")

    print(f"  vehicle : {pf.vehicle_type} {_fmt(pf.vehicle_box)}")
    print(f"  plate   : {_fmt(pf.plate_box)}")
    for name in ("persons", "seatbelts", "driver_actions", "objects"):
        dets = getattr(pf, name)
        print(f"  {name:<14}: {len(dets)} " + " ".join(d.label for d in dets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
