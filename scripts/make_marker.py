"""Generate a printable ArUco reference marker of a known physical size.

Usage: python scripts/make_marker.py --size-cm 15 --id 7
Print at 100% scale (no 'fit to page'), then MEASURE the black square with a
ruler and record the real size in your ground truth. Use the measured value.
"""
import argparse

import cv2
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--size-cm", type=float, default=15.0)
ap.add_argument("--id", type=int, default=7)
ap.add_argument("--out", default="marker.png")
a = ap.parse_args()

DPI = 300
px = int(round(a.size_cm / 2.54 * DPI))
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
m = cv2.aruco.generateImageMarker(d, a.id, px)
border = int(DPI * 0.5)  # ~1.27 cm white quiet zone
canvas = np.full((px + 2 * border, px + 2 * border), 255, np.uint8)
canvas[border:border + px, border:border + px] = m
Image.fromarray(canvas).save(a.out, dpi=(DPI, DPI))
print(f"wrote {a.out}: marker {a.size_cm} cm, dict 4x4_50, id {a.id}")
