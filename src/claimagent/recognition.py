"""Conservative local spine-candidate detection and optional OCR for saved frames.

Contour scores are heuristic candidate scores, not measured detection probabilities.
OCR never writes text into title/author fields; all text remains review evidence.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .checks import check_frame
from .pipeline import render_report, build_packet
from .schema import ReviewEntry


@dataclass(frozen=True)
class SpineCandidate:
    box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class OCRResult:
    text: str = ""
    confidence: float | None = None


class DetectorError(RuntimeError):
    """A requested detector could not load or return valid detections."""


def _available_memory_bytes() -> int:
    """Read available system RAM without importing the deep-learning runtime."""
    try:
        import psutil
    except ImportError as exc:
        raise DetectorError("YOLO RAM preflight requires psutil from the Ultralytics environment") from exc
    return int(psutil.virtual_memory().available)


def _python_values(value):
    """Convert a tensor-like Ultralytics value to ordinary Python lists."""
    if value is None:
        raise DetectorError("YOLO OBB output is missing a required field")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


class YoloOBBDetector:
    """Lazy CPU adapter for the project's local YOLO11 OBB checkpoint.

    The model is loaded only when this backend is explicitly selected. A RAM
    preflight runs before importing Ultralytics or deserializing checkpoint data.
    """

    MIN_AVAILABLE_RAM_GIB = 4.0

    def __init__(self, model_path: str | Path, *, conf: float = 0.25,
                 iou: float = 0.3, imgsz: int = 1280, device: str = "cpu") -> None:
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_file():
            raise DetectorError(f"YOLO OBB checkpoint does not exist: {self.model_path}")
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            raise DetectorError("YOLO confidence threshold must be between 0 and 1")
        if not math.isfinite(iou) or not 0 <= iou <= 1:
            raise DetectorError("YOLO IoU threshold must be between 0 and 1")
        if imgsz <= 0:
            raise DetectorError("YOLO image size must be a positive integer")
        if device != "cpu":
            raise DetectorError("This configured runtime supports CPU inference only")

        available = _available_memory_bytes()
        minimum = int(self.MIN_AVAILABLE_RAM_GIB * 1024**3)
        if available < minimum:
            raise DetectorError(
                f"YOLO loading stopped by RAM preflight: {available / 1024**3:.2f} GiB available; "
                f"at least {self.MIN_AVAILABLE_RAM_GIB:.1f} GiB is required"
            )

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path), task="obb")
        except Exception as exc:
            raise DetectorError(f"could not load YOLO OBB checkpoint {self.model_path}: {exc}") from exc
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def detect(self, image: np.ndarray) -> list[SpineCandidate]:
        """Run OBB prediction and adapt axis-aligned crop bounds for OCR."""
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        try:
            results = self.model.predict(
                source=image, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                device=self.device, verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"YOLO OBB prediction failed: {exc}") from exc
        if results is None:
            raise DetectorError("YOLO OBB prediction returned no result collection")
        try:
            first = results[0]
        except Exception as exc:
            raise DetectorError("YOLO OBB prediction returned no image result") from exc
        obb = getattr(first, "obb", None)
        if obb is None:
            raise DetectorError("checkpoint returned no OBB output; confirm it is an OBB model")

        try:
            polygons = _python_values(getattr(obb, "xyxyxyxy", None))
            scores = _python_values(getattr(obb, "conf", None))
            classes = _python_values(getattr(obb, "cls", None))
            if not (len(polygons) == len(scores) == len(classes)):
                raise DetectorError("YOLO OBB polygons, scores, and classes have inconsistent lengths")
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"YOLO returned malformed OBB outputs: {exc}") from exc

        candidates: list[SpineCandidate] = []
        for polygon, score, class_id in zip(polygons, scores, classes):
            try:
                points = [tuple(map(float, point)) for point in polygon]
                if len(points) != 4 or any(len(point) != 2 for point in points):
                    raise ValueError("each OBB must have four x/y corner points")
                confidence = float(score)
                class_value = float(class_id)
            except (TypeError, ValueError, OverflowError) as exc:
                raise DetectorError("YOLO OBB output contains non-numeric values") from exc
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence, class_value)):
                raise DetectorError("YOLO OBB output contains non-finite values")
            if not 0 <= confidence <= 1:
                raise DetectorError("YOLO OBB confidence is outside [0, 1]")
            if class_value != 0:
                continue
            left = max(0, min(width, math.floor(x1)))
            top = max(0, min(height, math.floor(y1)))
            right = max(0, min(width, math.ceil(x2)))
            bottom = max(0, min(height, math.ceil(y2)))
            if right <= left or bottom <= top:
                continue
            candidates.append(SpineCandidate((left, top, right - left, bottom - top), confidence))
        return sorted(candidates, key=lambda item: (item.box[0], item.box[1]))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _same_detection_box(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    """Match near-identical or contained contour boxes without merging neighbors."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    smaller_area = min(aw * ah, bw * bh)
    if smaller_area <= 0 or intersection / smaller_area < 0.82:
        return False
    center_x = abs((ax + aw / 2) - (bx + bw / 2))
    # A short interior contour can be vertically offset within a complete
    # spine; containment plus matching horizontal center is enough to reject it.
    return center_x <= min(aw, bw) * 0.22


def _suppress_contour_duplicates(candidates: list[SpineCandidate]) -> list[SpineCandidate]:
    """Collapse nested edge contours while retaining spatially distinct spines."""
    kept: list[SpineCandidate] = []
    # Prefer the larger complete region so a nested text/edge contour cannot
    # replace the full spine box just because its heuristic score is higher.
    ordered = sorted(candidates, key=lambda item: (item.box[2] * item.box[3], item.confidence), reverse=True)
    for candidate in ordered:
        if all(not _same_detection_box(candidate.box, other.box) for other in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda item: (item.box[0], item.box[1]))


def _edge_coverage(edges: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    """Measure support for all four sides of a proposed complete rectangle."""
    x, y, width, height = box
    image_h, image_w = edges.shape[:2]
    y1, y2 = max(0, y), min(image_h, y + height)
    x1, x2 = max(0, x), min(image_w, x + width)
    if x2 <= x1 or y2 <= y1:
        return (0.0, 0.0, 0.0, 0.0)
    vertical_pad = max(0, x - 2), min(image_w, x + 3)
    right_pad = max(0, x + width - 2), min(image_w, x + width + 3)
    top_pad = max(0, y - 2), min(image_h, y + 3)
    bottom_pad = max(0, y + height - 2), min(image_h, y + height + 3)
    left = float(np.mean(np.any(edges[y1:y2, vertical_pad[0]:vertical_pad[1]] > 0, axis=1)))
    right = float(np.mean(np.any(edges[y1:y2, right_pad[0]:right_pad[1]] > 0, axis=1)))
    top = float(np.mean(np.any(edges[top_pad[0]:top_pad[1], x1:x2] > 0, axis=0)))
    bottom = float(np.mean(np.any(edges[bottom_pad[0]:bottom_pad[1], x1:x2] > 0, axis=0)))
    return left, right, top, bottom


def _is_complete_spine_region(box: tuple[int, int, int, int], edges: np.ndarray) -> bool:
    """Reject small fragments and boxes without evidence on each spine edge."""
    x, y, width, height = box
    image_h, image_w = edges.shape[:2]
    minimum_height = max(64, int(round(image_h * 0.20)))
    if height < minimum_height or width < max(12, int(round(image_w * 0.012))):
        return False
    if width > image_w * 0.20 or width / max(height, 1) > 0.45 or height / max(width, 1) < 1.7:
        return False
    if x <= 1 or y <= 1 or x + width >= image_w - 1 or y + height >= image_h - 1:
        return False
    left, right, top, bottom = _edge_coverage(edges, box)
    # Require both long sides and visible top/bottom boundaries. This errs on
    # the side of missing partially occluded spines rather than boxing fragments.
    return left >= 0.58 and right >= 0.58 and top >= 0.42 and bottom >= 0.42


def detect_spines(image: np.ndarray) -> list[SpineCandidate]:
    """Return contour regions with support for a complete vertical spine outline."""
    if image is None or image.size == 0:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_h, image_w = gray.shape[:2]
    candidates: list[SpineCandidate] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        aspect = height / max(width, 1)
        if area < 700 or aspect < 1.7 or aspect > 10.0:
            continue
        if not _is_complete_spine_region((x, y, width, height), edges):
            continue
        contour_area = cv2.contourArea(contour)
        rectangularity = min(1.0, contour_area / max(area, 1))
        if rectangularity < 0.35:
            continue
        # A ranking heuristic only; it is not calibrated as a probability.
        confidence = round(min(0.85, 0.35 + 0.45 * rectangularity + min(aspect, 5.0) * 0.01), 3)
        candidates.append(SpineCandidate((x, y, width, height), confidence))

    return _suppress_contour_duplicates(candidates)


def _fingerprint(crop: np.ndarray) -> tuple[int, np.ndarray] | None:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if float(np.std(gray)) < 3.0:
        # Uniform crops have too little information for safe cross-frame matching.
        return None
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()[1:]
    median = float(np.median(dct))
    phash = sum((1 << index) for index, value in enumerate(dct) if value > median)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if crop.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return phash, hist


def _same_crop(first, second) -> bool:
    a, b = _fingerprint(first), _fingerprint(second)
    if a is None or b is None:
        return False
    hamming = (a[0] ^ b[0]).bit_count()
    color_distance = cv2.compareHist(a[1], b[1], cv2.HISTCMP_BHATTACHARYYA)
    return hamming <= 6 and color_distance <= 0.22


def _frame_features(image: np.ndarray):
    """Return compact ORB features used only to align nearby video frames."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 8:
        return keypoints, None
    return keypoints, descriptors


def _estimate_frame_affine(current_features, previous_features) -> np.ndarray | None:
    """Map current-frame coordinates into a nearby previous frame, conservatively."""
    current_points, current_descriptors = current_features
    previous_points, previous_descriptors = previous_features
    if current_descriptors is None or previous_descriptors is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        pairs = matcher.knnMatch(current_descriptors, previous_descriptors, k=2)
    except cv2.error:
        return None
    matches = [first for pair in pairs if len(pair) == 2
               for first, second in [pair] if first.distance < 0.72 * second.distance]
    if len(matches) < 10:
        return None
    source = np.float32([current_points[item.queryIdx].pt for item in matches])
    target = np.float32([previous_points[item.trainIdx].pt for item in matches])
    affine, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=1500, confidence=0.99, refineIters=10,
    )
    if affine is None or inliers is None or int(inliers.sum()) < 8:
        return None
    scale = float(np.linalg.norm(affine[:, :2], axis=0).mean())
    if not np.isfinite(affine).all() or not 0.65 <= scale <= 1.55:
        return None
    return affine


def _transform_box(box: tuple[int, int, int, int], affine: np.ndarray) -> tuple[float, float, float, float]:
    x, y, width, height = box
    corners = np.float32([[[x, y]], [[x + width, y]], [[x + width, y + height]], [[x, y + height]]])
    mapped = cv2.transform(corners, affine).reshape(-1, 2)
    left, top = mapped.min(axis=0)
    right, bottom = mapped.max(axis=0)
    return float(left), float(top), float(right - left), float(bottom - top)


def _box_iou(first, second) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _spatial_duplicate(current_box, previous_box, affine: np.ndarray | None,
                       image_size: tuple[int, int]) -> bool:
    """Require plausible size/shape plus strong overlap after frame alignment."""
    mapped = _transform_box(current_box, affine) if affine is not None else current_box
    _, _, cw, ch = mapped
    _, _, pw, ph = previous_box
    if min(cw, ch, pw, ph) <= 0:
        return False
    current_aspect, previous_aspect = ch / cw, ph / pw
    if abs(math.log(current_aspect / previous_aspect)) > 0.4:
        return False
    area_ratio = (cw * ch) / (pw * ph)
    if not 0.45 <= area_ratio <= 2.2:
        return False
    overlap = _box_iou(mapped, previous_box)
    intersection_ratio = overlap * (cw * ch + pw * ph) / min(cw * ch, pw * ph) if overlap else 0.0
    if intersection_ratio < 0.68:
        return False
    cx, cy = mapped[0] + cw / 2, mapped[1] + ch / 2
    px, py = previous_box[0] + pw / 2, previous_box[1] + ph / 2
    # Localized center agreement avoids joining adjacent, similarly sized books.
    # Requiring overlap also means visual similarity alone never suppresses a book.
    return (abs(cx - px) <= min(cw, pw) * 0.25
            and abs(cy - py) <= min(ch, ph) * 0.18)


def _candidate_warnings(box: tuple[int, int, int, int], image_size: tuple[int, int]) -> list[str]:
    """Flag low-specificity shapes for a separate manual-review group."""
    _, _, width, height = box
    image_height, image_width = image_size
    warnings = []
    if height / max(width, 1) < 1.5:
        warnings.append("Low-specificity shape; could be a shelf edge or another object")
    if width >= image_width * 0.4:
        warnings.append("Unusually wide candidate; inspect for a shelf or background region")
    if height >= image_height * 0.9:
        warnings.append("Candidate spans nearly the full frame height")
    return warnings



class TesseractOCR:
    """Optional local Tesseract CLI adapter; no network access or model download."""

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("tesseract")

    @property
    def available(self) -> bool:
        return bool(self.executable)

    def read(self, crop: np.ndarray) -> OCRResult:
        if not self.executable:
            return OCRResult()
        ok, encoded = cv2.imencode(".png", crop)
        if not ok:
            return OCRResult()
        try:
            result = subprocess.run(
                [self.executable, "stdin", "stdout", "tsv"],
                input=encoded.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False, timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return OCRResult()
        if result.returncode != 0:
            return OCRResult()
        rows = csv.DictReader(io.StringIO(result.stdout.decode("utf-8", errors="replace")), delimiter="\t")
        words: list[str] = []
        confidences: list[float] = []
        for row in rows:
            word = (row.get("text") or "").strip()
            try:
                confidence = float(row.get("conf", "-1"))
            except ValueError:
                continue
            if word and confidence >= 0:
                words.append(word)
                confidences.append(confidence / 100.0)
        return OCRResult(" ".join(words), round(sum(confidences) / len(confidences), 3) if confidences else None)


_CAPTURE_STAMP = re.compile(r"^capture_(\d{8}T\d{6}_\d{6}Z)(?:_\d+)?$", re.IGNORECASE)


def _frame_timestamp(path: Path) -> str:
    match = _CAPTURE_STAMP.match(path.stem)
    if not match:
        return ""
    try:
        value = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S_%fZ").replace(tzinfo=timezone.utc)
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    except ValueError:
        return ""


def _file_timestamp(path: Path) -> str:
    """Return filesystem mtime as traceability metadata, not asserted capture time."""
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def scan_frames(frames_dir: str | Path, ocr: TesseractOCR | None = None,
                detector: Callable[[np.ndarray], list[SpineCandidate]] | None = None):
    """Build a normal ClaimPacket from frames, queuing all machine-generated candidates."""
    root = Path(frames_dir)
    engine = ocr or TesseractOCR()
    image_paths = sorted((p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}),
                         key=lambda path: path.name.casefold()) if root.exists() else []
    observations: list[dict] = []
    extra_reviews: list[ReviewEntry] = []
    recent_frames: list[dict] = []
    sequence = 0

    for frame_index, path in enumerate(image_paths):
        frame_ref = path.name
        timestamp = _frame_timestamp(path)
        timestamp_source = "capture_filename" if timestamp else "file_modified_time"
        if not timestamp:
            try:
                timestamp = _file_timestamp(path)
            except OSError:
                timestamp = ""
        valid, reason = check_frame(frame_ref, root)
        if not valid:
            extra_reviews.append(ReviewEntry(ref_id=frame_ref, reason=f"frame skipped during book detection: {reason}"))
            recent_frames = []
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            extra_reviews.append(ReviewEntry(ref_id=frame_ref, reason="frame skipped during book detection: OpenCV could not decode image"))
            recent_frames = []
            continue
        candidates = (detector or detect_spines)(image)
        height, width = image.shape[:2]
        image_size = (height, width)
        features = _frame_features(image)
        current_tracks: list[dict] = []
        alignments = {
            frame["frame_ref"]: _estimate_frame_affine(features, frame["features"])
            for frame in recent_frames
        }
        for candidate in candidates:
            x, y, width, height = candidate.box
            crop = image[y:y + height, x:x + width]
            duplicate = None
            for prior_frame in reversed(recent_frames):
                affine = alignments[prior_frame["frame_ref"]]
                duplicate = next((track for track in prior_frame["tracks"]
                                  if _spatial_duplicate(candidate.box, track["box"], affine, image_size)), None)
                if duplicate is not None:
                    break
            if duplicate is not None:
                adjacent = prior_frame["frame_index"] == frame_index - 1
                reason = ("similar spine candidate in adjacent frame not counted again" if adjacent
                          else "possible duplicate candidate suppressed across sampled frames")
                extra_reviews.append(ReviewEntry(
                    ref_id=duplicate["book_id"], reason=reason,
                    details={"primary_frame_ref": duplicate["primary_frame_ref"],
                             "primary_frame_timestamp": duplicate["primary_frame_timestamp"],
                             "repeated_frame_ref": frame_ref, "repeated_frame_timestamp": timestamp or None,
                             "repeated_box": list(candidate.box)},
                ))
                current_tracks.append({**duplicate, "box": candidate.box})
                continue

            sequence += 1
            book_id = f"detected-book-{sequence:04d}"
            recognized = engine.read(crop)
            observation = {
                "id": book_id,
                "frame_ref": frame_ref,
                "frame_timestamp": timestamp,
                "timestamp_source": timestamp_source,
                "status": "unidentified",
                "detection_confidence": candidate.confidence,
                "detection_box": candidate.box,
                "detection_image_size": image_size,
                "candidate_warnings": _candidate_warnings(candidate.box, image_size),
                "ocr_confidence": recognized.confidence,
                "spine_text": recognized.text,
            }
            observations.append(observation)
            if not timestamp:
                extra_reviews.append(ReviewEntry(ref_id=book_id, reason="capture timestamp unavailable; frame filename retained",
                                                 details={"frame_ref": frame_ref}))
            elif timestamp_source == "file_modified_time":
                extra_reviews.append(ReviewEntry(ref_id=book_id, reason="timestamp is filesystem modification time, not verified capture time",
                                                 details={"frame_ref": frame_ref, "frame_timestamp": timestamp}))
            if not engine.available:
                extra_reviews.append(ReviewEntry(ref_id=book_id, reason="OCR engine unavailable; title and author remain unknown",
                                                 details={"frame_ref": frame_ref, "ocr_confidence": None}))
            elif recognized.text:
                extra_reviews.append(ReviewEntry(ref_id=book_id, reason="OCR spine text requires human review; title and author are not inferred",
                                                 details={"spine_text": recognized.text, "ocr_confidence": recognized.confidence,
                                                          "frame_ref": frame_ref}))
            else:
                extra_reviews.append(ReviewEntry(ref_id=book_id, reason="spine text was not confidently recognized; title and author remain unknown",
                                                 details={"frame_ref": frame_ref, "ocr_confidence": recognized.confidence}))
            current_tracks.append({
                "box": candidate.box, "book_id": book_id, "primary_frame_ref": frame_ref,
                "primary_frame_timestamp": timestamp or None,
            })
        recent_frames.append({"frame_index": frame_index, "frame_ref": frame_ref,
                              "features": features, "tracks": current_tracks})
        # The sample is small (normally <=10 frames), so compare against the
        # full sweep. This catches a spine missed in an intermediate frame.

    packet = build_packet({"candidates": observations, "items": []}, root)
    packet.review_queue.extend(extra_reviews)
    if not observations:
        packet.review_queue.append(ReviewEntry(ref_id="scan", reason="no spine candidates detected; inspect source frames manually"))
    return packet


def process_frames(frames_dir: str | Path = "data/frames", output_dir: str | Path = "output",
                   ocr: TesseractOCR | None = None,
                   detector: Callable[[np.ndarray], list[SpineCandidate]] | None = None) -> tuple[Path, Path]:
    packet = scan_frames(frames_dir, ocr, detector)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "claim_packet.json"
    report_path = output / "inventory_report.md"
    if json_path.exists() or report_path.exists():
        raise FileExistsError(f"Scan output already exists in {output}; choose a new --output-dir to preserve it")
    json_created = False
    report_created = False
    try:
        with json_path.open("x", encoding="utf-8", newline="\n") as stream:
            json_created = True
            stream.write(json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        with report_path.open("x", encoding="utf-8", newline="\n") as stream:
            report_created = True
            stream.write(render_report(packet))
    except Exception:
        if json_created:
            json_path.unlink(missing_ok=True)
        if report_created:
            report_path.unlink(missing_ok=True)
        raise
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect book-spine candidates in saved JPEG frames for human review.")
    parser.add_argument("--frames-dir", default="data/frames", help="Directory containing captured JPEG frames")
    parser.add_argument("--output-dir", default="output", help="Directory for the validated claim packet and report")
    parser.add_argument("--tesseract", help="Optional path to a locally installed Tesseract executable")
    parser.add_argument("--detector", choices=("contour", "yolo"), default="contour",
                        help="Candidate detector (YOLO loads the configured local checkpoint)")
    default_model = Path(__file__).resolve().parents[2] / ".training_envs" / "bookspine-yolo-cpu-py311" / "weights" / "yolo11n-obb.pt"
    parser.add_argument("--model", default=str(default_model), help="Local YOLO OBB checkpoint path")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.3, help="YOLO NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--device", choices=("cpu",), default="cpu", help="YOLO inference device")
    args = parser.parse_args()
    engine = TesseractOCR(args.tesseract) if args.tesseract else None
    detector = None
    if args.detector == "yolo":
        try:
            detector = YoloOBBDetector(args.model, conf=args.conf, iou=args.iou,
                                       imgsz=args.imgsz, device=args.device).detect
        except DetectorError as exc:
            parser.error(str(exc))
    try:
        json_path, report_path = process_frames(args.frames_dir, args.output_dir, engine, detector)
    except DetectorError as exc:
        parser.error(str(exc))
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
