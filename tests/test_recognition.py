import json
import sys

from types import SimpleNamespace

import cv2
import numpy as np

import pytest

from claimagent.recognition import (
    DetectorError, OCRResult, SpineCandidate, YoloOBBDetector,
    detect_spines, process_frames, scan_frames,
)
from claimagent import recognition
from claimagent.schema import ClaimPacket


def shelf_scene() -> np.ndarray:
    image = np.full((240, 360, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (35, 40), (75, 210), (20, 60, 180), -1)
    cv2.line(image, (40, 50), (40, 200), (255, 255, 255), 2)
    cv2.rectangle(image, (90, 45), (135, 208), (120, 30, 20), -1)
    cv2.line(image, (130, 55), (130, 198), (255, 255, 255), 2)
    return image


class FakeOCR:
    available = True

    def read(self, crop):
        return OCRResult("SYNTHETIC SPINE TEXT", 0.72)


def write_frame(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_contour_detector_finds_candidate_regions_in_synthetic_scene():
    found = detect_spines(shelf_scene())
    assert len(found) == 2
    assert all(0 <= item.confidence <= 1 for item in found)
    assert found[0].box[0] < found[1].box[0]
    # The boxes should cover the whole two source spines, not just their text or edges.
    assert found[0].box[1] <= 40 and found[0].box[1] + found[0].box[3] >= 210
    assert found[1].box[1] <= 45 and found[1].box[1] + found[1].box[3] >= 208


def test_partial_spine_contours_without_full_region_support_are_rejected():
    edges = np.zeros((240, 360), dtype=np.uint8)
    cv2.rectangle(edges, (100, 20), (140, 220), 255, 2)
    assert recognition._is_complete_spine_region((100, 20, 41, 201), edges)

    edges.fill(0)
    cv2.rectangle(edges, (50, 80), (78, 135), 255, 2)
    assert not recognition._is_complete_spine_region((50, 80, 29, 56), edges)

    # A tall box inferred around a text fragment is still invalid when it has
    # no sustained evidence for both long spine edges and both end boundaries.
    edges.fill(0)
    cv2.line(edges, (50, 50), (50, 200), 255, 2)
    cv2.line(edges, (50, 50), (85, 50), 255, 2)
    cv2.line(edges, (50, 200), (85, 200), 255, 2)
    assert not recognition._is_complete_spine_region((50, 50, 36, 151), edges)

    # A single wide box spanning two adjacent spine widths is not a spine.
    edges.fill(0)
    cv2.rectangle(edges, (40, 30), (120, 200), 255, 2)
    assert not recognition._is_complete_spine_region((40, 30, 81, 171), edges)


def test_nested_contour_suppression_keeps_adjacent_books_separate():
    nested = [
        SpineCandidate((20, 20, 60, 170), 0.70),
        SpineCandidate((22, 75, 56, 90), 0.75),
    ]
    neighboring = [
        SpineCandidate((20, 20, 40, 170), 0.70),
        SpineCandidate((59, 20, 40, 170), 0.71),
    ]
    nested_kept = recognition._suppress_contour_duplicates(nested)
    assert len(nested_kept) == 1
    assert nested_kept[0].box == nested[0].box  # keep the complete outer spine region
    assert len(recognition._suppress_contour_duplicates(neighboring)) == 2


def test_cross_frame_matching_requires_spatial_overlap_and_tracks_past_two_frames(tmp_path):
    frames = tmp_path / "frames"
    for name in ("capture_20261029T123456_123456Z.jpg", "capture_20261029T123457_123456Z.jpg",
                 "capture_20261029T123458_123456Z.jpg"):
        write_frame(frames / name, shelf_scene())
    calls = [0]

    def detector(_image):
        calls[0] += 1
        if calls[0] == 1:
            return [SpineCandidate((35, 40, 40, 170), 0.72)]
        if calls[0] == 2:
            return []
        return [SpineCandidate((35, 40, 40, 170), 0.72),
                SpineCandidate((250, 40, 40, 170), 0.72)]

    packet = scan_frames(frames, ocr=FakeOCR(), detector=detector)
    assert len(packet.candidates) == 2
    repeats = [entry for entry in packet.review_queue if "suppressed across sampled frames" in entry.reason]
    assert len(repeats) == 1
    assert repeats[0].details["repeated_frame_ref"] == "capture_20261029T123458_123456Z.jpg"


def test_scan_uses_existing_book_schema_and_keeps_identity_unknown(tmp_path):
    frames = tmp_path / "frames"
    image_path = frames / "capture_20261029T123456_123456Z.jpg"
    write_frame(image_path, shelf_scene())

    packet = scan_frames(frames, ocr=FakeOCR())

    assert isinstance(packet, ClaimPacket)
    assert len(packet.candidates) == 2
    first = packet.candidates[0]
    assert first.frame_ref == image_path.name
    assert first.frame_timestamp == "2026-10-29T12:34:56.123456Z"
    assert first.timestamp_source == "capture_filename"
    assert first.detection_confidence is not None
    assert first.ocr_confidence == 0.72
    assert first.spine_text == "SYNTHETIC SPINE TEXT"
    assert packet.books == []
    assert any("title and author are not inferred" in item.reason for item in packet.review_queue)
    assert packet.totals.book_count == 0
    assert len(packet.candidates) == 2


def test_scan_deduplicates_matching_spines_across_adjacent_frames_and_keeps_both_refs(tmp_path):
    frames = tmp_path / "frames"
    write_frame(frames / "capture_20261029T123456_123456Z.jpg", shelf_scene())
    write_frame(frames / "capture_20261029T123457_123456Z.jpg", shelf_scene())

    packet = scan_frames(frames, ocr=FakeOCR())

    assert packet.totals.book_count == 0
    assert len(packet.candidates) == 2
    repeats = [entry for entry in packet.review_queue if "adjacent frame" in entry.reason]
    assert len(repeats) == 2
    assert all(entry.details["primary_frame_ref"].endswith("123456Z.jpg") for entry in repeats)
    assert all(entry.details["repeated_frame_ref"].endswith("123457_123456Z.jpg") for entry in repeats)


def test_scan_matches_camera_shifted_box_and_preserves_alternate_frame_evidence(tmp_path):
    frames = tmp_path / "frames"
    first = shelf_scene()
    transform = np.float32([[1, 0, 12], [0, 1, 5]])
    second = cv2.warpAffine(first, transform, (first.shape[1], first.shape[0]))
    first_path = frames / "capture_20261029T123456_123456Z.jpg"
    second_path = frames / "capture_20261029T123457_123456Z.jpg"
    write_frame(first_path, first)
    write_frame(second_path, second)
    calls = [0]

    def detector(_image):
        calls[0] += 1
        if calls[0] == 1:
            return [SpineCandidate((35, 40, 40, 170), 0.72)]
        return [SpineCandidate((47, 45, 40, 170), 0.71),
                SpineCandidate((220, 45, 40, 170), 0.7)]

    packet = scan_frames(frames, ocr=FakeOCR(), detector=detector)

    assert len(packet.candidates) == 2
    assert packet.candidates[0].frame_ref == first_path.name
    assert packet.candidates[0].frame_timestamp == "2026-10-29T12:34:56.123456Z"
    assert packet.candidates[0].detection_box == (35, 40, 40, 170)
    suppressed = [entry for entry in packet.review_queue if "not counted again" in entry.reason]
    assert len(suppressed) == 1
    assert suppressed[0].details["primary_frame_ref"] == first_path.name
    assert suppressed[0].details["repeated_frame_ref"] == second_path.name
    assert suppressed[0].details["repeated_frame_timestamp"] == "2026-10-29T12:34:57.123456Z"


def test_low_specificity_geometry_is_flagged_but_kept_for_review(tmp_path):
    frames = tmp_path / "frames"
    image_path = frames / "capture_20261029T123456_123456Z.jpg"
    write_frame(image_path, shelf_scene())

    packet = scan_frames(
        frames, ocr=FakeOCR(),
        detector=lambda _image: [SpineCandidate((15, 15, 160, 215), 0.75)],
    )

    assert len(packet.candidates) == 1
    assert packet.candidates[0].frame_ref == image_path.name
    assert "Unusually wide candidate; inspect for a shelf or background region" in packet.candidates[0].candidate_warnings


def test_missing_ocr_keeps_spine_text_unknown_and_scan_outputs_validate(tmp_path):
    frames = tmp_path / "frames"
    write_frame(frames / "ordinary-name.jpg", shelf_scene())
    packet = scan_frames(frames)
    assert packet.candidates
    assert all(not book.spine_text and book.ocr_confidence is None for book in packet.candidates)
    assert packet.books == []
    assert all(book.timestamp_source == "file_modified_time" for book in packet.candidates)
    assert any("OCR engine unavailable" in entry.reason for entry in packet.review_queue)
    assert any("not verified capture time" in entry.reason for entry in packet.review_queue)

    json_path, report_path = process_frames(frames, tmp_path / "output")
    restored = ClaimPacket.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert restored == scan_frames(frames)
    report = report_path.read_text(encoding="utf-8")
    assert "OCR engine unavailable" in report
    assert "## Unreviewed detected candidates" in report
    assert "Unreviewed detected candidates: 2" in report

    original = json_path.read_bytes()
    try:
        process_frames(frames, tmp_path / "output")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing scan output should not be overwritten")
    assert json_path.read_bytes() == original


def test_optional_tesseract_tsv_text_and_confidence_are_parsed(monkeypatch):
    tsv = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
           "5\t1\t1\t1\t1\t1\t0\t0\t20\t10\t80.0\tTITLE\n"
           "5\t1\t1\t1\t1\t2\t22\t0\t20\t10\t60.0\tAUTHOR\n")
    monkeypatch.setattr(recognition.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=tsv.encode(), stderr=b""))
    result = recognition.TesseractOCR("local-tesseract").read(np.zeros((10, 40, 3), dtype=np.uint8))
    assert result.text == "TITLE AUTHOR"
    assert result.confidence == 0.7


def test_injected_obb_detector_keeps_ocr_and_packet_review_behavior(tmp_path):
    frames = tmp_path / "frames"
    image_path = frames / "capture_20261029T123456_123456Z.jpg"
    write_frame(image_path, shelf_scene())
    called = []

    def detector(image):
        called.append(image.shape)
        return [SpineCandidate((30, 35, 48, 176), 0.91)]

    packet = scan_frames(frames, ocr=FakeOCR(), detector=detector)

    assert called == [(240, 360, 3)]
    assert len(packet.candidates) == 1
    book = packet.candidates[0]
    assert book.frame_ref == image_path.name
    assert book.detection_confidence == 0.91
    assert book.ocr_confidence == 0.72
    assert book.spine_text == "SYNTHETIC SPINE TEXT"
    assert packet.books == []
    assert any("human review" in entry.reason for entry in packet.review_queue)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class FakeOBB:
    xyxyxyxy = FakeTensor([
        [[-2.2, 1.1], [20.1, 1.1], [20.1, 50.8], [-2.2, 50.8]],
        [[40, 5], [100, 5], [100, 60], [40, 60]],
    ])
    conf = FakeTensor([0.83, 0.55])
    cls = FakeTensor([0, 1])


class FakeResult:
    obb = FakeOBB()


def allow_fake_yolo_load(monkeypatch, yolo_factory):
    monkeypatch.setattr(recognition, "_available_memory_bytes", lambda: 4 * 1024**3)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=yolo_factory))


def test_yolo_obb_adapter_loads_local_checkpoint_with_settings_and_clips_boxes(tmp_path, monkeypatch):
    checkpoint = tmp_path / "existing-local.pt"
    checkpoint.touch()
    calls = {}

    class FakeYOLO:
        def __init__(self, model_path, task):
            calls["load"] = (model_path, task)

        def predict(self, **kwargs):
            calls["predict"] = kwargs
            return [FakeResult()]

    allow_fake_yolo_load(monkeypatch, FakeYOLO)
    detector = YoloOBBDetector(checkpoint, conf=0.25, iou=0.3, imgsz=1280, device="cpu")
    found = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert calls["load"] == (str(checkpoint), "obb")
    assert calls["predict"]["conf"] == 0.25
    assert calls["predict"]["iou"] == 0.3
    assert calls["predict"]["imgsz"] == 1280
    assert calls["predict"]["device"] == "cpu"
    assert found == [SpineCandidate((0, 1, 21, 50), 0.83)]


def test_rotated_obb_is_converted_to_enclosing_crop_preserving_ocr_evidence(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    image_path = frames / "capture_20261029T123456_123456Z.jpg"
    write_frame(image_path, np.zeros((100, 100, 3), dtype=np.uint8))
    checkpoint = tmp_path / "existing-local.pt"
    checkpoint.touch()
    corners = [[40, 20], [60, 25], [50, 80], [30, 75]]

    class RotatedOBB:
        xyxyxyxy = FakeTensor([corners])
        conf = FakeTensor([0.88])
        cls = FakeTensor([0])

    class RotatedResult:
        obb = RotatedOBB()

    class FakeYOLO:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, **kwargs):
            return [RotatedResult()]

    class CapturingOCR(FakeOCR):
        crop_shape = None

        def read(self, crop):
            self.crop_shape = crop.shape
            return super().read(crop)

    allow_fake_yolo_load(monkeypatch, FakeYOLO)
    detector = YoloOBBDetector(checkpoint)
    ocr = CapturingOCR()
    packet = scan_frames(frames, ocr=ocr, detector=detector.detect)

    # The four rotated corners are enclosed by x=[30,60], y=[20,80].
    assert ocr.crop_shape == (60, 30, 3)
    assert len(packet.candidates) == 1
    assert packet.candidates[0].detection_confidence == 0.88
    assert packet.candidates[0].spine_text == "SYNTHETIC SPINE TEXT"
    assert packet.candidates[0].ocr_confidence == 0.72
    assert packet.books == []


def test_yolo_obb_adapter_rejects_missing_checkpoint_before_ram_check(tmp_path, monkeypatch):
    monkeypatch.setattr(
        recognition, "_available_memory_bytes",
        lambda: pytest.fail("RAM must not be checked when the checkpoint is missing"),
    )
    with pytest.raises(DetectorError, match="checkpoint does not exist"):
        YoloOBBDetector(tmp_path / "missing.pt")


def test_yolo_obb_adapter_stops_before_loading_when_ram_is_low(tmp_path, monkeypatch):
    checkpoint = tmp_path / "existing-local.pt"
    checkpoint.touch()
    loaded = []

    class FakeYOLO:
        def __init__(self, *args, **kwargs):
            loaded.append(True)

    monkeypatch.setattr(recognition, "_available_memory_bytes", lambda: 512 * 1024**2)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    with pytest.raises(DetectorError, match="RAM preflight"):
        YoloOBBDetector(checkpoint)
    assert loaded == []


def test_yolo_obb_adapter_reports_prediction_and_wrong_task_failures(tmp_path, monkeypatch):
    checkpoint = tmp_path / "existing-local.pt"
    checkpoint.touch()

    class FailingYOLO:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, **kwargs):
            raise RuntimeError("synthetic inference failure")

    allow_fake_yolo_load(monkeypatch, FailingYOLO)
    failing = YoloOBBDetector(checkpoint)
    with pytest.raises(DetectorError, match="prediction failed: synthetic inference failure"):
        failing.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    class NonOBBResult:
        obb = None

    class NonOBBModel:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, **kwargs):
            return [NonOBBResult()]

    allow_fake_yolo_load(monkeypatch, NonOBBModel)
    non_obb = YoloOBBDetector(checkpoint)
    with pytest.raises(DetectorError, match="no OBB output"):
        non_obb.detect(np.zeros((100, 100, 3), dtype=np.uint8))


def test_malformed_yolo_output_fails_without_partial_claim_packet(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    write_frame(frames / "shelf.jpg", shelf_scene())
    output = tmp_path / "invalid-output"
    checkpoint = tmp_path / "existing-local.pt"
    checkpoint.touch()

    class InvalidOBB:
        xyxyxyxy = FakeTensor(object())
        conf = FakeTensor([0.9])
        cls = FakeTensor([0])

    class InvalidResult:
        obb = InvalidOBB()

    class InvalidYOLO:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, **kwargs):
            return [InvalidResult()]

    allow_fake_yolo_load(monkeypatch, InvalidYOLO)
    detector = YoloOBBDetector(checkpoint)
    with pytest.raises(DetectorError, match="malformed OBB outputs"):
        process_frames(frames, output, ocr=FakeOCR(), detector=detector.detect)
    assert not output.exists()


def test_detector_failure_does_not_write_partial_scan_outputs(tmp_path):
    frames = tmp_path / "frames"
    write_frame(frames / "shelf.jpg", shelf_scene())
    output = tmp_path / "failed-output"

    def failing_detector(_image):
        raise DetectorError("detector unavailable")

    with pytest.raises(DetectorError, match="detector unavailable"):
        process_frames(frames, output, ocr=FakeOCR(), detector=failing_detector)
    assert not output.exists()
