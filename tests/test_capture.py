from datetime import datetime, timezone

import numpy as np
import pytest
from PIL import Image

from claimagent.capture import CaptureError, CaptureSession, FrameStore, format_elapsed, run_capture_and_scan


FIXED_TIME = datetime(2026, 10, 29, 12, 34, 56, 123456, tzinfo=timezone.utc)


def test_frame_store_writes_readable_timestamped_images_without_overwriting(tmp_path):
    store = FrameStore(tmp_path, now=lambda: FIXED_TIME)
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    first = store.save(frame)
    second = store.save(frame)

    assert first.name == "capture_20261029T123456_123456Z.jpg"
    assert second.name == "capture_20261029T123456_123456Z_001.jpg"
    with Image.open(first) as saved:
        saved.verify()
    with Image.open(second) as saved:
        saved.verify()


class FakeCamera:
    def __init__(self, frames=None, opened=True):
        self.frames = list(frames or [(True, np.zeros((4, 5, 3), dtype=np.uint8))])
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return self.frames.pop(0) if self.frames else (False, None)

    def release(self):
        self.released = True


def test_capture_session_saves_at_interval_and_stops_cleanly(tmp_path):
    now = [10.0]
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    camera = FakeCamera(frames=[(True, frame), (True, frame), (True, frame)])
    session = CaptureSession(FrameStore(tmp_path), camera_factory=lambda _: camera,
                             clock=lambda: now[0], interval_s=1.0)
    session.start()
    frame, first = session.read()
    assert frame.shape == (4, 5, 3)
    assert first is not None and first.exists()

    now[0] += 0.5
    _, skipped = session.read()
    assert skipped is None
    now[0] += 0.5
    _, next_frame = session.read()
    assert next_frame is not None and next_frame.exists()
    assert session.elapsed_s() == 1.0

    session.stop()
    assert camera.released
    assert not session.running


def test_unavailable_camera_and_read_failure_are_reported_and_released(tmp_path):
    unavailable = FakeCamera(opened=False)
    session = CaptureSession(FrameStore(tmp_path), camera_factory=lambda _: unavailable)
    with pytest.raises(CaptureError, match="unavailable"):
        session.start()
    assert unavailable.released

    failed = FakeCamera(frames=[(False, None)])
    session = CaptureSession(FrameStore(tmp_path), camera_factory=lambda _: failed)
    session.start()
    with pytest.raises(CaptureError, match="stopped returning frames"):
        session.read()
    assert failed.released
    assert not session.running


def test_elapsed_time_is_formatted_as_hours_minutes_seconds():
    assert format_elapsed(0) == "00:00:00"
    assert format_elapsed(3661.9) == "01:01:01"
    assert format_elapsed(-1) == "00:00:00"


def test_capture_only_path_does_not_scan(monkeypatch, tmp_path):
    import claimagent.capture as capture

    calls = []
    monkeypatch.setattr(capture, "run_app", lambda frames, camera: calls.append((frames, camera)))
    result = run_capture_and_scan(tmp_path / "frames", 2)
    assert result is None
    assert calls == [(tmp_path / "frames", 2)]


def test_opt_in_scans_after_capture_with_default_detector(monkeypatch, tmp_path):
    import claimagent.capture as capture
    import claimagent.recognition as recognition

    calls = []
    monkeypatch.setattr(capture, "run_app", lambda frames, camera: calls.append("capture"))
    expected = (tmp_path / "scan" / "claim_packet.json", tmp_path / "scan" / "inventory_report.md")

    def fake_process_frames(frames, output):
        calls.append((frames, output))
        return expected

    monkeypatch.setattr(recognition, "process_frames", fake_process_frames)
    monkeypatch.setattr(recognition, "YoloOBBDetector", lambda *args, **kwargs: pytest.fail("YOLO must not load"))

    result = run_capture_and_scan(tmp_path / "frames", 0, tmp_path / "scan")
    assert result == expected
    assert calls == ["capture", (tmp_path / "frames", tmp_path / "scan")]


def test_scan_failure_reports_saved_frame_location_and_preserves_frames(monkeypatch, tmp_path):
    import claimagent.capture as capture
    import claimagent.recognition as recognition

    frames = tmp_path / "frames"
    frames.mkdir()
    saved_frame = frames / "capture.jpg"
    saved_frame.write_bytes(b"captured evidence")
    monkeypatch.setattr(capture, "run_app", lambda *_: None)

    def fail_scan(*args, **kwargs):
        raise FileExistsError("scan output already exists")

    monkeypatch.setattr(recognition, "process_frames", fail_scan)
    with pytest.raises(RuntimeError, match="saved frames remain.*scanning failed"):
        run_capture_and_scan(frames, 0, tmp_path / "scan")
    assert saved_frame.read_bytes() == b"captured evidence"


def test_process_frames_removes_partial_files_after_write_error(monkeypatch, tmp_path):
    import claimagent.recognition as recognition
    from claimagent.schema import ClaimPacket

    frames = tmp_path / "frames"
    frames.mkdir()
    output = tmp_path / "scan"
    packet = ClaimPacket()
    monkeypatch.setattr(recognition, "scan_frames", lambda *args, **kwargs: packet)
    monkeypatch.setattr(recognition, "render_report", lambda *_: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        recognition.process_frames(frames, output)
    assert not (output / "claim_packet.json").exists()
    assert not (output / "inventory_report.md").exists()
