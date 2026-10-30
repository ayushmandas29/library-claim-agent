"""Local webcam preview and timestamped frame capture.

This module only saves raw evidence frames. It does not create observations or
send data to the claim packet pipeline.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2


class CaptureError(RuntimeError):
    """A camera could not be opened or stopped producing valid frames."""


class FrameStore:
    """Write encoded camera frames without replacing existing evidence."""

    def __init__(self, directory: str | Path,
                 now: Callable[[], datetime] | None = None) -> None:
        self.directory = Path(directory)
        self._now = now or (lambda: datetime.now(timezone.utc))

    def save(self, frame) -> Path:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise CaptureError("Camera frame could not be encoded as JPEG")
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = self._now().astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        stem = f"capture_{stamp}"
        payload = encoded.tobytes()
        for suffix in range(10000):
            name = f"{stem}.jpg" if suffix == 0 else f"{stem}_{suffix:03d}.jpg"
            path = self.directory / name
            try:
                with path.open("xb") as output:
                    output.write(payload)
                return path
            except FileExistsError:
                continue
        raise CaptureError("Could not choose a unique filename for the camera frame")


class CaptureSession:
    """Small testable camera lifecycle used by the desktop preview."""

    def __init__(self, store: FrameStore, camera_factory: Callable[[int], object] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 interval_s: float = 1.0) -> None:
        self.store = store
        self.camera_factory = camera_factory or cv2.VideoCapture
        self.clock = clock
        self.interval_s = interval_s
        self.camera = None
        self.started_at: float | None = None
        self.last_capture_at: float | None = None

    @property
    def running(self) -> bool:
        return self.camera is not None

    def start(self, camera_index: int = 0) -> None:
        if self.running:
            return
        camera = None
        try:
            camera = self.camera_factory(camera_index)
            if not camera.isOpened():
                camera.release()
                camera = None
                raise CaptureError(
                    "Camera is unavailable. Check that it is connected and that camera permission is enabled."
                )
        except CaptureError:
            raise
        except Exception as exc:
            if camera is not None:
                try:
                    camera.release()
                except Exception:
                    pass
            raise CaptureError(f"Could not start the camera: {exc}") from exc
        self.camera = camera
        self.started_at = self.clock()
        self.last_capture_at = None

    def read(self) -> tuple[object, Path | None]:
        if not self.running:
            raise CaptureError("Camera capture is not running")
        try:
            ok, frame = self.camera.read()
        except Exception as exc:
            self.stop()
            raise CaptureError(f"Camera capture failed: {exc}") from exc
        if not ok or frame is None:
            self.stop()
            raise CaptureError("Camera stopped returning frames. Check camera permission or connection.")
        now = self.clock()
        saved = None
        if self.last_capture_at is None or now - self.last_capture_at >= self.interval_s:
            try:
                saved = self.store.save(frame)
            except Exception as exc:
                self.stop()
                if isinstance(exc, CaptureError):
                    raise
                raise CaptureError(f"Could not save camera evidence: {exc}") from exc
            self.last_capture_at = now
        return frame, saved

    def elapsed_s(self) -> float:
        return max(0.0, self.clock() - self.started_at) if self.started_at is not None else 0.0

    def stop(self) -> None:
        camera, self.camera = self.camera, None
        self.started_at = None
        self.last_capture_at = None
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass


def format_elapsed(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class WindowsVoiceGuide:
    """Use Windows' built-in SAPI speech when available; visual guidance is primary."""

    def __init__(self) -> None:
        self.process = None

    def speak(self, message: str) -> bool:
        if sys.platform != "win32":
            return False
        escaped = message.replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speaker.Speak('{escaped}')"
        )
        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.process = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=flags,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.process = None


def run_app(frames_dir: str | Path = "data/frames", camera_index: int = 0) -> None:
    """Open the Tkinter webcam preview. Tkinter is included with standard Windows Python."""
    try:
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
    except ImportError as exc:
        raise RuntimeError("The capture window requires Tkinter and Pillow. Install Python with Tcl/Tk support.") from exc

    session = CaptureSession(FrameStore(frames_dir))
    voice = WindowsVoiceGuide()
    root = tk.Tk()
    root.title("Library Claim Agent — Camera Capture")
    root.minsize(720, 520)
    status = tk.StringVar(value="Ready. Start capture to preview and save evidence frames.")
    elapsed = tk.StringVar(value="Elapsed sweep time: 00:00:00")
    guidance = tk.StringVar(value="Guidance: slowly sweep across shelves and visible contents.")

    preview = ttk.Label(root, text="Camera preview will appear here", anchor="center")
    preview.pack(fill="both", expand=True, padx=12, pady=(12, 6))
    ttk.Label(root, textvariable=status, wraplength=680).pack(fill="x", padx=12, pady=4)
    ttk.Label(root, textvariable=elapsed).pack(anchor="w", padx=12, pady=2)
    ttk.Label(root, textvariable=guidance, wraplength=680).pack(fill="x", padx=12, pady=4)
    controls = ttk.Frame(root)
    controls.pack(fill="x", padx=12, pady=(4, 12))
    start_button = ttk.Button(controls, text="Start capture")
    stop_button = ttk.Button(controls, text="Stop", state="disabled")
    start_button.pack(side="left")
    stop_button.pack(side="left", padx=8)
    image_ref = {"value": None}
    timer_ref = {"value": None}

    def stop_capture(message: str = "Capture stopped.") -> None:
        if timer_ref["value"] is not None:
            try:
                root.after_cancel(timer_ref["value"])
            except tk.TclError:
                pass
            timer_ref["value"] = None
        session.stop()
        voice.stop()
        status.set(message)
        start_button.configure(state="normal")
        stop_button.configure(state="disabled")

    def tick() -> None:
        try:
            frame, saved = session.read()
            height, width = frame.shape[:2]
            scale = min(680 / max(width, 1), 360 / max(height, 1), 1.0)
            preview_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            image_ref["value"] = photo
            preview.configure(image=photo, text="")
            elapsed.set(f"Elapsed sweep time: {format_elapsed(session.elapsed_s())}")
            if saved is not None:
                status.set(f"Capturing. Saved evidence frame: {saved.name}")
            timer_ref["value"] = root.after(50, tick)
        except Exception as exc:
            message = str(exc) if isinstance(exc, CaptureError) else f"Camera preview failed: {exc}"
            stop_capture(message)

    def start_capture() -> None:
        try:
            session.start(camera_index)
        except CaptureError as exc:
            status.set(str(exc))
            return
        message = "Capture started. Slowly sweep across shelves and visible contents."
        guidance.set(f"Guidance: {message}")
        status.set("Camera active; saving one timestamped frame per second.")
        start_button.configure(state="disabled")
        stop_button.configure(state="normal")
        voice.speak(message)
        tick()

    start_button.configure(command=start_capture)
    stop_button.configure(command=lambda: stop_capture())

    def close_window() -> None:
        stop_capture("Capture cancelled.")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.mainloop()


def run_capture_and_scan(frames_dir: str | Path, camera_index: int,
                         scan_output_dir: str | Path | None = None) -> tuple[Path, Path] | None:
    """Run capture, then optionally scan its frames for review candidates."""
    run_app(frames_dir, camera_index)
    if scan_output_dir is None:
        return None

    try:
        # Import only after capture and use process_frames' default contour detector.
        from .recognition import process_frames

        return process_frames(frames_dir, scan_output_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Capture finished and saved frames remain in {frames_dir}, but scanning failed: {exc}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a local webcam and save timestamped evidence frames.")
    parser.add_argument("--frames-dir", default="data/frames", help="Directory where evidence frames are saved")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--scan-after-capture", action="store_true",
                        help="After the capture window closes, scan frames with the default contour detector")
    parser.add_argument("--scan-output-dir", default="output/capture-scan",
                        help="Separate output directory for the optional candidate packet and report")
    args = parser.parse_args()
    try:
        outputs = run_capture_and_scan(
            args.frames_dir, args.camera,
            args.scan_output_dir if args.scan_after_capture else None,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    if outputs is not None:
        print("Scan results are unreviewed candidates; verify them against the original frames.")
        print(f"Candidate packet: {outputs[0]}")
        print(f"Markdown report: {outputs[1]}")


if __name__ == "__main__":
    main()
