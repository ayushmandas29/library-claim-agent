# Library Contents Claim Agent

The project accepts reviewed observations as JSON, checks saved frame evidence, computes deterministic totals, and writes a claim packet plus a readable Markdown report. A separate local camera utility saves timestamped frames. A review-only scanner can flag possible book spines and optionally transcribe spine text when local Tesseract is already installed. It does not confirm book identities, measure from images, or look up prices.

## Requirement status and known gaps

| Requirement | Current implementation | Not provided or not verified |
|---|---|---|
| Live voice-and-camera sweep | Tkinter camera preview saves timestamped frames at about one frame per second. Windows SAPI reads one start prompt aloud. | No microphone input or voice control; capture does not automatically analyze frames or create a claim packet. |
| Book detection, identification, measurement | `claimagent.recognition` uses OpenCV contours by default and can explicitly use the local YOLO11 OBB checkpoint with `--detector yolo`. Optional Tesseract returns raw OCR text for review. | Real checkpoint loading/inference has not been verified here. It does not identify titles/authors or measure spine dimensions. OCR text is not an identification. |
| Book valuation | Packet fields validate user-entered price evidence, currency, retrieval date, used condition, and conversion arithmetic. | No local catalog or price-source lookup is implemented. Prices must be researched and entered by a reviewer. |
| Non-book detection and valuation | The packet accepts reviewed non-book item observations and traceable replacement price ranges. | There is no non-book detector or automatic valuation. |
| Room area | The packet derives rectangular floor and wall area from reviewer-provided room dimensions when permitted by the shape note. | It does not measure dimensions or areas from camera frames; shelved wall area is never inferred. |
| Evidence-backed claim packet | Frame references are checked; unsupported measurements/prices are queued or excluded; totals and Markdown/JSON packets are deterministic. | Automated detections and OCR remain unreviewed candidates until a person verifies them. |

The contour detector remains the default so existing scans keep their prior behavior. The opt-in YOLO backend uses `conf=0.25`, `iou=0.3`, and `imgsz=1280` by default and loads only the configured local checkpoint. It refuses to load when available RAM is below 4 GiB; this preflight is a guard, not a guarantee that inference will fit. The saved 30-image contour baseline is not evidence for YOLO performance: it reports 3,520 false positives among 3,598 candidates (78 true positives, 395 false negatives), using axis-aligned matching against enclosures of OBB annotations. The proposed eight-image validation has no result artifact in this repository, so YOLO accuracy remains unverified. Do not describe the contour baseline as trained-model accuracy.

## Setup and tests

Python 3.10+ and the dependencies in `requirements.txt` are sufficient for this offline pipeline. No API key or network access is needed.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m pytest -q
```

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
```

## Capture webcam evidence (Windows)

The capture utility uses OpenCV and Pillow, which are already listed in `requirements.txt`. It uses Python's Tkinter window toolkit and attempts a short spoken cue with Windows' built-in speech synthesis. Guidance remains visible in the window if speech synthesis is unavailable. No microphone is used.

From the project root in PowerShell, activate the existing project environment and start the utility:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m claimagent.capture
```

If setting up a fresh environment, install the project's declared dependencies first:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m claimagent.capture
```

The window previews the default camera (device 0). Select **Start capture** to begin and **Stop** to end the sweep. The status shows elapsed time and each saved frame. Frames are written about once per second to `data\frames` as timestamped JPEGs. Existing files are never overwritten; a numeric suffix is added if a filename already exists. To choose another camera or output directory, run `python -m claimagent.capture --camera 1 --frames-dir data\frames` with the desired device index and directory.

For a combined capture-and-scan demonstration, run:

```powershell
$env:PYTHONPATH = "src"
python -m claimagent.capture --frames-dir data\frames --scan-after-capture --scan-output-dir output\capture-scan-001
```

When the capture window closes, the existing offline scanner processes those saved frames with its default contour detector and writes `claim_packet.json` and `inventory_report.md` in the separate scan output directory. Choose a fresh output directory for each run; existing scan outputs are preserved and the scanner stops if either output file already exists. The packet contains **unreviewed detection candidates**, not confirmed books: inspect each candidate and its source frame before using it as an identity or claim. This option does not load YOLO, and the default capture-only command remains unchanged.

If Windows blocks camera access, allow desktop apps to use the camera in Windows privacy settings and try again. If the camera is disconnected or capture fails, the utility reports the problem and releases the camera. Closing the window cancels capture cleanly. In capture-only mode, output consists only of raw frames: it does not create book or item records, identify contents, infer dimensions or prices, or update a claim packet. Review the frames and add observations separately using the existing JSON pipeline.

This interface requires a graphical desktop session. Webcam and Windows speech behavior depend on device permissions, drivers, and installed Windows speech voices; the preview and spoken cue cannot be hardware-verified by automated tests.

## Detect book-spine candidates (offline)

With the project environment active, scan saved JPEGs using the existing OpenCV and Pillow dependencies:

```powershell
$env:PYTHONPATH = "src"
python -m claimagent.recognition --frames-dir data\frames --output-dir output\scan-001
```

The scanner processes `.jpg` and `.jpeg` frames in filename order and writes the normal `claim_packet.json` and `inventory_report.md` formats. Use a new output directory each run; existing scan outputs are preserved and the command stops rather than replacing them. Each candidate is an `unidentified` book with its source frame, timestamp, `detection_confidence` (a contour ranking heuristic or the YOLO model score, depending on the selected detector), optional separate `ocr_confidence`, and optional raw `spine_text`. The existing evidence checks and deterministic totals run when the packet is built. Dimensions and prices remain empty. Extracted text is kept as review evidence; the scanner does not decide which words are the title or author.

To explicitly select the local YOLO11 OBB backend, use the checkpoint and settings below. This is an **inference smoke scan**, not an accuracy evaluation. Put the eight selected images in `data\frames\yolo-validation-8`; a labeled metric still requires comparing predictions with their label files. The checkpoint is never loaded unless `--detector yolo` is selected.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m claimagent.recognition `
  --frames-dir data\frames\yolo-validation-8 `
  --output-dir output\yolo-validation-8 `
  --detector yolo `
  --model .training_envs\bookspine-yolo-cpu-py311\weights\yolo11n-obb.pt `
  --conf 0.25 --iou 0.3 --imgsz 1280 --device cpu
```

The command stops before importing/loading Ultralytics when less than 4 GiB of system RAM is available. No real inference has been run as part of this integration work because available RAM was below that guard.

The current environment has no OCR engine installed, so this command reports OCR as unavailable and leaves spine text, title, and author unknown. If Tesseract is already installed locally, specify its executable explicitly, for example `python -m claimagent.recognition --frames-dir data\frames --output-dir output\scan-001 --tesseract C:\Tools\Tesseract-OCR\tesseract.exe`. Tesseract runs locally and requires no API key or network access. No OCR software or models are installed or downloaded by this project.

Detection uses geometric contour heuristics, not a trained book detector. The score is a ranking hint, not a calibrated probability; book-like rectangles and partial spines can be missed or falsely flagged. Cross-frame suppression only removes visually similar crops in adjacent files, so it can miss a repeated book or suppress similar-looking books. OCR output also needs human verification. There are no measured accuracy results; inspect every candidate and the original frame before using it as an identity claim. Frames whose names were not made by the capture utility use filesystem modification time as traceability metadata, clearly labeled as not verified capture time.

## Review a video in the Streamlit app

Install the project requirements in the active environment, then from the project root launch:

```powershell
$env:PYTHONPATH = "src"
python -m streamlit run src/claimagent/video_app.py
```

Upload an MP4 and select **Extract frames and detect candidates**. The app samples up to ten frames evenly across the clip and calls the contour scanner. Nested edge contours are suppressed only when their boxes are almost coincident; cross-frame matching requires strong spatial agreement after alignment and checks the full short sweep. Candidate detections and repeated matches are reported as detections, never as actual book counts. Expand each candidate to see its source frame with the detected bounding box outlined in red, its timestamp, and OCR text evidence. Choose **Confirm as a physical book**, **Reject as not a book**, or leave it pending. Only explicitly confirmed candidates enter inventory totals; rejected detections remain documented in the review queue, and pending candidates remain in the packet's separate candidate list. A confirmed but unidentified book is counted separately from identified books. Tesseract OCR is editable evidence and is never treated as a confirmed title or author. Unknown prices and dimensions stay Unknown. Enter country, currency, and recording time when known, then build and download `claim_packet.json` and `inventory_report.md`.

The optional Google Books lookup is available only for candidates marked identified. It requires an exact ISBN match when ISBN is supplied, or exact normalized title and author matches otherwise. It reports the Google Books eBook retail price, its currency, link, and retrieval date; this is an eBookstore price and may not represent a physical copy's replacement value. A missing, ambiguous, unavailable, or unpriced result remains **Unknown**. Price lookup requires internet access; the rest of video extraction, detection, review, and export runs locally. Uploaded videos and extracted frames are kept in a temporary app-session directory. The interface does not upload media to Google Books.

## Build a packet

Prepare a UTF-8 JSON file containing reviewed observations. Missing values should be omitted or `null`; do not fill unknown values with estimates. Each book and item should have a stable `id` and either a `frame_ref` relative to the frames directory or a `frame_timestamp` formatted as `HH:MM:SS[.fraction]` or an ISO timestamp. A frame file reference must resolve to a readable image inside that directory. Timestamp-only references are retained as traceability metadata but cannot be verified as image files.

Non-null dimensions must be positive and include a measurement method. Book and item measurements without a valid frame or timestamp are moved to the review queue and are not included in shelf-run calculations. Room measurements without a scale method are also left unknown. For a room with a non-rectangular `shape_note`, area is not derived from length and width; provide a separately measured area and its method if available.

Example input shape:

```json
{
  "sweep": {
    "id": "sweep-001",
    "captured_at": "2026-10-29T10:00:00Z",
    "duration_s": 95,
    "country": "IN",
    "currency": "INR"
  },
  "room": {
    "length_m": 4.2,
    "width_m": 3.1,
    "height_m": 2.6,
    "shelved_wall_area_m2": 5.4,
    "scale_method": "reviewed known wall dimensions",
    "confidence": 0.8
  },
  "books": [
    {
      "id": "book-001",
      "shelf": "A",
      "position": 1,
      "frame_ref": "shelf-a-001.jpg",
      "frame_timestamp": "00:00:08.200",
      "orientation": "upright",
      "status": "unidentified",
      "spine_height_cm": 21.0,
      "spine_thickness_cm": 2.4,
      "measurement_method": "reviewed marker scale",
      "measurement_confidence": 0.7
    }
  ],
  "items": []
}
```

Only provide a price amount with its source name, URL, retrieval date, and currency. Used-book values also require a condition assumption. A converted book price additionally requires `converted: true`, `original_amount`, `original_currency`, `conversion_rate` (original currency to reported currency), `conversion_source`, and `conversion_date`. The reported amount must match `original_amount × conversion_rate` within rounding tolerance. The pipeline does not perform lookups or exchange-rate retrieval. Prices in a currency different from the sweep currency remain in the line record but are excluded from totals and sent to review.

From the repository root, run:

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m claimagent.pipeline path\to\reviewed_observations.json --frames-dir data\frames --output-dir output
```

Bash:

```bash
PYTHONPATH=src python -m claimagent.pipeline path/to/reviewed_observations.json --frames-dir data/frames --output-dir output
```

The command writes `output/claim_packet.json` and `output/inventory_report.md`. Invalid or missing evidence, invalid dimensions, unsupported prices, duplicate IDs, and missing information are recorded in `review_queue`; a bad price is removed from the applicable total without dropping the rest of its inventory line. A repeated non-empty ID is counted once. Missing IDs receive a deterministic local line ID and a review entry.

For programmatic calculations, pass the packet through `validate_packet_evidence(packet, frames_dir)` before calling `compute_totals` or `finalize`. Those functions require the returned `EvidenceCheckedPacket`; passing a raw packet raises `TypeError`. Lines without a valid image or traceable timestamp remain in the packet and review queue but are excluded from counts and financial/dimension totals, with the count recorded in `lines_without_valid_evidence`.

## Deterministic rules

- Book count includes evidence-validated book lines of every status. Duplicates with the same non-empty ID are excluded after the first accepted occurrence; lines without valid evidence remain in the packet but not calculations.
- Shelf run sums only known thicknesses of upright books and converts centimeters to meters.
- Book prices count only when the book is identified, the amount has evidence, and its currency matches the sweep currency.
- Item totals include only valid sourced amounts/ranges in the sweep currency and valid `priced`/`range` status.
- Unknown prices and dimensions remain unknown; the report shows counts of unpriced lines separately from monetary totals.
- For rooms without a non-rectangular `shape_note`, floor area is derived as length × width only when absent, and wall area as 2 × (length + width) × height only when absent. Conflicting supplied area is preserved and flagged for review. Shelved wall area is never inferred.
- Capture and recognition utilities do not confirm book identities, calibrate or measure rooms, or retrieve prices. Detection candidates and OCR text remain subject to human review.

## Current scaffold assets

`src/claimagent/schema.py` defines the packet models, `src/claimagent/totals.py` calculates totals, and `src/claimagent/checks.py` validates frame files. `scripts/make_marker.py` generates an ArUco marker image; print size and physical calibration still need independent verification before use.
