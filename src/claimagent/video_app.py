"""Streamlit interface for reviewing book-spine candidates from uploaded MP4s."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tempfile

_SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if _SOURCE_ROOT not in sys.path:
    # Streamlit executes this file as a script, so ensure the src-layout package
    # is importable even when the caller did not set PYTHONPATH.
    sys.path.insert(0, _SOURCE_ROOT)

from claimagent.recognition import TesseractOCR, scan_frames
from claimagent.video_workflow import (
    build_reviewed_video_packet,
    extract_representative_frames,
    lookup_google_books_price,
    packet_downloads,
)


def main() -> None:
    import streamlit as st
    from PIL import Image
    from PIL import ImageDraw

    st.set_page_config(page_title="Library Claim Agent", layout="wide")
    st.title("Library Contents Claim Agent")
    st.write("Upload an MP4, review detected spine candidates, and export a claim packet.")
    st.info("Detection and OCR are suggestions for review. Confirm each identity and evidence item yourself.")

    uploaded = st.file_uploader("Upload a video", type=["mp4"], help="MP4 videos only")
    if uploaded is None:
        st.caption("The video is processed locally in this app session. No video is uploaded to a recognition service.")
        return

    video_bytes = uploaded.getvalue()
    signature = sha256(video_bytes).hexdigest()
    old_signature = st.session_state.get("video_signature")
    if old_signature != signature:
        old_workdir = st.session_state.get("video_workdir")
        if old_workdir:
            shutil.rmtree(old_workdir, ignore_errors=True)
        for key in list(st.session_state):
            if key.startswith(("book_price_quote_", "book_price_checked_", "review_", "id_confidence_", "use_price_")):
                del st.session_state[key]
        st.session_state.pop("video_captured_at", None)
        st.session_state.pop("video_scan_packet", None)
        st.session_state.pop("video_frames", None)
        st.session_state.pop("video_timecodes", None)
        st.session_state.pop("video_duration", None)
        st.session_state.pop("video_exports", None)
        st.session_state["video_signature"] = signature
        st.session_state["video_filename"] = uploaded.name

    if st.button("Extract frames and detect candidates", type="primary"):
        workdir = Path(tempfile.mkdtemp(prefix="library-claim-video-"))
        video_path = workdir / "upload.mp4"
        frames_dir = workdir / "frames"
        try:
            video_path.write_bytes(video_bytes)
            with st.spinner("Extracting representative frames and scanning with OpenCV contours…"):
                extracted, duration = extract_representative_frames(video_path, frames_dir)
                packet = scan_frames(frames_dir)
            previous_workdir = st.session_state.get("video_workdir")
            if previous_workdir:
                shutil.rmtree(previous_workdir, ignore_errors=True)
            st.session_state["video_workdir"] = str(workdir)
            st.session_state["video_frames_dir"] = str(frames_dir)
            st.session_state["video_frames"] = [frame.__dict__ for frame in extracted]
            st.session_state["video_timecodes"] = {frame.frame_ref: frame.timecode for frame in extracted}
            st.session_state["video_duration"] = duration
            st.session_state["video_scan_packet"] = packet
            st.session_state.pop("video_exports", None)
        except Exception as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            st.error(f"Could not process this video: {exc}")
            return

    scan_packet = st.session_state.get("video_scan_packet")
    if scan_packet is None:
        st.caption("Choose a valid MP4, then select **Extract frames and detect candidates**.")
        return

    frames_dir = Path(st.session_state["video_frames_dir"])
    frames = st.session_state["video_frames"]
    st.success(f"Extracted {len(frames)} representative frames from {st.session_state['video_filename']}.")
    if TesseractOCR().available:
        st.caption("Local Tesseract was found; OCR text is included as unverified evidence.")
    else:
        st.caption("Tesseract is unavailable; candidates can still be reviewed, but OCR text is blank.")

    st.subheader("Extracted frame previews")
    columns = st.columns(3)
    for index, frame in enumerate(frames):
        with columns[index % len(columns)]:
            st.image(Image.open(frames_dir / frame["frame_ref"]), caption=frame["timecode"], use_container_width=True)

    st.subheader("Review detected book candidates")
    duplicate_entries = [
        entry for entry in scan_packet.review_queue
        if "not counted again" in entry.reason or "duplicate candidate suppressed" in entry.reason
    ]
    metrics = st.columns(2)
    metrics[0].metric("Candidate detections retained", len(scan_packet.candidates))
    metrics[1].metric("Repeated detections matched", len(duplicate_entries))
    flagged_count = sum(bool(book.candidate_warnings) for book in scan_packet.candidates)
    if flagged_count:
        st.warning(
            f"{flagged_count} remaining candidate(s) have low-specificity geometry. "
            "They are retained for manual review and are not automatically identified as books."
        )
    if not scan_packet.candidates:
        st.warning("No book-spine candidates were detected. Review the extracted frames manually.")

    reviewed_books = []
    excluded_candidates = []
    rejected_candidates = []
    regular_section_shown = False
    low_specificity_section_shown = False
    for book in scan_packet.candidates:
        if book.candidate_warnings and not low_specificity_section_shown:
            st.markdown("#### Low-specificity candidates — possible false positives, available for review")
            low_specificity_section_shown = True
        elif not book.candidate_warnings and not regular_section_shown:
            st.markdown("#### Other book-like candidates")
            regular_section_shown = True
        with st.expander(f"{book.id} · source {book.frame_ref} · contour score {book.detection_confidence or 0:.3f} (not a probability)", expanded=False):
            if book.candidate_warnings:
                st.warning(" · ".join(book.candidate_warnings) + ". Retained for manual review.")
            source_image = Image.open(frames_dir / book.frame_ref).convert("RGB")
            if book.detection_box:
                x, y, box_width, box_height = book.detection_box
                ImageDraw.Draw(source_image).rectangle(
                    (x, y, x + box_width, y + box_height), outline="red", width=max(2, source_image.width // 400)
                )
                st.caption(f"Source frame: {book.frame_ref} · time: {st.session_state['video_timecodes'].get(book.frame_ref, book.frame_timestamp or 'Unknown')} · bounding box (x, y, width, height): {book.detection_box}")
            else:
                st.caption(f"Source frame: {book.frame_ref} · bounding box: Unknown")
            st.image(source_image, caption="Candidate box outlined in red", use_container_width=True)
            alternate_evidence = [
                entry.details for entry in duplicate_entries
                if entry.ref_id == book.id and entry.details.get("primary_frame_ref") == book.frame_ref
            ]
            if alternate_evidence:
                st.caption("Additional matching evidence frames: " + ", ".join(
                    f"{item['repeated_frame_ref']} ({item.get('repeated_frame_timestamp') or 'time unknown'})"
                    for item in alternate_evidence
                ))
            disposition = st.selectbox(
                "Review this detection",
                ["Pending review", "Confirm as a physical book", "Reject as not a book"],
                index=0,
                key=f"disposition_{book.id}",
            )
            if disposition == "Reject as not a book":
                rejected_candidates.append(book.model_dump(mode="json"))
                continue
            if disposition != "Confirm as a physical book":
                continue
            st.caption(f"OCR suggestion (not an identification): {book.spine_text or 'Unknown'}")
            status_label = st.selectbox(
                "Review status", ["Unidentified", "Identified", "Needs appraisal"],
                index=0, key=f"review_status_{book.id}",
            )
            status = {"Unidentified": "unidentified", "Identified": "identified", "Needs appraisal": "needs_appraisal"}[status_label]
            title = st.text_input("Title", key=f"review_title_{book.id}")
            author = st.text_input("Author", key=f"review_author_{book.id}")
            publisher = st.text_input("Publisher", key=f"review_publisher_{book.id}")
            isbn = st.text_input("ISBN (optional, useful for exact price lookup)", key=f"review_isbn_{book.id}")
            corrected_spine_text = st.text_input("Spine text evidence", value=book.spine_text, key=f"review_text_{book.id}")
            identity_confidence = None
            if status == "identified":
                identity_confidence = st.slider("Your identification confidence", 0.0, 1.0, 0.9, 0.05, key=f"id_confidence_{book.id}")

            quote = st.session_state.get(f"book_price_quote_{book.id}")
            quote_checked = st.session_state.get(f"book_price_checked_{book.id}", False)
            if status == "identified" and (isbn.strip() or title.strip()):
                if st.button("Look up Google Books eBook price", key=f"lookup_price_{book.id}"):
                    with st.spinner("Searching Google Books…"):
                        quote = lookup_google_books_price(title, author, isbn, st.session_state.get("video_country", ""))
                    st.session_state[f"book_price_quote_{book.id}"] = quote
                    st.session_state[f"book_price_checked_{book.id}"] = True
                    quote_checked = True
                if quote is None and quote_checked:
                    st.info("Price: Unknown (no exact match with a listed retail price).")
                elif quote is not None:
                    st.write(f"Google Books eBook retail price: {quote.amount:g} {quote.currency}")
                    st.caption(f"Source: {quote.url} · Retrieved {quote.retrieved_at}")
            else:
                quote = None

            row = book.model_dump(mode="python")
            row.update({
                "status": status,
                "title": title.strip() if status == "identified" else "",
                "author": author.strip() if status == "identified" else "",
                "publisher": publisher.strip() if status == "identified" else "",
                "isbn": isbn.strip() if status == "identified" else "",
                "spine_text": corrected_spine_text.strip(),
                "frame_timestamp": st.session_state["video_timecodes"].get(book.frame_ref, book.frame_timestamp),
                "timestamp_source": "uploaded_video_timecode",
                "id_confidence": identity_confidence,
                "replacement_cost": None,
                "price_quote": None,
            })
            if quote is not None and st.checkbox(
                "Record this eBook retail quote in the packet (excluded from replacement totals)",
                key=f"use_price_{book.id}",
            ):
                row["price_quote"] = {
                    "amount": quote.amount,
                    "currency": quote.currency,
                    "source": quote.source,
                    "price_type": quote.price_type,
                    "url": quote.url,
                    "retrieved_at": quote.retrieved_at,
                }
            reviewed_books.append(row)

    st.metric("Reviewer-confirmed books this pass", len(reviewed_books))

    st.subheader("Sweep details")
    left, right = st.columns(2)
    with left:
        country = st.text_input("Country code", key="video_country", max_chars=2)
    with right:
        currency = st.text_input("Currency code", key="video_currency", max_chars=3)
    captured_at = st.text_input("Video recorded at (ISO 8601, if known)", key="video_captured_at")
    sweep = {
        "id": f"video-{signature[:12]}",
        "captured_at": captured_at.strip(),
        "device": f"uploaded video: {st.session_state['video_filename']}",
        "duration_s": st.session_state["video_duration"],
        "country": country.strip().upper(),
        "currency": currency.strip().upper(),
    }
    review_data = {"books": reviewed_books, "excluded_candidates": excluded_candidates,
                   "rejected_candidates": rejected_candidates, "sweep": sweep}
    review_signature = sha256(json.dumps(review_data, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    if st.button("Build claim packet and report", type="primary"):
        try:
            packet = build_reviewed_video_packet(
                scan_packet, reviewed_books, frames_dir,
                st.session_state["video_timecodes"], sweep, excluded_candidates, rejected_candidates,
            )
            json_bytes, report_bytes = packet_downloads(packet)
            st.session_state["video_exports"] = (review_signature, json_bytes, report_bytes)
        except Exception as exc:
            st.error(f"Could not build the claim packet: {exc}")

    exports = st.session_state.get("video_exports")
    if exports and exports[0] == review_signature:
        _, json_bytes, report_bytes = exports
        st.download_button("Download claim_packet.json", json_bytes, "claim_packet.json", "application/json")
        st.download_button("Download inventory_report.md", report_bytes, "inventory_report.md", "text/markdown")
    elif exports:
        st.info("Review fields changed. Build the packet again to refresh the downloads.")


if __name__ == "__main__":
    main()
