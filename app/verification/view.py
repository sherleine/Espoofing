# app/verification/view.py
import logging
import os
import re
import tempfile
import time

import cv2
import numpy as np
from fastapi.concurrency import run_in_threadpool

from app.verification import faces, main
from app.verification.service import verify_faces
from app.monitor.session_store import mark_verified

log = logging.getLogger("app.verification")

UPLOAD_DIR = "uploads/verification"
os.makedirs(UPLOAD_DIR, exist_ok=True)
KEEP_UPLOADS = os.getenv("VERIFICATION_KEEP_UPLOADS", "0") == "1"

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
CONTENT_TYPE_EXT = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
}


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _video_ext(filename: str | None, content_type: str | None) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in VIDEO_EXTS:
        return ext
    base_type = (content_type or "").split(";")[0].strip()
    return CONTENT_TYPE_EXT.get(base_type, ".webm")


def _write(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def _remove(*paths):
    if KEEP_UPLOADS:
        return
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


async def verify_email_photo(
    email: str,
    examId: int,
    profile_photo_bytes: bytes,
    video_bytes: bytes,
    video_filename: str | None = None,
    video_content_type: str | None = None,
    challenge_nonce: str | None = None,
):
    if not profile_photo_bytes:
        return {"verified": False, "is_match": False, "reason_code": "PROFILE_UNREADABLE",
                "reason": "Profile photo is empty", "failed_checks": ["PROFILE_UNREADABLE"]}
    if not video_bytes:
        return {"verified": False, "is_match": False, "reason_code": "VIDEO_UNREADABLE",
                "reason": "Video is empty", "failed_checks": ["VIDEO_UNREADABLE"]}

    stamp = int(time.time() * 1000)
    base = f"{_safe(email)}_{examId}_{stamp}"
    photo_path = os.path.join(UPLOAD_DIR, f"{base}_profile.jpg")
    video_path = os.path.join(UPLOAD_DIR, f"{base}_live{_video_ext(video_filename, video_content_type)}")

    _write(photo_path, profile_photo_bytes)
    _write(video_path, video_bytes)

    started = time.perf_counter()
    try:
        result = await run_in_threadpool(
            verify_faces, photo_path, video_path, challenge_nonce, False
        )
    finally:
        _remove(photo_path, video_path)

    log.info("Registration verified: exam=%s code=%s failed=%s similarity=%s ms=%d",
             examId, result["reason_code"], result["failed_checks"], result.get("similarity"),
             (time.perf_counter() - started) * 1000)

    return {
        **result,
        "is_match": result["verified"],
    }


async def verify_pad(
    email: str,
    video_bytes: bytes,
    video_filename: str | None = None,
    video_content_type: str | None = None,
):
    if not video_bytes:
        return {"ok": False, "reason_code": "VIDEO_UNREADABLE", "reason": "Video is empty"}

    stamp = int(time.time() * 1000)
    video_path = os.path.join(
        UPLOAD_DIR, f"pad_{stamp}{_video_ext(video_filename, video_content_type)}"
    )
    _write(video_path, video_bytes)
    try:
        result = await run_in_threadpool(main.run_pad, video_path)
    finally:
        _remove(video_path)

    response = {"ok": True, "anti_spoof": result}
    if result["status"] == "LOW_RISK":
        mark_verified(email)
    return response


async def start_exam(email: str, examId: int, profile_photo_bytes: bytes):
    img = None
    if profile_photo_bytes:
        img = cv2.imdecode(np.frombuffer(profile_photo_bytes, np.uint8), cv2.IMREAD_COLOR)
    return await run_in_threadpool(main.start_exam_session, email, examId, img)


def _clip_frames(clip_bytes: bytes, filename: str | None, content_type: str | None):
    fd, path = tempfile.mkstemp(suffix=_video_ext(filename, content_type))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(clip_bytes)
        frames, _, _ = faces.read_video(path, main.EXAM_MAX_FRAMES)
        return frames
    finally:
        os.remove(path)


def _window(session_id, images, clip):
    if images:
        frames = faces.decode_images(images)
    else:
        frames = _clip_frames(*clip)
    return main.process_exam_window(session_id, frames)


async def exam_window(session_id: str, images: list[bytes] | None = None,
                      clip: tuple[bytes, str | None, str | None] | None = None):
    """One window of exam frames: several JPEG/PNG images or one short clip."""
    if not images and not (clip and clip[0]):
        return {"ok": False, "reason_code": "NO_FRAMES", "reason": "Send frames or a clip"}
    return await run_in_threadpool(_window, session_id, images, clip)
