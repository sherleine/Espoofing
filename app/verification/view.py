import os
import re
import time

from fastapi.concurrency import run_in_threadpool

from app.verification.service import verify_faces

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
    content_type = (content_type or "").split(";")[0].strip()
    return CONTENT_TYPE_EXT.get(content_type, ".webm")


def _write(path: str, data: bytes):
    with open(path, "wb") as file:
        file.write(data)


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
        return {
            "verified": False,
            "is_match": False,
            "reason_code": "PROFILE_UNREADABLE",
            "reason": "Profile photo is empty",
            "failed_checks": ["PROFILE_UNREADABLE"],
        }

    if not video_bytes:
        return {
            "verified": False,
            "is_match": False,
            "reason_code": "VIDEO_UNREADABLE",
            "reason": "Video is empty",
            "failed_checks": ["VIDEO_UNREADABLE"],
        }

    stamp = int(time.time() * 1000)
    base = f"{_safe(email)}_{examId}_{stamp}"
    photo_path = os.path.join(UPLOAD_DIR, f"{base}_profile.jpg")
    video_path = os.path.join(
        UPLOAD_DIR,
        f"{base}_live{_video_ext(video_filename, video_content_type)}",
    )

    _write(photo_path, profile_photo_bytes)
    _write(video_path, video_bytes)

    try:
        return await run_in_threadpool(
            verify_faces,
            photo_path,
            video_path,
            challenge_nonce,
        )
    finally:
        _remove(photo_path, video_path)
