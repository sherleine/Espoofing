# app/verification/view.py
import os
import re
import time

from fastapi.concurrency import run_in_threadpool

from app.verification.service import verify_faces   # file is service.py, not services.py
from app.monitor.session_store import mark_verified

UPLOAD_DIR = "uploads/verification"
os.makedirs(UPLOAD_DIR, exist_ok=True)

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
CONTENT_TYPE_EXT = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
}


def _safe(name: str) -> str:
    """Make a string safe to use in a file name (emails contain @ and dots)."""
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


async def verify_email_photo(
    email: str,
    examId: int,
    profile_photo_bytes: bytes,
    video_bytes: bytes,
    video_filename: str | None = None,
    video_content_type: str | None = None,
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

    # InsightFace is CPU-heavy and blocking: keep it off the event loop.
    result = await run_in_threadpool(verify_faces, photo_path, video_path)

    print(f"[VERIFY] email={email} exam={examId} code={result['reason_code']} "
          f"failed={result['failed_checks']} similarity={result.get('similarity')} "
          f"liveness={result.get('liveness')}")

    if result["verified"]:
        mark_verified(email)

    return {
        **result,
        "is_match": result["verified"],   # kept for existing frontend code
    }
