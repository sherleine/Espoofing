# app/verification/router.py
import hmac
import logging
import os

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from app.verification.main import REASONS, EXAM_MAX_FRAMES, end_exam_session, get_exam_status
from app.verification.registration_liveness import issue_challenge
from app.verification.view import exam_window, start_exam, verify_email_photo, verify_identity

# Only the .NET backend should call this service: it supplies the trusted
# profile photo. Set VERIFICATION_SERVICE_KEY and send it as X-Service-Key.
# Unset = no check (local development only).
SERVICE_KEY = os.getenv("VERIFICATION_SERVICE_KEY", "")
if not SERVICE_KEY:
    logging.getLogger("app.verification").warning(
        "VERIFICATION_SERVICE_KEY is not set: verification endpoints are unauthenticated")


def require_service_key(x_service_key: str = Header(default="")):
    if SERVICE_KEY and not hmac.compare_digest(x_service_key.encode(), SERVICE_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid service key")


router = APIRouter(dependencies=[Depends(require_service_key)])

MAX_VIDEO_BYTES = 25 * 1024 * 1024   # 25 MB
MAX_WINDOW_BYTES = 8 * 1024 * 1024   # one exam window (frames or clip)


@router.post("/verify-email-photo")
async def verify_email_photo_endpoint(
    email: str = Form(...),
    examId: int = Form(...),
    email_photo: UploadFile = File(...),
    live_photo: UploadFile = File(...),   # this is the selfie VIDEO (name kept for the frontend)
    challenge_nonce: str | None = Form(None),   # from GET /verification/challenge
):
    profile_bytes = await email_photo.read()
    video_bytes = await live_photo.read()

    if len(video_bytes) > MAX_VIDEO_BYTES:
        return _video_too_large()

    return await verify_email_photo(
        email,
        examId,
        profile_bytes,
        video_bytes,
        video_filename=live_photo.filename,
        video_content_type=live_photo.content_type,
        challenge_nonce=challenge_nonce,
    )


def _video_too_large():
    return {"verified": False, "is_match": False, "reason_code": "VIDEO_TOO_LARGE",
            "reason": "Video is larger than 25 MB", "failed_checks": ["VIDEO_TOO_LARGE"]}


@router.post("/register")
async def register_endpoint(
    email: str = Form(...),
    examId: int = Form(...),
    profile_photo: UploadFile = File(...),
    video: UploadFile = File(...),
    challenge_nonce: str = Form(...),           # from GET /verification/challenge
):
    """Registration always uses the active challenge. On success the response
    holds `reference_embedding`: .NET stores it and sends it to /verify and
    /exam/session/start."""
    # an empty nonce must not fall back to the weaker movement check
    if not challenge_nonce.strip():
        return {"verified": False, "is_match": False, "reason_code": "CHALLENGE_MISSING",
                "reason": REASONS["CHALLENGE_MISSING"], "failed_checks": ["CHALLENGE_MISSING"]}
    return await verify_email_photo_endpoint(email, examId, profile_photo, video, challenge_nonce)


@router.post("/verify")
async def verify_endpoint(
    email: str = Form(...),
    examId: int = Form(...),
    reference_embedding: str = Form(...),       # JSON saved from the /register response
    video: UploadFile = File(...),
    challenge_nonce: str | None = Form(None),   # with a nonce: active; without: passive
):
    video_bytes = await video.read()
    if len(video_bytes) > MAX_VIDEO_BYTES:
        return _video_too_large()
    return await verify_identity(email, examId, reference_embedding, video_bytes,
                                 video_filename=video.filename,
                                 video_content_type=video.content_type,
                                 challenge_nonce=challenge_nonce or None)


@router.get("/verification/challenge")
async def challenge_endpoint():
    """Random ordered actions + single-use nonce for registration liveness."""
    return issue_challenge()


# Exam monitoring is passive: the candidate is never asked to do anything.
@router.post("/exam/session/start")
async def exam_start_endpoint(
    email: str = Form(...),
    examId: int = Form(...),
    reference_embedding: str | None = Form(None),   # JSON saved from the registration response
    email_photo: UploadFile | None = File(None),    # fallback for candidates registered earlier
):
    photo = await email_photo.read() if email_photo else None
    return await start_exam(email, examId, photo, reference_embedding)


@router.post("/exam/session/{session_id}/window")
async def exam_window_endpoint(
    session_id: str,
    frames: list[UploadFile] | None = File(None),   # JPEG/PNG snapshots, in time order
    clip: UploadFile | None = File(None),           # or one short video clip
):
    if frames:
        if len(frames) > EXAM_MAX_FRAMES:
            raise HTTPException(status_code=413, detail=f"At most {EXAM_MAX_FRAMES} frames per window")
        images = [await f.read() for f in frames]
        if sum(len(b) for b in images) > MAX_WINDOW_BYTES:
            raise HTTPException(status_code=413, detail="Window too large")
        return await exam_window(session_id, images=images)
    if clip:
        data = await clip.read()
        if len(data) > MAX_WINDOW_BYTES:
            raise HTTPException(status_code=413, detail="Window too large")
        return await exam_window(session_id, clip=(data, clip.filename, clip.content_type))
    return await exam_window(session_id)


@router.get("/exam/session/{session_id}")
async def exam_status_endpoint(session_id: str):
    return get_exam_status(session_id)


@router.post("/exam/session/{session_id}/end")
async def exam_end_endpoint(session_id: str):
    return end_exam_session(session_id)
