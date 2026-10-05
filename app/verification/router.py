# app/verification/router.py
import hmac
import logging
import os

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from app.verification.main import EXAM_MAX_FRAMES, end_exam_session, get_exam_status
from app.verification.registration_liveness import issue_challenge
from app.verification.view import exam_window, start_exam, verify_email_photo, verify_pad

SERVICE_KEY = os.getenv("VERIFICATION_SERVICE_KEY", "")
if not SERVICE_KEY:
    logging.getLogger("app.verification").warning(
        "VERIFICATION_SERVICE_KEY is not set: verification endpoints are unauthenticated")


def require_service_key(x_service_key: str = Header(default="")):
    if SERVICE_KEY and not hmac.compare_digest(x_service_key.encode(), SERVICE_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid service key")


router = APIRouter(dependencies=[Depends(require_service_key)])

MAX_VIDEO_BYTES = 25 * 1024 * 1024
MAX_WINDOW_BYTES = 8 * 1024 * 1024


@router.post("/verify-email-photo")
async def verify_email_photo_endpoint(
    email: str = Form(...),
    examId: int = Form(...),
    email_photo: UploadFile = File(...),
    live_photo: UploadFile = File(...),
    challenge_nonce: str | None = Form(None),
):
    """Registration call: identity and liveness only."""
    profile_bytes = await email_photo.read()
    video_bytes = await live_photo.read()

    if len(video_bytes) > MAX_VIDEO_BYTES:
        return {"verified": False, "is_match": False, "reason_code": "VIDEO_TOO_LARGE",
                "reason": "Video is larger than 25 MB", "failed_checks": ["VIDEO_TOO_LARGE"]}

    return await verify_email_photo(
        email,
        examId,
        profile_bytes,
        video_bytes,
        video_filename=live_photo.filename,
        video_content_type=live_photo.content_type,
        challenge_nonce=challenge_nonce,
    )


@router.post("/verify-pad")
async def verify_pad_endpoint(
    email: str = Form(...),
    live_photo: UploadFile = File(...),
):
    """Second and final registration call: passive presentation-attack detection."""
    video_bytes = await live_photo.read()
    if len(video_bytes) > MAX_VIDEO_BYTES:
        return {"ok": False, "reason_code": "VIDEO_TOO_LARGE",
                "reason": "Video is larger than 25 MB"}

    return await verify_pad(
        email,
        video_bytes,
        video_filename=live_photo.filename,
        video_content_type=live_photo.content_type,
    )


@router.get("/verification/challenge")
async def challenge_endpoint():
    return issue_challenge()


@router.post("/exam/session/start")
async def exam_start_endpoint(
    email: str = Form(...),
    examId: int = Form(...),
    email_photo: UploadFile = File(...),
):
    return await start_exam(email, examId, await email_photo.read())


@router.post("/exam/session/{session_id}/window")
async def exam_window_endpoint(
    session_id: str,
    frames: list[UploadFile] | None = File(None),
    clip: UploadFile | None = File(None),
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
