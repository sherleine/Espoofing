# app/verification/router.py
from fastapi import APIRouter, File, Form, UploadFile

from app.verification.view import verify_email_photo

router = APIRouter()

MAX_VIDEO_BYTES = 25 * 1024 * 1024   # 25 MB


@router.post("/verify-email-photo")
async def verify_email_photo_endpoint(
    email: str = Form(...),
    examId: int = Form(...),
    email_photo: UploadFile = File(...),
    live_photo: UploadFile = File(...),   # this is the selfie VIDEO (name kept for the frontend)
):
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
    )
