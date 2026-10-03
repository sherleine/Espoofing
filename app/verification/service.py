# app/verification/service.py
"""
Compatibility wrapper. The logic now lives in main.py and the capability
modules (face_verification, registration_liveness, face_tracking,
anti_spoof, risk_engine). Existing callers keep using verify_faces().
"""

from app.verification.main import REASONS, read_video, verify_candidate  # noqa: F401
from app.verification.main import TARGET_FRAMES


def verify_faces(profile_photo_path, video_path, challenge_nonce=None):
    return verify_candidate(profile_photo_path, video_path, mode="registration",
                            challenge_nonce=challenge_nonce)


def extract_frames(video_path, target=TARGET_FRAMES):
    """Sample `target` frames evenly from the video. Returns (frames, info)."""
    frames, _, info = read_video(video_path, target)
    return frames, info
