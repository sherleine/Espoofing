# app/verification/service.py
"""Kept so existing callers of verify_faces/extract_frames keep working."""

from app.verification.verification_engine import read_video
from app.verification.main import REASONS, TARGET_FRAMES, verify_registration  # noqa: F401


def verify_faces(profile_photo_path, video_path, challenge_nonce=None):
    return verify_registration(profile_photo_path, video_path, challenge_nonce)


def extract_frames(video_path, target=TARGET_FRAMES):
    """Sample `target` frames evenly from the video. Returns (frames, info)."""
    frames, _, info = read_video(video_path, target)
    return frames, info
