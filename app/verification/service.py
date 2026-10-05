"""Kept so existing callers of verify_faces/extract_frames keep working."""

from app.verification.faces import read_video
from app.verification.main import REASONS, TARGET_FRAMES, verify_registration


def verify_faces(profile_photo_path, video_path, challenge_nonce=None, include_pad=True):
    """Keep the existing verification entry point; PAD can be split into its own call."""
    return verify_registration(profile_photo_path, video_path, challenge_nonce, include_pad=include_pad)


def extract_frames(video_path, target=TARGET_FRAMES):
    """Sample `target` frames evenly from the video. Returns (frames, info)."""
    frames, _, info = read_video(video_path, target)
    return frames, info
