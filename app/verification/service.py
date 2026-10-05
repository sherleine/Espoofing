"""Compatibility entry points for the existing verification integration."""

from app.verification.faces import read_video
from app.verification.main import TARGET_FRAMES, verify_registration


def verify_faces(profile_photo_path, video_path, challenge_nonce=None, include_pad=True):
    """Run the complete registration verification, including PAD.

    `include_pad` remains in the signature for compatibility with existing
    callers. Registration now always includes PAD so the public API needs
    only one verification call.
    """
    return verify_registration(
        profile_photo_path,
        video_path,
        challenge_nonce,
        include_pad=True,
    )


def extract_frames(video_path, target=TARGET_FRAMES):
    """Sample `target` frames evenly from the video. Returns (frames, info)."""
    frames, _, info = read_video(video_path, target)
    return frames, info
