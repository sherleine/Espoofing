from app.verification.main import verify_registration


def verify_faces(profile_photo_path, video_path, challenge_nonce=None):
    return verify_registration(profile_photo_path, video_path, challenge_nonce)
