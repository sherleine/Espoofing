"""Registration behaviour that differs from the original service.py."""

from app.verification import face_verification
from app.verification.faces import analyze_frame, pose
from conftest import jitter_frames, models, write_image, write_video


@models
def test_side_profile_frames_do_not_count_as_a_different_person(tmp_path, people, side_profile):
    """Candidates often turn far to the side. Face recognition is unreliable on
    side profiles, so those frames are left out of the same-person and
    identity checks instead of being reported as a different person."""
    turned = analyze_frame(side_profile, embedding=False)[0]
    assert abs(pose(turned)[1]) > face_verification.MAX_IDENTITY_YAW_DEG

    from app.verification import main
    profile = write_image(tmp_path / "profile.jpg", people[0])
    video = write_video(tmp_path / "turning.mp4",
                        jitter_frames(people[0], 30, seed=1) + jitter_frames(side_profile, 10, seed=2))
    result = main.verify_registration(profile, video)

    assert "DIFFERENT_PEOPLE_IN_VIDEO" not in result["failed_checks"]
    assert result["identity_match"] and result["identity"]["match_ratio"] == 1.0
