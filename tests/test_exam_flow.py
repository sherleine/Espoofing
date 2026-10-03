"""Exam monitoring end to end with synthetic frames (no user actions)."""

import numpy as np
import pytest

from conftest import jitter_frames, models


@pytest.fixture
def session(people):
    from app.verification import main
    started = main.start_exam_session("a@example.com", 1, people[0])
    assert started["started"]
    yield started["session_id"]
    main.end_exam_session(started["session_id"])


def window(main, sid, frames):
    return main.verify_candidate(None, frames, mode="exam", session_id=sid)


@models
def test_same_candidate_is_normal(cv2, people, session):
    from app.verification import main
    for seed in range(3):
        r = window(main, session, jitter_frames(cv2, people[0], 8, seed=seed))
        assert r["ok"] and r["state"] == "NORMAL", r
        assert r["signals"]["face_count"] == "ONE"
        assert r["signals"]["identity"] == "MATCH"
        assert r["signals"]["pad"]["status"] in ("LOW_RISK", "ELEVATED", "HIGH_RISK", "INSUFFICIENT_QUALITY")


@models
def test_person_swap_is_high_risk(cv2, people, session):
    from app.verification import main
    window(main, session, jitter_frames(cv2, people[0], 8, seed=1))
    r = window(main, session, jitter_frames(cv2, people[1], 8, seed=2))
    assert r["signals"]["identity"] == "MISMATCH"
    assert r["signals"]["continuity"] == "CHANGED"
    assert r["state"] == "HIGH_RISK" and r["recommended_action"] == "FLAG_FOR_REVIEW"
    status = main.get_exam_status(session)
    assert status["highest_state"] == "HIGH_RISK" and status["events"]


@models
def test_multiple_faces(cv2, group_photo, session):
    from app.verification import main
    r = window(main, session, jitter_frames(cv2, cv2.resize(group_photo, (960, 664)), 6))
    assert r["signals"]["face_count"] == "MULTIPLE"
    assert r["state"] in ("SUSPICIOUS", "HIGH_RISK")


@models
def test_absence_then_return_forces_identity_check(cv2, people, session, monkeypatch):
    from app.verification import main
    monkeypatch.setattr(main, "IDENTITY_EVERY_N_WINDOWS", 1000)   # only forced checks
    blank = [np.full((480, 480, 3), 90, np.uint8)] * 6
    window(main, session, jitter_frames(cv2, people[0], 6))          # window 0: scheduled check
    for _ in range(2):
        r = window(main, session, blank)
    assert r["signals"]["face_count"] == "NONE" and r["state"] == "SUSPICIOUS"
    assert r["reasons"] == ["CANDIDATE_ABSENT"]
    r = window(main, session, jitter_frames(cv2, people[1], 6, seed=9))   # someone else sits down
    assert r["signals"]["face_returned"] and r["signals"]["identity"] == "MISMATCH"


@models
def test_unknown_session(people):
    from app.verification import main
    r = main.verify_candidate(None, [people[0]], mode="exam", session_id="nope")
    assert r == {"ok": False, "reason_code": "SESSION_NOT_FOUND", "reason": "Unknown or expired exam session"}


@models
def test_profile_without_face_cannot_start():
    from app.verification import main
    r = main.start_exam_session("x@example.com", 1, np.full((300, 300, 3), 127, np.uint8))
    assert not r["started"] and r["reason_code"] == "NO_FACE_IN_PROFILE"
