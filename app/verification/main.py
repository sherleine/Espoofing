"""
Verification flow: which checks run, and in what order.
Registration is one request (verify_registration); the exam is a session of
short frame windows (start_exam_session, process_exam_window, ...).
"""

import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.verification import (
    anti_spoof,
    face_tracking,
    face_verification,
    faces,
    registration_liveness,
    risk_engine,
)

log = logging.getLogger("app.verification")

faces.load_models()
anti_spoof.load_model()

TARGET_FRAMES = 25
MIN_FRAMES = 10
MIN_DURATION_SEC = 1.5

CHALLENGE_TARGET_FRAMES = 40
CHALLENGE_EMBED_FRAMES = 16

REQUIRE_CHALLENGE = os.getenv("VERIFICATION_REQUIRE_CHALLENGE", "0") == "1"
PAD_ENFORCE_REGISTRATION = os.getenv("PAD_ENFORCE_REGISTRATION", "1") == "1"

EXAM_DET_SIZE = (480, 480)
EXAM_WINDOW_FRAMES = 8
EXAM_FRAME_INTERVAL_MS = 250
EXAM_MAX_FRAMES = 32
SESSION_TTL_SEC = 6 * 3600

REASONS = {
    "OK": "Verified successfully",
    "PROFILE_UNREADABLE": "Profile photo could not be read",
    "NO_FACE_IN_PROFILE": "No face found in the profile photo",
    "VIDEO_UNREADABLE": "Video could not be decoded (unsupported format or corrupt file)",
    "VIDEO_TOO_SHORT": "Video is too short or has too few frames",
    "FACE_NOT_CONSISTENT": "Face was not visible in enough of the video",
    "MULTIPLE_FACES_IN_VIDEO": "More than one person appeared in the video",
    "DIFFERENT_PEOPLE_IN_VIDEO": "The face changed to a different person during the video",
    "NO_HEAD_MOVEMENT": "No head movement detected (turn your head left and right)",
    "NO_FACIAL_MOVEMENT": "No natural facial movement detected (blink or open your mouth)",
    "FACE_MISMATCH": "The person in the video does not match the profile photo",
    "CHALLENGE_MISSING": "A liveness challenge is required: request one first",
    "CHALLENGE_INVALID": "The liveness challenge is unknown, already used or expired",
    "CHALLENGE_FAILED": "The requested actions were not performed in the requested order",
    "SPOOF_SUSPECTED": "The video looks like a photo or screen held in front of the camera",
}


def _result(reason_code, failed_checks=None, **extra):
    return {
        "verified": reason_code == "OK",
        "reason_code": reason_code,
        "reason": REASONS[reason_code],
        "failed_checks": failed_checks or ([] if reason_code == "OK" else [reason_code]),
        **extra,
    }


def verify_registration(profile_photo_path, video_path, challenge_nonce=None, include_pad=True):
    img = cv2.imread(profile_photo_path)
    if img is None:
        return _result("PROFILE_UNREADABLE")
    profile_faces = faces.analyze_frame(faces.resize(img), landmarks=False)
    if not profile_faces:
        return _result("NO_FACE_IN_PROFILE")
    profile = face_verification.largest(profile_faces)

    challenge = None
    if challenge_nonce:
        challenge, error = registration_liveness.consume_challenge(challenge_nonce)
        if error:
            return _result(error)
    elif REQUIRE_CHALLENGE:
        return _result("CHALLENGE_MISSING")

    frames, times, video_info = faces.read_video(
        video_path, CHALLENGE_TARGET_FRAMES if challenge else TARGET_FRAMES)
    if not frames:
        return _result("VIDEO_UNREADABLE", video=video_info)
    dur = video_info.get("duration_sec")
    if len(frames) < MIN_FRAMES or (dur is not None and dur < MIN_DURATION_SEC):
        return _result("VIDEO_TOO_SHORT", video={**video_info, "frames_sampled": len(frames)})

    faces_per_frame = [faces.analyze_frame(f, embedding=False) for f in frames]
    summary = face_tracking.summarize(faces_per_frame)
    video_info.update({
        "frames_sampled": summary["frames_sampled"],
        "frames_with_face": summary["frames_with_face"],
        "face_ratio": round(summary["face_ratio"], 3),
        "multi_face_ratio": round(summary["multi_face_ratio"], 3),
    })
    presence_failure = face_tracking.check_presence(summary)
    if presence_failure:
        return _result(presence_failure, video=video_info)

    primary = summary["primary"]
    video_faces = [f for f in primary if f is not None]

    with_face = [i for i, f in enumerate(primary) if f is not None]
    if challenge and len(with_face) > CHALLENGE_EMBED_FRAMES:
        pick = np.linspace(0, len(with_face) - 1, CHALLENGE_EMBED_FRAMES).astype(int)
        with_face = [with_face[j] for j in pick]
    emb = np.stack([faces.embed(frames[i], primary[i]) for i in with_face])
    consistency = face_tracking.consistency(emb)
    video_info["min_consistency"] = round(float(consistency.min()), 3)

    liveness, movement_failed = registration_liveness.movement_check(video_faces)
    challenge_block = None
    if challenge:
        challenge_block = registration_liveness.evaluate_challenge(
            challenge, [registration_liveness.measure(f) for f in primary], times)
        liveness_failed = [] if challenge_block["passed"] else ["CHALLENGE_FAILED"]
    else:
        liveness_failed = movement_failed

    identity, matched = face_verification.compare(emb, profile)

    failed = []
    if consistency.min() < face_tracking.SAME_PERSON_THRESHOLD:
        failed.append("DIFFERENT_PEOPLE_IN_VIDEO")
    failed += liveness_failed
    if not matched:
        failed.append("FACE_MISMATCH")

    result = _result(
        failed[0] if failed else "OK",
        failed_checks=failed,
        live=not liveness_failed,
        identity_match=matched,
        similarity=identity["similarity"],
        identity=identity,
        liveness=liveness,
        video=video_info,
    )
    if challenge_block:
        result["challenge"] = challenge_block

    if include_pad:
        pad = anti_spoof.detect_spoof(frames, primary)
        result["anti_spoof"] = pad
        if PAD_ENFORCE_REGISTRATION and pad["status"] == "HIGH_RISK":
            result["failed_checks"].insert(0, "SPOOF_SUSPECTED")
            result["reason_code"] = "SPOOF_SUSPECTED"
            result["reason"] = REASONS["SPOOF_SUSPECTED"]
            result["verified"] = False
            result["live"] = False

    return result


def run_pad(video_path):
    frames, _, video_info = faces.read_video(video_path, TARGET_FRAMES)
    if not frames:
        return {
            "status": "NOT_RUN",
            "reason_code": "VIDEO_UNREADABLE",
            "reason": REASONS["VIDEO_UNREADABLE"],
            "frames_scored": 0,
            "frames_skipped": 0,
        }

    faces_per_frame = [faces.analyze_frame(f, embedding=False) for f in frames]
    summary = face_tracking.summarize(faces_per_frame)
    if summary["frames_with_face"] == 0:
        return {
            "status": "NOT_RUN",
            "reason_code": "NO_FACE",
            "reason": "No face found in the video",
            "frames_scored": 0,
            "frames_skipped": len(frames),
        }

    pad = anti_spoof.detect_spoof(frames, summary["primary"])
    return {
        **pad,
        "video": {
            "frames_sampled": summary["frames_sampled"],
            "frames_with_face": summary["frames_with_face"],
            "face_ratio": round(summary["face_ratio"], 3),
        },
    }


@dataclass
class ExamSession:
    session_id: str
    email: str
    exam_id: int
    tracker: face_tracking.ExamTracker
    risk: risk_engine.RiskState = field(default_factory=risk_engine.RiskState)
    last_seen: float = field(default_factory=time.time)
    windows: int = 0
    events: list = field(default_factory=list)
    last_assessment: dict | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_sessions = {}
_sessions_lock = threading.Lock()
MAX_EVENTS = 500


def _get_session(session_id):
    now = time.time()
    with _sessions_lock:
        for sid in [s for s, v in _sessions.items() if now - v.last_seen > SESSION_TTL_SEC]:
            del _sessions[sid]
        return _sessions.get(session_id)


def start_exam_session(email, exam_id, profile_image):
    if profile_image is None:
        return {"started": False, "reason_code": "PROFILE_UNREADABLE",
                "reason": REASONS["PROFILE_UNREADABLE"]}
    profile_faces = faces.analyze_frame(faces.resize(profile_image), landmarks=False)
    if not profile_faces:
        return {"started": False, "reason_code": "NO_FACE_IN_PROFILE",
                "reason": REASONS["NO_FACE_IN_PROFILE"]}
    reference = face_verification.largest(profile_faces).normed_embedding

    session = ExamSession(
        session_id=secrets.token_urlsafe(16), email=email, exam_id=exam_id,
        tracker=face_tracking.ExamTracker(reference))
    _get_session(None)
    with _sessions_lock:
        _sessions[session.session_id] = session
    log.info("Exam session started: session=%s exam=%s", session.session_id, exam_id)
    return {
        "started": True,
        "session_id": session.session_id,
        "window_frames": EXAM_WINDOW_FRAMES,
        "frame_interval_ms": EXAM_FRAME_INTERVAL_MS,
        "next_window_sec": risk_engine.NEXT_WINDOW_SEC["NORMAL"],
    }


def process_exam_window(session_id, frames):
    started = time.perf_counter()
    session = _get_session(session_id)
    if session is None:
        return {"ok": False, "reason_code": "SESSION_NOT_FOUND",
                "reason": "Unknown or expired exam session"}
    frames = [faces.resize(f) for f in frames if f is not None][:EXAM_MAX_FRAMES]
    if not frames:
        return {"ok": False, "reason_code": "NO_FRAMES", "reason": "No decodable frames in window"}

    with session.lock:
        faces_per_frame = [faces.analyze_frame(f, landmarks=False, embedding=False, det_size=EXAM_DET_SIZE)
                           for f in frames]
        summary = face_tracking.summarize(faces_per_frame,
                                          min_rel_area=face_tracking.EXAM_MIN_REL_FACE_AREA)
        tracker = session.tracker
        face_count = tracker.face_count(summary)

        embedding = None
        if face_count != "NONE":
            picked = face_verification.pick_identity_face(frames, summary["primary"])
            if picked:
                embedding = faces.embed(*picked)

        facts = tracker.update(face_count, embedding)
        if face_count == "NONE":
            pad = {"status": "NOT_RUN", "spoof_score": None}
        else:
            pad = anti_spoof.detect_spoof(frames, summary["primary"])

        signals = {
            "face_count": face_count,
            "identity": facts["identity"],
            "continuity": facts["continuity"],
            "pad": pad["status"],
            "absent_windows": facts["absent_windows"],
        }
        assessment = session.risk.assess(signals)
        session.windows += 1
        session.last_seen = time.time()

        response = {
            "ok": True,
            "session_id": session_id,
            "window": session.windows,
            **assessment,
            "signals": {
                **facts,
                "pad": {k: pad.get(k) for k in
                        ("status", "spoof_score", "frames_scored", "frames_skipped")},
            },
            "faces": {
                "frames_sampled": summary["frames_sampled"],
                "frames_with_face": summary["frames_with_face"],
                "face_ratio": round(summary["face_ratio"], 3),
                "multi_face_ratio": round(summary["multi_face_ratio"], 3),
            },
            "processing_ms": round((time.perf_counter() - started) * 1000),
        }
        session.last_assessment = response
        flagged = assessment["state_changed"] or assessment["state"] != "NORMAL"
        if flagged:
            session.events.append({
                "time": time.time(), "window": session.windows, "state": assessment["state"],
                "reasons": assessment["reasons"], "signals": signals,
                "spoof_score": pad.get("spoof_score"),
                "reference_similarity": facts["reference_similarity"],
            })
            del session.events[:-MAX_EVENTS]

    if flagged:
        log.info("Exam window flagged: session=%s window=%d state=%s reasons=%s "
                 "spoof_score=%s similarity=%s",
                 session_id, response["window"], assessment["state"], assessment["reasons"],
                 pad.get("spoof_score"), facts["reference_similarity"])
    return response


def get_exam_status(session_id):
    session = _get_session(session_id)
    if session is None:
        return {"ok": False, "reason_code": "SESSION_NOT_FOUND",
                "reason": "Unknown or expired exam session"}
    return {
        "ok": True,
        "session_id": session_id,
        "email": session.email,
        "exam_id": session.exam_id,
        "windows": session.windows,
        "state": session.risk.state,
        "highest_state": _highest_state(session),
        "events": session.events,
        "last_assessment": session.last_assessment,
    }


def end_exam_session(session_id):
    status = get_exam_status(session_id)
    with _sessions_lock:
        _sessions.pop(session_id, None)
    if status["ok"]:
        log.info("Exam session ended: session=%s windows=%d highest_state=%s",
                 session_id, status["windows"], status["highest_state"])
    return status


def _highest_state(session):
    order = ["NORMAL", "SUSPICIOUS", "HIGH_RISK"]
    states = [e["state"] for e in session.events] + [session.risk.state]
    return max(states, key=order.index)
