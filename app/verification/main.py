# app/verification/main.py
"""
Orchestrator - the only module the rest of the application calls.

    verify_candidate(profile_photo_path, video_path, mode="registration",
                     challenge_nonce=None)              -> legacy-compatible dict
    verify_candidate(None, frames, mode="exam",
                     session_id=...)                    -> risk assessment dict

Exam monitoring is stateful (identity continuity, risk history), so it runs
inside a session:

    start_exam_session(email, exam_id, profile_image) -> {"session_id": ...}
    process_exam_window(session_id, frames)           -> assessment per window
    get_exam_status(session_id) / end_exam_session(session_id)

Faces are analysed ONCE here (detection, landmarks, embeddings) and handed
to the capability modules, which never call InsightFace themselves:

    face_verification      identity (profile vs live)
    registration_liveness  active challenge          - registration only
    face_tracking          face count / continuity
    anti_spoof             passive PAD               - registration (+) and exam
    risk_engine            NORMAL / SUSPICIOUS / HIGH_RISK - exam only
"""

import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

from app.verification import (
    anti_spoof,
    face_tracking,
    face_verification,
    registration_liveness,
    risk_engine,
)

log = logging.getLogger("app.verification")

_face_app = FaceAnalysis(
    name="buffalo_l",               # 1k3d68 -> face.pose, landmark_3d_68
    allowed_modules=["detection", "landmark_3d_68", "recognition"],
    providers=["CPUExecutionProvider"],
)
_face_app.prepare(ctx_id=0, det_size=(640, 640))
_detector = _face_app.det_model
_landmarks = _face_app.models["landmark_3d_68"]
_recognizer = _face_app.models["recognition"]
anti_spoof.warm_up()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MIN_DET_SCORE = 0.50            # ignore weak detections
MAX_SIDE = 960                  # downscale big frames for speed

TARGET_FRAMES = 25              # frames sampled from the video (legacy mode)
MIN_FRAMES = 10                 # minimum frames decoded
MIN_DURATION_SEC = 1.5

CHALLENGE_TARGET_FRAMES = 40    # denser sampling so short blinks are not missed
CHALLENGE_EMBED_FRAMES = 16     # identity needs fewer frames than liveness

_flag = lambda name, default: os.getenv(name, default).strip().lower() in ("1", "true", "yes")
REQUIRE_CHALLENGE = _flag("VERIFICATION_REQUIRE_CHALLENGE", "0")
PAD_ENFORCE_REGISTRATION = _flag("PAD_ENFORCE_REGISTRATION", "1")       # HIGH_RISK fails registration
PAD_REQUIRE_QUALITY = _flag("PAD_REQUIRE_QUALITY_AT_REGISTRATION", "0")  # unscorable video fails

EXAM_DET_SIZE = (480, 480)      # detection only, every exam frame
EXAM_WINDOW_FRAMES = 8          # suggested frames per window ...
EXAM_FRAME_INTERVAL_MS = 250    # ... ~4 fps -> a 2 s window
EXAM_MAX_FRAMES = 32            # hard cap per window
IDENTITY_EVERY_N_WINDOWS = 1    # embedding is the costly part (~0.2 s per face on CPU)
MAX_IDENTITY_YAW_DEG = 35.0     # profile views give unreliable embeddings
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
    # new
    "CHALLENGE_MISSING": "A liveness challenge is required: request one first",
    "CHALLENGE_INVALID": "The liveness challenge is unknown, already used or expired",
    "CHALLENGE_FAILED": "The requested actions were not performed in the requested order",
    "SPOOF_SUSPECTED": "The video looks like a photo or screen held in front of the camera",
    "VIDEO_QUALITY_TOO_LOW": "Face could not be analysed - improve lighting and hold the camera steady",
}


def _result(reason_code, failed_checks=None, **extra):
    return {
        "verified": reason_code == "OK",
        "reason_code": reason_code,
        "reason": REASONS[reason_code],
        "failed_checks": failed_checks or ([] if reason_code == "OK" else [reason_code]),
        **extra,
    }


# ---------------------------------------------------------------------------
# Frames and faces
# ---------------------------------------------------------------------------
def resize(img):
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def read_video(video_path, target=TARGET_FRAMES):
    """Sample `target` frames evenly. Returns (frames, timestamps_sec, info)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], {"error": "open_failed"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_ok = 0 < fps < 240

    # Browser WebM often reports 0 / garbage frame count -> count manually.
    if count <= 0 or count > 100000:
        count = 0
        while cap.grab():
            count += 1
        cap.release()
        cap = cv2.VideoCapture(video_path)

    if count == 0:
        cap.release()
        return [], [], {"error": "no_frames"}

    wanted = set(np.linspace(0, count - 1, min(target, count)).astype(int).tolist())
    frames, times, idx = [], [], 0
    while cap.grab():
        if idx in wanted:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                frames.append(resize(frame))
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                times.append(pos_ms / 1000.0 if pos_ms and pos_ms > 0
                             else (idx / fps if fps_ok else None))
        idx += 1
        if idx > max(wanted):
            break
    cap.release()

    duration = count / fps if fps_ok else None
    return frames, times, {"total_frames": count, "fps": fps, "duration_sec": duration}


def decode_images(images):
    """JPEG/PNG bytes -> BGR frames (undecodable images are dropped)."""
    frames = []
    for data in images:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if img is not None:
            frames.append(resize(img))
    return frames


def analyze_frame(frame, landmarks=True, embedding=True, det_size=None):
    """Detect faces (det_score >= MIN_DET_SCORE) and optionally add the 68
    3D landmarks + pose and the identity embedding to each."""
    bboxes, kpss = _detector.detect(frame, input_size=det_size)
    faces = []
    for i in range(bboxes.shape[0]):
        face = Face(bbox=bboxes[i, 0:4], kps=None if kpss is None else kpss[i],
                    det_score=bboxes[i, 4])
        if face.det_score < MIN_DET_SCORE:
            continue
        if landmarks:
            _landmarks.get(frame, face)
        if embedding:
            _recognizer.get(frame, face)
        faces.append(face)
    return faces


def _embed(frame, face):
    if face.get("embedding") is None:
        _recognizer.get(frame, face)
    return face.normed_embedding


# ---------------------------------------------------------------------------
# Single entry point
# ---------------------------------------------------------------------------
def verify_candidate(profile, live, mode="registration", *,
                     challenge_nonce=None, session_id=None, timestamps=None):
    """
    mode="registration": `profile` = path to the trusted profile photo,
        `live` = path to the selfie video. Active liveness + face match (+ PAD).
    mode="exam": `live` = list of BGR frames forming one short window for the
        exam session `session_id` (the session already holds the reference,
        `profile` is ignored). Passive PAD + tracking + risk engine.
    """
    if mode == "registration":
        return verify_registration(profile, live, challenge_nonce)
    if mode == "exam":
        return process_exam_window(session_id, live, timestamps)
    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def verify_registration(profile_photo_path, video_path, challenge_nonce=None):
    started = time.perf_counter()

    # 1. Profile photo -------------------------------------------------------
    img = cv2.imread(profile_photo_path)
    if img is None:
        return _result("PROFILE_UNREADABLE")
    profile_faces = analyze_frame(resize(img), landmarks=False)
    if not profile_faces:
        return _result("NO_FACE_IN_PROFILE")
    profile = face_verification.largest(profile_faces)

    # 2. Challenge (single use: consumed even if verification fails) ---------
    challenge = None
    if challenge_nonce:
        challenge, error = registration_liveness.consume_challenge(challenge_nonce)
        if error:
            return _result(error)
    elif REQUIRE_CHALLENGE:
        return _result("CHALLENGE_MISSING")

    # 3. Video frames --------------------------------------------------------
    frames, times, video_info = read_video(
        video_path, CHALLENGE_TARGET_FRAMES if challenge else TARGET_FRAMES)
    if not frames:
        return _result("VIDEO_UNREADABLE", video=video_info)
    dur = video_info.get("duration_sec")
    if len(frames) < MIN_FRAMES or (dur is not None and dur < MIN_DURATION_SEC):
        return _result("VIDEO_TOO_SHORT", video={**video_info, "frames_sampled": len(frames)})

    # 4. Faces per frame, presence, face count -------------------------------
    faces_per_frame = [analyze_frame(f, embedding=False) for f in frames]
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

    # 5. Embeddings + same person throughout ---------------------------------
    with_face = [i for i, f in enumerate(primary) if f is not None]
    if challenge and len(with_face) > CHALLENGE_EMBED_FRAMES:
        pick = np.linspace(0, len(with_face) - 1, CHALLENGE_EMBED_FRAMES).astype(int)
        with_face = [with_face[j] for j in pick]
    emb = np.stack([_embed(frames[i], primary[i]) for i in with_face])
    consistency = face_tracking.consistency(emb)
    video_info["min_consistency"] = round(float(consistency.min()), 3)

    # 6. Active liveness -----------------------------------------------------
    liveness, movement_failed = registration_liveness.movement_check(video_faces)
    challenge_block = None
    if challenge:
        challenge_block = registration_liveness.evaluate_challenge(
            challenge, [registration_liveness.measure(f) for f in primary], times)
        liveness_failed = [] if challenge_block["passed"] else ["CHALLENGE_FAILED"]
    else:
        liveness_failed = movement_failed

    # 7. Passive PAD ---------------------------------------------------------
    pad = anti_spoof.detect_spoof(frames, primary)

    # 8. Identity ------------------------------------------------------------
    identity, matched = face_verification.compare(emb, profile)

    # Collect every failed check, report the most important first.
    failed = []
    if consistency.min() < face_tracking.SAME_PERSON_THRESHOLD:
        failed.append("DIFFERENT_PEOPLE_IN_VIDEO")
    if PAD_ENFORCE_REGISTRATION and pad["status"] == "HIGH_RISK":
        failed.append("SPOOF_SUSPECTED")
    if PAD_ENFORCE_REGISTRATION and PAD_REQUIRE_QUALITY and pad["status"] == "INSUFFICIENT_QUALITY":
        failed.append("VIDEO_QUALITY_TOO_LOW")
    failed += liveness_failed
    if not matched:
        failed.append("FACE_MISMATCH")

    result = _result(
        failed[0] if failed else "OK",
        failed_checks=failed,
        live=not liveness_failed and "SPOOF_SUSPECTED" not in failed,
        identity_match=matched,
        similarity=identity["similarity"],
        identity=identity,
        liveness=liveness,
        video=video_info,
        anti_spoof=pad,
    )
    if challenge_block:
        result["challenge"] = challenge_block

    log.info(json.dumps({
        "event": "registration_verify",
        "reason_code": result["reason_code"],
        "failed": failed,
        "similarity": identity["similarity"],
        "pad_status": pad["status"],
        "spoof_score": pad["spoof_score"],
        "challenge": None if not challenge_block else {
            "steps": challenge_block["steps"], "failure": challenge_block["failure"]},
        "processing_ms": round((time.perf_counter() - started) * 1000),
    }))
    return result


# ---------------------------------------------------------------------------
# Exam monitoring
# ---------------------------------------------------------------------------
@dataclass
class ExamSession:
    session_id: str
    email: str
    exam_id: int
    tracker: face_tracking.ExamTracker
    risk: risk_engine.RiskState = field(default_factory=risk_engine.RiskState)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    windows: int = 0
    events: list = field(default_factory=list)      # state changes + flagged windows
    last_assessment: dict | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


# In memory, single process. With several API workers use a shared store
# (Redis) or route a session's windows to the same worker.
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
    """`profile_image`: BGR image of the trusted profile photo."""
    if profile_image is None:
        return {"started": False, "reason_code": "PROFILE_UNREADABLE",
                "reason": REASONS["PROFILE_UNREADABLE"]}
    faces = analyze_frame(resize(profile_image), landmarks=False)
    if not faces:
        return {"started": False, "reason_code": "NO_FACE_IN_PROFILE",
                "reason": REASONS["NO_FACE_IN_PROFILE"]}
    reference = face_verification.largest(faces).normed_embedding

    session = ExamSession(
        session_id=secrets.token_urlsafe(16), email=email, exam_id=exam_id,
        tracker=face_tracking.ExamTracker(reference))
    _get_session(None)  # purge expired sessions
    with _sessions_lock:
        _sessions[session.session_id] = session
    log.info(json.dumps({"event": "exam_session_start", "session_id": session.session_id,
                         "email": email, "exam_id": exam_id}))
    return {
        "started": True,
        "session_id": session.session_id,
        "window_frames": EXAM_WINDOW_FRAMES,
        "frame_interval_ms": EXAM_FRAME_INTERVAL_MS,
        "next_window_sec": risk_engine.NEXT_WINDOW_SEC["NORMAL"],
    }


def _identity_face(frames, primary):
    """Best frame/face of the window for an identity sample: good quality,
    roughly frontal, largest x most confident."""
    best, best_key = None, -1.0
    for frame, face in zip(frames, primary):
        if face is None:
            continue
        ok, _, _ = anti_spoof.face_quality(frame, face)
        _, yaw = registration_liveness.pose(face)
        if not ok or abs(yaw) > MAX_IDENTITY_YAW_DEG:
            continue
        key = float(face.det_score) * float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        if key > best_key:
            best, best_key = (frame, face), key
    return best


def process_exam_window(session_id, frames, timestamps=None):
    """Analyse one short window of exam frames (BGR). Never blocks the exam:
    the result is a risk assessment, not a pass/fail."""
    started = time.perf_counter()
    session = _get_session(session_id)
    if session is None:
        return {"ok": False, "reason_code": "SESSION_NOT_FOUND",
                "reason": "Unknown or expired exam session"}
    frames = [resize(f) for f in frames if f is not None][:EXAM_MAX_FRAMES]
    if not frames:
        return {"ok": False, "reason_code": "NO_FRAMES", "reason": "No decodable frames in window"}

    with session.lock:  # windows of one session are processed in order
        faces_per_frame = [analyze_frame(f, landmarks=False, embedding=False, det_size=EXAM_DET_SIZE)
                           for f in frames]
        summary = face_tracking.summarize(faces_per_frame,
                                          min_rel_area=face_tracking.EXAM_MIN_REL_FACE_AREA)
        tracker = session.tracker
        face_count = tracker.face_count(summary)

        embedding = None
        wants_identity = (session.windows % IDENTITY_EVERY_N_WINDOWS == 0
                          or tracker.needs_identity_check()
                          or session.risk.state != "NORMAL")
        if face_count != "NONE" and wants_identity:
            picked = _identity_face(frames, summary["primary"])
            if picked:
                embedding = _embed(*picked)

        facts = tracker.update(face_count, embedding)
        if face_count == "NONE":
            pad = {"status": "NOT_RUN", "spoof_score": None, "confidence": None}
        else:
            pad = anti_spoof.detect_spoof(frames, summary["primary"])

        signals = {
            "face_count": face_count,
            "identity": facts["identity"],
            "continuity": facts["continuity"],
            "pad": pad["status"],
            "integrity": "NOT_CHECKED",
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
                        ("status", "spoof_score", "confidence", "attack_type",
                         "frames_scored", "frames_skipped")},
                "integrity": "NOT_CHECKED",
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
        if assessment["state_changed"] or assessment["state"] != "NORMAL":
            session.events.append({
                "time": time.time(), "window": session.windows, "state": assessment["state"],
                "reasons": assessment["reasons"], "signals": signals,
                "spoof_score": pad.get("spoof_score"),
                "reference_similarity": facts["reference_similarity"],
            })
            del session.events[:-MAX_EVENTS]

    log.info(json.dumps({
        "event": "exam_window", "session_id": session_id, "window": session.windows,
        "state": assessment["state"], "reasons": assessment["reasons"], "signals": signals,
        "spoof_score": pad.get("spoof_score"), "reference_similarity": facts["reference_similarity"],
        "processing_ms": response["processing_ms"],
    }))
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
        log.info(json.dumps({"event": "exam_session_end", "session_id": session_id,
                             "windows": status["windows"], "highest_state": status["highest_state"]}))
    return status


def _highest_state(session):
    order = ["NORMAL", "SUSPICIOUS", "HIGH_RISK"]
    states = [e["state"] for e in session.events] + [session.risk.state]
    return max(states, key=order.index)
