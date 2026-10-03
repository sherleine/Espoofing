# app/verification/registration_liveness.py
"""
ACTIVE liveness - registration / initial verification only.

Never used during the exam: the candidate must not be asked to act there.

Two modes:

1. Challenge (preferred). The server issues a random ordered sequence of
   actions with a single-use nonce, e.g. ["turn_left", "blink", "turn_right"].
   The video must show THOSE actions IN THAT ORDER. A generic prerecorded clip
   of the candidate moving does not pass, and head turns that were not asked
   for fail the check.

2. Legacy movement check (fallback while the frontend has no challenge UI):
   "did the head turn and did the eyes/mouth change at all". Kept exactly as
   in the original service.py. Weak: a prerecorded clip of the candidate passes.

Limitations (challenge mode): someone who pre-records the candidate doing all
24 possible sequences, or who drives a real-time face swap through a virtual
camera, can still pass. That is what passive PAD (anti_spoof.py) and stream
integrity checks are for.
"""

import os
import secrets
import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# Legacy movement thresholds (unchanged from the original service.py)
# ---------------------------------------------------------------------------
MIN_YAW_RANGE_DEG = 10.0        # head turn left/right
MIN_PITCH_RANGE_DEG = 8.0       # or nod up/down
REQUIRE_FACIAL_MOVEMENT = True
MIN_EAR_CHANGE = 0.25           # relative eye-opening change (blink)
MIN_MAR_CHANGE = 0.35           # relative mouth-opening change

# ---------------------------------------------------------------------------
# Challenge settings (placeholders: tune with the webcam demo)
# ---------------------------------------------------------------------------
ACTIONS = {
    "turn_left": "Turn your head to your left, then back to the centre",
    "turn_right": "Turn your head to your right, then back to the centre",
    "blink": "Close your eyes for a moment, then open them",
    "open_mouth": "Open your mouth wide, then close it",
}
TURN_ACTIONS = ("turn_left", "turn_right")
CHALLENGE_STEPS = 3             # 3 distinct actions out of 4 -> always >= 1 head turn
CHALLENGE_TTL_SEC = 180         # nonce must be used within this time
STEP_SECONDS = 2.5              # suggested time per prompt in the UI

TURN_DEG = 15.0                 # yaw change from the clip's median to count as a turn
BLINK_EAR_DROP = 0.30           # eye aspect ratio drops 30% below the clip's median
MOUTH_OPEN_DELTA = 0.25         # mouth aspect ratio rises this much above the median
MAX_YAW_FOR_EYES_MOUTH = 20.0   # eye/mouth ratios are unreliable on turned faces

# InsightFace yaw is positive when the nose points to the image's right, which
# in a normal (non-mirrored) camera frame is the candidate turning to THEIR
# left. Set VERIFICATION_MIRRORED_INPUT=1 if the client records a mirrored
# (selfie-preview) stream.
MIRRORED_INPUT = os.getenv("VERIFICATION_MIRRORED_INPUT", "0") == "1"


# ---------------------------------------------------------------------------
# Per-face measurements
# ---------------------------------------------------------------------------
def pose(face):
    """(pitch, yaw) in degrees. Falls back to a rough 5-point estimate."""
    p = getattr(face, "pose", None)
    if p is not None:
        return float(p[0]), float(p[1])
    k = face.kps  # left_eye, right_eye, nose, mouth_l, mouth_r
    eye_mid = (k[0] + k[1]) / 2
    eye_dist = max(np.linalg.norm(k[1] - k[0]), 1.0)
    mouth_mid = (k[3] + k[4]) / 2
    yaw = (k[2][0] - eye_mid[0]) / eye_dist * 90
    pitch = ((k[2][1] - eye_mid[1]) / max(mouth_mid[1] - eye_mid[1], 1.0) - 0.5) * 90
    return float(pitch), float(yaw)


def eye_mouth_ratios(face):
    """Eye aspect ratio + mouth aspect ratio from 68-point landmarks."""
    lm = getattr(face, "landmark_3d_68", None)
    if lm is None:
        return None, None
    p = lm[:, :2]
    d = lambda a, b: np.linalg.norm(p[a] - p[b])

    def ear(i):  # i = first index of the 6 eye points
        return (d(i + 1, i + 5) + d(i + 2, i + 4)) / (2 * max(d(i, i + 3), 1e-6))

    eye = (ear(36) + ear(42)) / 2
    mouth = (d(61, 67) + d(62, 66) + d(63, 65)) / (2 * max(d(60, 64), 1e-6))
    return float(eye), float(mouth)


def _relative_change(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) < 3:
        return 0.0
    med = np.median(v)
    return float((v.max() - v.min()) / max(med, 1e-3))


# ---------------------------------------------------------------------------
# Legacy movement check
# ---------------------------------------------------------------------------
def movement_check(faces):
    """Original check on the faces of the video (frames without a face removed).

    Returns (liveness_block, failed_codes).
    """
    poses = np.array([pose(f) for f in faces])
    pitch_range = float(np.ptp(poses[:, 0]))
    yaw_range = float(np.ptp(poses[:, 1]))
    ratios = [eye_mouth_ratios(f) for f in faces]
    ear_change = _relative_change([r[0] for r in ratios])
    mar_change = _relative_change([r[1] for r in ratios])

    head_moved = yaw_range >= MIN_YAW_RANGE_DEG or pitch_range >= MIN_PITCH_RANGE_DEG
    face_moved = ear_change >= MIN_EAR_CHANGE or mar_change >= MIN_MAR_CHANGE

    liveness = {
        "yaw_range_deg": round(yaw_range, 2),
        "pitch_range_deg": round(pitch_range, 2),
        "eye_change": round(ear_change, 3),
        "mouth_change": round(mar_change, 3),
        "head_movement": head_moved,
        "facial_movement": face_moved,
    }
    failed = []
    if not head_moved:
        failed.append("NO_HEAD_MOVEMENT")
    if REQUIRE_FACIAL_MOVEMENT and not face_moved:
        failed.append("NO_FACIAL_MOVEMENT")
    return liveness, failed


# ---------------------------------------------------------------------------
# Challenge store (in memory, single process).
# With several API workers use a shared store (e.g. Redis with TTL) instead.
# ---------------------------------------------------------------------------
_challenges = {}
_lock = threading.Lock()


def issue_challenge():
    now = time.time()
    steps = secrets.SystemRandom().sample(list(ACTIONS), CHALLENGE_STEPS)
    challenge = {
        "nonce": secrets.token_urlsafe(16),
        "steps": steps,
        "issued_at": now,
        "expires_at": now + CHALLENGE_TTL_SEC,
    }
    with _lock:
        for nonce in [n for n, c in _challenges.items() if c["expires_at"] < now]:
            del _challenges[nonce]
        _challenges[challenge["nonce"]] = challenge
    return {
        **challenge,
        "prompts": [ACTIONS[s] for s in steps],
        "step_seconds": STEP_SECONDS,
    }


def consume_challenge(nonce):
    """Single use: the nonce is removed whether verification passes or not.
    Returns (challenge, None) or (None, "CHALLENGE_INVALID")."""
    with _lock:
        challenge = _challenges.pop(nonce, None)
    if challenge is None or time.time() > challenge["expires_at"]:
        return None, "CHALLENGE_INVALID"
    return challenge, None


# ---------------------------------------------------------------------------
# Challenge evaluation
# ---------------------------------------------------------------------------
def measure(face):
    if face is None:
        return None
    pitch, yaw = pose(face)
    eye, mouth = eye_mouth_ratios(face)
    return {"pitch": pitch, "yaw": yaw, "eye": eye, "mouth": mouth}


def _median(values):
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def label_frames(measurements):
    """Which actions are visible in each frame, relative to the clip's median
    (neutral) pose. Returns (labels, baseline)."""
    valid = [m for m in measurements if m is not None]
    base = {
        "yaw": _median([m["yaw"] for m in valid]),
        "eye": _median([m["eye"] for m in valid]),
        "mouth": _median([m["mouth"] for m in valid]),
    }
    sign = -1.0 if MIRRORED_INPUT else 1.0
    labels = []
    for m in measurements:
        actions = set()
        if m is not None and base["yaw"] is not None:
            rel_yaw = sign * (m["yaw"] - base["yaw"])
            if rel_yaw >= TURN_DEG:
                actions.add("turn_left")
            elif rel_yaw <= -TURN_DEG:
                actions.add("turn_right")
            if abs(rel_yaw) < MAX_YAW_FOR_EYES_MOUTH:
                if m["eye"] is not None and base["eye"] and m["eye"] <= base["eye"] * (1 - BLINK_EAR_DROP):
                    actions.add("blink")
                if m["mouth"] is not None and base["mouth"] is not None and m["mouth"] - base["mouth"] >= MOUTH_OPEN_DELTA:
                    actions.add("open_mouth")
        labels.append(actions)
    return labels, base


def detect_events(labels):
    """Onsets of each action: [(frame_index, action), ...] in time order."""
    events, previous = [], set()
    for i, actions in enumerate(labels):
        for action in sorted(actions - previous):
            events.append((i, action))
        previous = actions
    return events


def evaluate_challenge(challenge, measurements, timestamps=None):
    """Did the video show the requested actions in the requested order?

    `measurements` is one `measure(face)` per sampled frame (None = no face),
    `timestamps` the matching times in seconds (optional, for reporting).
    Returns the `challenge` block of the API response; `passed` is the verdict.
    """
    steps = challenge["steps"]
    labels, base = label_frames(measurements)
    events = detect_events(labels)

    def at(i):
        if timestamps is None or i >= len(timestamps) or timestamps[i] is None:
            return None
        return round(float(timestamps[i]), 2)

    failure, matched, position = None, [], -1
    for step in steps:
        hit = next((i for i, a in events if a == step and i > position), None)
        if hit is None:
            seen = any(a == step for _, a in events)
            failure = f"WRONG_ORDER:{step}" if seen else f"NOT_DETECTED:{step}"
            break
        matched.append({"action": step, "frame": hit, "at_sec": at(hit)})
        position = hit

    # Head turns are never spontaneous: the turns in the video must be exactly
    # the requested ones, in order. Defeats "do every movement" recordings.
    if failure is None:
        seen_turns = []
        for _, a in events:
            if a in TURN_ACTIONS and (not seen_turns or seen_turns[-1] != a):
                seen_turns.append(a)
        wanted_turns = [s for s in steps if s in TURN_ACTIONS]
        if seen_turns != wanted_turns:
            failure = "UNREQUESTED_HEAD_TURN"

    return {
        "nonce": challenge["nonce"],
        "steps": steps,
        "passed": failure is None,
        "failure": failure,
        "matched": matched,
        "detected": [{"action": a, "frame": i, "at_sec": at(i)} for i, a in events],
        "baseline": {k: None if v is None else round(v, 3) for k, v in base.items()},
    }
