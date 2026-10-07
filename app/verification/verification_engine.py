# app/verification/verification_engine.py
"""Consolidated face verification components.
This module combines the face, tracking, identity, liveness, and anti-spoof
implementation without changing their algorithms or thresholds.
"""


# --- faces.py ---

# app/verification/faces.py
"""
Turns uploaded images/videos into detected faces: decoding, resizing,
InsightFace detection / landmarks / embeddings, and the face measurements
several modules share (quality, head pose).
"""

import functools

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

# Stored with every saved embedding: embeddings from different models
# cannot be compared, so a model change must invalidate old ones.
EMBEDDING_MODEL = "insightface/buffalo_l/w600k_r50"

MIN_DET_SCORE = 0.50            # ignore weak detections
MAX_SIDE = 960                  # downscale big frames for speed

# Frames below these limits are not used for PAD or identity samples.
MIN_FACE_PX = 60                # face box width in pixels
MIN_SHARPNESS = 15.0            # variance of Laplacian on the 112x112 face crop
MIN_BRIGHTNESS = 40.0           # mean grey level of the face crop
MAX_BRIGHTNESS = 225.0


@functools.cache
def load_models():
    """Load buffalo_l once and reuse it for every request. Only the three
    sub-models this project uses are loaded (1k3d68 provides pose and the
    68 landmarks)."""
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "landmark_3d_68", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app.det_model, app.models["landmark_3d_68"], app.models["recognition"]


def resize(img):
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def read_video(video_path, target):
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
    """JPEG/PNG bytes -> BGR frames. Undecodable images are dropped."""
    frames = []
    for data in images:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if img is not None:
            frames.append(resize(img))
    return frames


def analyze_frame(frame, landmarks=True, embedding=True, det_size=None):
    """Faces with det_score >= MIN_DET_SCORE. Landmarks (+ pose) and the
    embedding are optional because they cost far more than detection."""
    detector, landmark_model, recognizer = load_models()
    bboxes, kpss = detector.detect(frame, input_size=det_size)
    faces = []
    for i in range(bboxes.shape[0]):
        face = Face(bbox=bboxes[i, 0:4], kps=None if kpss is None else kpss[i],
                    det_score=bboxes[i, 4])
        if face.det_score < MIN_DET_SCORE:
            continue
        if landmarks:
            landmark_model.get(frame, face)
        if embedding:
            recognizer.get(frame, face)
        faces.append(face)
    return faces


def embed(frame, face):
    """Normed identity embedding, computed only if not done already."""
    if face.get("embedding") is None:
        load_models()[2].get(frame, face)
    return face.normed_embedding


def face_quality(frame, face):
    """Returns (ok, reason, metrics). A bad frame is skipped, never treated
    as evidence of an attack."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    width = x2 - x1
    if width < MIN_FACE_PX or y2 - y1 < MIN_FACE_PX:
        return False, "FACE_TOO_SMALL", {"face_px": int(max(width, 0))}
    gray = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (112, 112)), cv2.COLOR_BGR2GRAY)
    metrics = {
        "face_px": int(width),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "brightness": round(float(gray.mean()), 1),
    }
    if metrics["sharpness"] < MIN_SHARPNESS:
        return False, "BLURRY", metrics
    if not MIN_BRIGHTNESS <= metrics["brightness"] <= MAX_BRIGHTNESS:
        return False, "BAD_LIGHTING", metrics
    return True, None, metrics


def pose(face):
    """(pitch, yaw) in degrees. Falls back to a rough 5-point estimate when
    the 3D landmarks were not computed (exam frames)."""
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


# --- face_verification.py ---

# app/verification/face_verification.py
"""
Identity only: does the live face belong to the registered candidate?
A perfect match says nothing about whether the face is physically present.
Embeddings are L2-normalised, so cosine similarity is a plain dot product.
"""

import numpy as np


# Log the returned similarities on real users and adjust.
SIMILARITY_THRESHOLD = 0.45     # profile photo vs video face (cosine)
MIN_MATCH_RATIO = 0.6           # fraction of video frames that must match
MAX_IDENTITY_YAW_DEG = 35.0     # profile views give unreliable embeddings


def largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def pick_identity_face(frames, faces):
    """The (frame, face) in a window best suited for an identity sample:
    good quality, roughly frontal, then largest and most confident.
    `faces` holds the main face per frame (None = no face). Returns None if
    no frame qualifies."""
    best, best_key = None, -1.0
    for frame, face in zip(frames, faces):
        if face is None:
            continue
        ok, _, _ = face_quality(frame, face)
        _, yaw = pose(face)
        if not ok or abs(yaw) > MAX_IDENTITY_YAW_DEG:
            continue
        area = float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        key = float(face.det_score) * area
        if key > best_key:
            best, best_key = (frame, face), key
    return best


def build_template(embeddings):
    """One reference embedding from several frames: the normalised mean."""
    mean = embeddings.mean(axis=0)
    return mean / np.linalg.norm(mean)


def export_template(template):
    """JSON-safe form returned at registration and stored by the caller."""
    return {"model": EMBEDDING_MODEL, "vector": [round(float(x), 6) for x in template]}


def import_template(data):
    """Reverse of export_template. Returns (template, None) or (None, reason_code)."""
    if not isinstance(data, dict) or not isinstance(data.get("vector"), list):
        return None, "INVALID_REFERENCE_EMBEDDING"
    if data.get("model") != EMBEDDING_MODEL:
        return None, "REFERENCE_MODEL_MISMATCH"
    try:
        vector = np.asarray(data["vector"], dtype=np.float64)
    except (TypeError, ValueError):
        return None, "INVALID_REFERENCE_EMBEDDING"
    norm = np.linalg.norm(vector) if vector.shape == (512,) else 0.0
    if not np.isfinite(norm) or norm == 0:
        return None, "INVALID_REFERENCE_EMBEDDING"
    return vector / norm, None


def similarity(embedding, reference_embedding):
    """Cosine similarity of two normed embeddings."""
    return float(np.dot(embedding, reference_embedding))


def compare(embeddings, reference_embedding):
    """Compare a stack of live embeddings (n, 512) with one reference embedding.

    Returns (identity, matched): `identity` is the response block (rounded
    values), `matched` is decided on the unrounded values.
    """
    sims = embeddings @ reference_embedding
    median = float(np.median(sims))
    match_ratio = float(np.mean(sims >= SIMILARITY_THRESHOLD))
    identity = {
        "similarity": round(median, 4),
        "best_similarity": round(float(sims.max()), 4),
        "match_ratio": round(match_ratio, 3),
        "threshold": SIMILARITY_THRESHOLD,
    }
    matched = median >= SIMILARITY_THRESHOLD and match_ratio >= MIN_MATCH_RATIO
    return identity, matched


# --- face_tracking.py ---

# app/verification/face_tracking.py
"""
Face presence, face count and same-person continuity, for one registration
video (summarize, check_presence, consistency) and across exam windows
(ExamTracker).
"""

    SIMILARITY_THRESHOLD, build_template, largest, similarity)

SAME_PERSON_THRESHOLD = 0.50    # every video face vs the video's mean face
MIN_FACE_FRAME_RATIO = 0.6      # face must be visible in 60% of frames
MAX_MULTI_FACE_RATIO = 0.2      # >20% frames with 2+ faces -> reject
MIN_FACE_FRAMES = 5

# Exam - placeholders, calibrate on real exam data
EXAM_MIN_FACE_RATIO = 0.3       # below this a window counts as "no face"
EXAM_MULTI_FACE_RATIO = 0.3     # at/above this a window counts as "multiple faces"
EXAM_MIN_REL_FACE_AREA = 0.04   # ignore extra faces smaller than 4% of the main face
                                # (posters, photos far in the background)


def summarize(faces_per_frame, min_rel_area=0.0):
    """Per-window face statistics. `faces_per_frame` is a list of face lists.

    Returns a dict with the response fields plus `primary`: the largest face
    of every frame (None when the frame has no face).
    """
    primary, multi = [], 0
    for faces in faces_per_frame:
        if not faces:
            primary.append(None)
            continue
        main = largest(faces)
        primary.append(main)
        if min_rel_area > 0:
            main_area = _area(main)
            faces = [f for f in faces if f is main or _area(f) >= min_rel_area * main_area]
        if len(faces) > 1:
            multi += 1

    n = len(faces_per_frame)
    with_face = sum(f is not None for f in primary)
    return {
        "frames_sampled": n,
        "frames_with_face": with_face,
        "face_ratio": with_face / n if n else 0.0,
        "multi_face_ratio": multi / n if n else 0.0,
        "primary": primary,
    }


def check_presence(summary):
    """Registration gate. Returns a reason code or None."""
    if summary["multi_face_ratio"] > MAX_MULTI_FACE_RATIO:
        return "MULTIPLE_FACES_IN_VIDEO"
    if (summary["face_ratio"] < MIN_FACE_FRAME_RATIO
            or summary["frames_with_face"] < MIN_FACE_FRAMES):
        return "FACE_NOT_CONSISTENT"
    return None


def consistency(embeddings):
    """Similarity of every embedding to the mean embedding (same person check)."""
    return embeddings @ build_template(embeddings)


def _area(face):
    return float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


class ExamTracker:
    """Per exam session: turns each window into face_count (NONE | ONE |
    MULTIPLE), identity vs the reference and continuity vs the last window."""

    def __init__(self, reference_embedding):
        self.reference = reference_embedding
        self.last_embedding = None      # embedding of the last identity sample
        self.absent_windows = 0         # consecutive windows without a face

    def face_count(self, summary):
        if summary["face_ratio"] < EXAM_MIN_FACE_RATIO:
            return "NONE"
        if summary["multi_face_ratio"] >= EXAM_MULTI_FACE_RATIO:
            return "MULTIPLE"
        return "ONE"

    def update(self, face_count, embedding):
        """Feed one window. `embedding` is the identity sample taken in this
        window (or None if none was taken). Returns the window facts."""
        returned = face_count != "NONE" and self.absent_windows > 0
        self.absent_windows = self.absent_windows + 1 if face_count == "NONE" else 0

        identity, continuity, ref_sim, prev_sim = "UNKNOWN", "UNKNOWN", None, None
        if embedding is not None:
            ref_sim = similarity(embedding, self.reference)
            identity = "MATCH" if ref_sim >= SIMILARITY_THRESHOLD else "MISMATCH"
            if self.last_embedding is not None:
                prev_sim = similarity(embedding, self.last_embedding)
                continuity = "STABLE" if prev_sim >= SAME_PERSON_THRESHOLD else "CHANGED"
            self.last_embedding = embedding

        return {
            "face_count": face_count,
            "identity": identity,
            "continuity": continuity,
            "reference_similarity": None if ref_sim is None else round(ref_sim, 4),
            "previous_similarity": None if prev_sim is None else round(prev_sim, 4),
            "face_returned": returned,
            "absent_windows": self.absent_windows,
        }


# --- registration_liveness.py ---

# app/verification/registration_liveness.py
"""
Active liveness for registration only - never used during the exam.

The challenge (random ordered actions + single-use nonce) defeats generic
recordings of the candidate. It does not stop someone who pre-records every
possible sequence or uses a real-time face swap through a virtual camera.
The movement check is the weaker fallback used when no challenge is sent.
"""

import os
import secrets
import threading
import time

import numpy as np


# Movement check (no challenge)
# A still photo moves the pose estimate by less than 1 degree, so small,
# natural turns are enough; candidates cannot judge angles.
MIN_YAW_RANGE_DEG = 5.0         # head turn left/right
MIN_PITCH_RANGE_DEG = 4.0       # or nod up/down
REQUIRE_FACIAL_MOVEMENT = True
MIN_EAR_CHANGE = 0.25           # relative eye-opening change (blink)
MIN_MAR_CHANGE = 0.35           # relative mouth-opening change

# Challenge - thresholds are placeholders, tune them with the webcam demo
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

TURN_DEG = 10.0                 # yaw change from the clip's median to count as a turn
BLINK_EAR_DROP = 0.30           # eye aspect ratio drops 30% below the clip's median
MOUTH_OPEN_DELTA = 0.25         # mouth aspect ratio rises this much above the median
MAX_YAW_FOR_EYES_MOUTH = 20.0   # eye/mouth ratios are unreliable on turned faces

# InsightFace yaw is positive when the nose points to the image's right, which
# in a normal (non-mirrored) camera frame is the candidate turning to THEIR
# left. Set VERIFICATION_MIRRORED_INPUT=1 if the client records a mirrored
# (selfie-preview) stream.
MIRRORED_INPUT = os.getenv("VERIFICATION_MIRRORED_INPUT", "0") == "1"


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


def movement_check(faces):
    """Did the head turn and the eyes/mouth change at all? `faces` excludes
    frames without a face. Returns (liveness_block, failed_codes)."""
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


# In process memory: needs a shared store (e.g. Redis) with several API workers.
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


# --- anti_spoof.py ---

# app/verification/anti_spoof.py
"""
Passive presentation-attack detection (PAD): the candidate does nothing.

detect_spoof() scores how much the frames look like a face re-captured from
a printed photo or a phone/tablet/monitor screen. It does not prove that a
person is physically present, and it cannot see injected video (virtual
cameras, digital deepfakes): those frames were never re-captured, so they
carry no presentation artefacts. That needs separate stream-integrity checks.

To use a different PAD model, replace load_model() and live_probabilities();
detect_spoof() and its callers stay the same.
"""

import functools
import hashlib
import os
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from skimage.transform import SimilarityTransform


# Placeholders, not calibrated: calibrate with tools/evaluate_pad.py on
# your own bona fide and attack recordings.
SPOOF_ELEVATED = 0.20           # the model's own default operating point (live >= 0.8)
SPOOF_HIGH = 0.50
MIN_SCORED_FRAMES = 3           # fewer usable frames -> INSUFFICIENT_QUALITY

# InsightFace RGB liveness model: 80x80 input, aligned on the 5 face points.
# InsightFace states its pretrained weights are for non-commercial research
# use only - see the licensing section of the README.
MODEL_URL = "https://github.com/deepinsight/insightface-model-addons/releases/download/addons/liveness.onnx"
MODEL_SHA256 = "87a9ac1dbb16a61eec212957e5095e62a8769c1e188af9b0198f253302c4afdb"
MODEL_PATH = Path(os.getenv("PAD_MODEL_PATH", "~/.insightface/addons/liveness.onnx")).expanduser()
INPUT_SIZE = 80
MAX_OUT_OF_BOUNDS = 0.30        # skip faces whose aligned crop is mostly outside the frame
ALIGN_TO = np.array([[32.03, 38.06], [47.89, 37.98], [40.01, 47.08],
                     [33.50, 56.36], [46.63, 56.29]], dtype=np.float32)


@functools.cache
def load_model():
    """Download on first use, verify the checksum, and keep one session."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".download")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, MODEL_PATH)
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError(f"PAD model checksum mismatch: {MODEL_PATH}")
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def _aligned_crop(frame, kps):
    """80x80 crop aligned like the model's training data, or None when too
    much of it would fall outside the frame."""
    src = np.asarray(kps, dtype=np.float32)
    # skimage >= 0.26 deprecates estimate() in favour of from_estimate()
    if hasattr(SimilarityTransform, "from_estimate"):
        tform = SimilarityTransform.from_estimate(src, ALIGN_TO)
        if not tform:
            return None
    else:
        tform = SimilarityTransform()
        if not tform.estimate(src, ALIGN_TO):
            return None
    m = tform.params[:2, :]
    inside = cv2.warpAffine(np.ones(frame.shape[:2], np.float32), m, (INPUT_SIZE, INPUT_SIZE))
    if 1.0 - float(inside.mean()) > MAX_OUT_OF_BOUNDS:
        return None
    return cv2.warpAffine(frame, m, (INPUT_SIZE, INPUT_SIZE), borderMode=cv2.BORDER_REPLICATE)


def live_probabilities(frames, faces):
    """Model output per (frame, face): probability the face is not a
    presentation attack, or None where the face could not be cropped."""
    crops = [_aligned_crop(frame, face.kps) for frame, face in zip(frames, faces)]
    usable = [i for i, crop in enumerate(crops) if crop is not None]
    probabilities = [None] * len(crops)
    if usable:
        # BGR -> RGB, HWC -> CHW, 0..1: the model's training format
        batch = np.stack([crops[i][:, :, ::-1].transpose(2, 0, 1) for i in usable])
        session = load_model()
        output = session.run(None, {session.get_inputs()[0].name: batch.astype(np.float32) / 255.0})[0]
        for i, p in zip(usable, np.asarray(output).reshape(-1)):
            probabilities[i] = float(p)
    return probabilities


def detect_spoof(frames, faces, min_scored_frames=MIN_SCORED_FRAMES):
    """`faces` holds the main face of each frame (None when there is none).

    Returns status (LOW_RISK | ELEVATED | HIGH_RISK | INSUFFICIENT_QUALITY),
    spoof_score (median over scored frames: 0 looks genuine, 1 looks like an
    attack), and how many frames were scored or skipped and why. Frames that
    are too small, blurry or badly lit are skipped, never counted as attacks.
    """
    skipped = Counter()
    usable_frames, usable_faces = [], []
    for frame, face in zip(frames, faces):
        if face is None:
            skipped["NO_FACE"] += 1
            continue
        ok, reason, _ = face_quality(frame, face)
        if not ok:
            skipped[reason] += 1
            continue
        usable_frames.append(frame)
        usable_faces.append(face)

    probabilities = live_probabilities(usable_frames, usable_faces) if usable_frames else []
    scores = [1.0 - p for p in probabilities if p is not None]
    if len(scores) < len(probabilities):
        skipped["FACE_AT_EDGE"] += len(probabilities) - len(scores)

    result = {
        "status": "INSUFFICIENT_QUALITY",
        "spoof_score": None,
        "frames_scored": len(scores),
        "frames_skipped": dict(skipped),
        "frame_scores": [round(s, 4) for s in scores],
    }
    if len(scores) < min_scored_frames:
        return result

    spoof_score = float(np.median(scores))
    if spoof_score >= SPOOF_HIGH:
        result["status"] = "HIGH_RISK"
    elif spoof_score >= SPOOF_ELEVATED:
        result["status"] = "ELEVATED"
    else:
        result["status"] = "LOW_RISK"
    result["spoof_score"] = round(spoof_score, 4)
    return result
