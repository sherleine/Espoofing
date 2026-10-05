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
