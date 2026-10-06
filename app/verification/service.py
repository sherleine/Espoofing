# app/verification/service.py
"""
Face verification from a short selfie video.

Pipeline:
  1. Read the profile photo and find one face.
  2. Decode the video and sample frames evenly across it.
  3. Detect the face in each frame (exactly one person must be visible).
  4. Check the video shows ONE person throughout (no swap mid-video).
  5. Liveness: head must turn (yaw/pitch change) and the face must
     change shape (blink / mouth movement), not just slide around.
  6. Identity: compare the profile face to every video face.

Every result carries `reason_code` + `reason` + `failed_checks`
so the API can tell the user exactly why verification failed.
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis

_face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)
_face_app.prepare(ctx_id=0, det_size=(640, 640))

SIMILARITY_THRESHOLD = 0.45
MIN_MATCH_RATIO = 0.6
SAME_PERSON_THRESHOLD = 0.50
MIN_DET_SCORE = 0.50

TARGET_FRAMES = 10
MIN_FRAMES = 10
MIN_DURATION_SEC = 1.5
MIN_FACE_FRAME_RATIO = 0.6
MAX_MULTI_FACE_RATIO = 0.2

MIN_YAW_RANGE_DEG = 10.0
MIN_PITCH_RANGE_DEG = 8.0
REQUIRE_FACIAL_MOVEMENT = True
MIN_EAR_CHANGE = 0.25
MIN_MAR_CHANGE = 0.35

MAX_SIDE = 960

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
}


def _result(reason_code, failed_checks=None, **extra):
    return {
        "verified": reason_code == "OK",
        "reason_code": reason_code,
        "reason": REASONS[reason_code],
        "failed_checks": failed_checks or ([] if reason_code == "OK" else [reason_code]),
        **extra,
    }


def _resize(img):
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def _largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def extract_frames(video_path, target=TARGET_FRAMES):
    """Sample `target` frames evenly from the video. Returns (frames, info)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], {"error": "open_failed"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if count <= 0 or count > 100000:
        count = 0
        while cap.grab():
            count += 1
        cap.release()
        cap = cv2.VideoCapture(video_path)

    if count == 0:
        cap.release()
        return [], {"error": "no_frames"}

    wanted = set(np.linspace(0, count - 1, min(target, count)).astype(int).tolist())
    frames, idx = [], 0
    while cap.grab():
        if idx in wanted:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                frames.append(_resize(frame))
        idx += 1
        if idx > max(wanted):
            break
    cap.release()

    duration = count / fps if fps and 0 < fps < 240 else None
    return frames, {"total_frames": count, "fps": fps, "duration_sec": duration}


def _pose(face):
    """(pitch, yaw) in degrees. Falls back to a rough 5-point estimate."""
    pose = getattr(face, "pose", None)
    if pose is not None:
        return float(pose[0]), float(pose[1])
    k = face.kps
    eye_mid = (k[0] + k[1]) / 2
    eye_dist = max(np.linalg.norm(k[1] - k[0]), 1.0)
    mouth_mid = (k[3] + k[4]) / 2
    yaw = (k[2][0] - eye_mid[0]) / eye_dist * 90
    pitch = ((k[2][1] - eye_mid[1]) / max(mouth_mid[1] - eye_mid[1], 1.0) - 0.5) * 90
    return float(pitch), float(yaw)


def _eye_mouth_ratios(face):
    """Eye aspect ratio + mouth aspect ratio from 68-point landmarks."""
    lm = getattr(face, "landmark_3d_68", None)
    if lm is None:
        return None, None
    p = lm[:, :2]
    d = lambda a, b: np.linalg.norm(p[a] - p[b])

    def ear(i):
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


def verify_faces(profile_photo_path, video_path):
    img = cv2.imread(profile_photo_path)
    if img is None:
        return _result("PROFILE_UNREADABLE")
    profile_faces = [f for f in _face_app.get(_resize(img)) if f.det_score >= MIN_DET_SCORE]
    if not profile_faces:
        return _result("NO_FACE_IN_PROFILE")
    profile = _largest(profile_faces)

    frames, video_info = extract_frames(video_path)
    if not frames:
        return _result("VIDEO_UNREADABLE", video=video_info)
    dur = video_info.get("duration_sec")
    if len(frames) < MIN_FRAMES or (dur is not None and dur < MIN_DURATION_SEC):
        return _result("VIDEO_TOO_SHORT", video={**video_info, "frames_sampled": len(frames)})

    video_faces, multi_face_frames = [], 0
    for frame in frames:
        faces = [f for f in _face_app.get(frame) if f.det_score >= MIN_DET_SCORE]
        if len(faces) > 1:
            multi_face_frames += 1
        if faces:
            video_faces.append(_largest(faces))

    face_ratio = len(video_faces) / len(frames)
    multi_ratio = multi_face_frames / len(frames)
    video_info.update({
        "frames_sampled": len(frames),
        "frames_with_face": len(video_faces),
        "face_ratio": round(face_ratio, 3),
        "multi_face_ratio": round(multi_ratio, 3),
    })

    if multi_ratio > MAX_MULTI_FACE_RATIO:
        return _result("MULTIPLE_FACES_IN_VIDEO", video=video_info)
    if face_ratio < MIN_FACE_FRAME_RATIO or len(video_faces) < MIN_FRAMES // 2:
        return _result("FACE_NOT_CONSISTENT", video=video_info)

    emb = np.stack([f.normed_embedding for f in video_faces])
    mean_emb = emb.mean(axis=0)
    mean_emb /= np.linalg.norm(mean_emb)
    consistency = emb @ mean_emb
    video_info["min_consistency"] = round(float(consistency.min()), 3)

    poses = np.array([_pose(f) for f in video_faces])
    pitch_range = float(np.ptp(poses[:, 0]))
    yaw_range = float(np.ptp(poses[:, 1]))
    ratios = [_eye_mouth_ratios(f) for f in video_faces]
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

    sims = emb @ profile.normed_embedding
    similarity = float(np.median(sims))
    match_ratio = float(np.mean(sims >= SIMILARITY_THRESHOLD))
    identity = {
        "similarity": round(similarity, 4),
        "best_similarity": round(float(sims.max()), 4),
        "match_ratio": round(match_ratio, 3),
        "threshold": SIMILARITY_THRESHOLD,
        "profile_det_score": round(float(profile.det_score), 3),
    }

    failed = []
    if consistency.min() < SAME_PERSON_THRESHOLD:
        failed.append("DIFFERENT_PEOPLE_IN_VIDEO")
    if not head_moved:
        failed.append("NO_HEAD_MOVEMENT")
    if REQUIRE_FACIAL_MOVEMENT and not face_moved:
        failed.append("NO_FACIAL_MOVEMENT")
    if similarity < SIMILARITY_THRESHOLD or match_ratio < MIN_MATCH_RATIO:
        failed.append("FACE_MISMATCH")

    return _result(
        failed[0] if failed else "OK",
        failed_checks=failed,
        live=not any(c in failed for c in ("NO_HEAD_MOVEMENT", "NO_FACIAL_MOVEMENT")),
        identity_match="FACE_MISMATCH" not in failed,
        similarity=identity["similarity"],
        identity=identity,
        liveness=liveness,
        video=video_info,
    )
