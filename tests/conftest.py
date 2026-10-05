"""Shared fixtures. Integration tests build synthetic videos from the sample
photos that ship with InsightFace, so no personal data is needed."""

import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HAS_MODELS = importlib.util.find_spec("insightface") is not None
models = pytest.mark.skipif(not HAS_MODELS, reason="insightface not installed")


@pytest.fixture(scope="session")
def sample_dir():
    import insightface
    return Path(insightface.__file__).parent / "data" / "images"


@pytest.fixture(scope="session")
def group_photo(sample_dir):
    """t1.jpg: six people."""
    return cv2.imread(str(sample_dir / "t1.jpg"))


@pytest.fixture(scope="session")
def group_faces(group_photo):
    from app.verification.faces import analyze_frame
    return analyze_frame(group_photo, landmarks=True, embedding=False)


def single_face_crop(photo, faces, face):
    """Crop around `face`, everybody else greyed out, upscaled so the face
    is a realistic webcam size."""
    from app.verification.faces import analyze_frame
    img = photo.copy()
    h, w = img.shape[:2]
    for other in faces:
        if other is face:
            continue
        ox1, oy1, ox2, oy2 = other.bbox
        pad = 0.25 * (ox2 - ox1)
        img[int(max(oy1 - pad, 0)):int(min(oy2 + pad, h)),
            int(max(ox1 - pad, 0)):int(min(ox2 + pad, w))] = 127
    x1, y1, x2, y2 = face.bbox
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.6
    crop = cv2.resize(img[int(max(cy - s, 0)):int(min(cy + s, h)),
                          int(max(cx - s, 0)):int(min(cx + s, w))], (480, 480))
    assert len(analyze_frame(crop, landmarks=False, embedding=False)) == 1
    return crop


@pytest.fixture(scope="session")
def people(group_photo, group_faces):
    """Two near-frontal single-person crops (A, B)."""
    frontal = sorted(group_faces, key=lambda f: abs(float(f.pose[1])))[:2]
    return [single_face_crop(group_photo, group_faces, f) for f in frontal]


@pytest.fixture(scope="session")
def side_profile(group_photo, group_faces):
    """Crop of the most strongly turned face in the group photo."""
    turned = max(group_faces, key=lambda f: abs(float(f.pose[1])))
    return single_face_crop(group_photo, group_faces, turned)


def jitter_frames(img, n, seed=0, shift=6.0, angle=2.0):
    """`n` slightly moved copies of `img` (hand-held camera feel)."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out = []
    for _ in range(n):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-angle, angle), rng.uniform(0.98, 1.02))
        m[:, 2] += rng.uniform(-shift, shift, 2)
        out.append(cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT))
    return out


def write_video(path, frames, fps=15):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "OpenCV cannot write mp4v on this machine"
    for f in frames:
        writer.write(cv2.resize(f, (w, h)))
    writer.release()
    return str(path)


def write_image(path, img):
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture(scope="session")
def media(tmp_path_factory, people, group_photo):
    """Synthetic profile photos and videos covering the main outcomes."""
    d = tmp_path_factory.mktemp("media")
    a, b = people
    group = cv2.resize(group_photo, (960, 664))
    m = {
        "profile_a": write_image(d / "profile_a.jpg", a),
        "profile_b": write_image(d / "profile_b.jpg", b),
        "profile_blank": write_image(d / "blank.jpg", np.full((480, 480, 3), 127, np.uint8)),
        "video_a": write_video(d / "a.mp4", jitter_frames(a, 45, seed=1)),
        "video_group": write_video(d / "group.mp4", jitter_frames(group, 45, seed=2)),
        "video_swap": write_video(d / "swap.mp4",
                                  jitter_frames(a, 23, seed=3) + jitter_frames(b, 22, seed=4)),
        "video_short": write_video(d / "short.mp4", jitter_frames(a, 8, seed=5)),
        "video_noface": write_video(d / "noface.mp4",
                                    [np.full((480, 480, 3), 90, np.uint8)] * 45),
    }
    corrupt = d / "corrupt.webm"
    corrupt.write_bytes(os.urandom(4096))
    m["video_corrupt"] = str(corrupt)
    return m
