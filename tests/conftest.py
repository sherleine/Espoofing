"""Shared fixtures. Integration tests build synthetic videos from the sample
photos that ship with InsightFace, so no personal data is needed."""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HAS_MODELS = importlib.util.find_spec("insightface") is not None
models = pytest.mark.skipif(not HAS_MODELS, reason="insightface not installed")


@pytest.fixture(scope="session")
def cv2():
    import cv2 as _cv2
    return _cv2


@pytest.fixture(scope="session")
def sample_dir():
    import insightface
    return Path(insightface.__file__).parent / "data" / "images"


@pytest.fixture(scope="session")
def group_photo(cv2, sample_dir):
    """t1.jpg: six people."""
    return cv2.imread(str(sample_dir / "t1.jpg"))


@pytest.fixture(scope="session")
def people(cv2, group_photo):
    """Two single-person, near-frontal crops (A, B) from the group photo,
    upscaled so the face is a realistic webcam size."""
    from app.verification import main
    faces = main.analyze_frame(group_photo, landmarks=True, embedding=False)
    frontal = sorted(faces, key=lambda f: abs(float(f.pose[1])))[:2]
    crops = []
    h, w = group_photo.shape[:2]
    for f in frontal:
        # grey out everybody else so the crop holds exactly one face
        img = group_photo.copy()
        for other in faces:
            if other is f:
                continue
            ox1, oy1, ox2, oy2 = other.bbox
            pad = 0.25 * (ox2 - ox1)
            img[int(max(oy1 - pad, 0)):int(min(oy2 + pad, h)),
                int(max(ox1 - pad, 0)):int(min(ox2 + pad, w))] = 127
        x1, y1, x2, y2 = f.bbox
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.6
        a, b = int(max(cy - s, 0)), int(min(cy + s, h))
        c, d = int(max(cx - s, 0)), int(min(cx + s, w))
        crops.append(cv2.resize(img[a:b, c:d], (480, 480)))
    for crop in crops:
        assert len(main.analyze_frame(crop, landmarks=False, embedding=False)) == 1
    return crops


def jitter_frames(cv2, img, n, seed=0, shift=6.0, angle=2.0):
    """`n` slightly moved copies of `img` (hand-held camera feel)."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out = []
    for _ in range(n):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-angle, angle), rng.uniform(0.98, 1.02))
        m[:, 2] += rng.uniform(-shift, shift, 2)
        out.append(cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT))
    return out


def write_video(cv2, path, frames, fps=15):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "OpenCV cannot write mp4v on this machine"
    for f in frames:
        writer.write(cv2.resize(f, (w, h)))
    writer.release()
    return str(path)


def write_image(cv2, path, img):
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture(scope="session")
def media(tmp_path_factory, cv2, people, group_photo):
    """Synthetic profile photos and videos covering the main outcomes."""
    d = tmp_path_factory.mktemp("media")
    a, b = people
    group = cv2.resize(group_photo, (960, 664))
    m = {
        "profile_a": write_image(cv2, d / "profile_a.jpg", a),
        "profile_b": write_image(cv2, d / "profile_b.jpg", b),
        "profile_blank": write_image(cv2, d / "blank.jpg", np.full((480, 480, 3), 127, np.uint8)),
        "video_a": write_video(cv2, d / "a.mp4", jitter_frames(cv2, a, 45, seed=1)),
        "video_group": write_video(cv2, d / "group.mp4", jitter_frames(cv2, group, 45, seed=2)),
        "video_swap": write_video(cv2, d / "swap.mp4",
                                  jitter_frames(cv2, a, 23, seed=3) + jitter_frames(cv2, b, 22, seed=4)),
        "video_short": write_video(cv2, d / "short.mp4", jitter_frames(cv2, a, 8, seed=5)),
        "video_noface": write_video(cv2, d / "noface.mp4",
                                    [np.full((480, 480, 3), 90, np.uint8)] * 45),
    }
    corrupt = d / "corrupt.webm"
    corrupt.write_bytes(os.urandom(4096))
    m["video_corrupt"] = str(corrupt)
    return m
