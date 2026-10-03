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

from app.verification.faces import face_quality

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
