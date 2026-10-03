# app/verification/anti_spoof.py
"""
PASSIVE presentation-attack detection (PAD). The candidate does nothing.

    detect_spoof(frames, faces) -> dict

`faces` holds the main face of every frame (None when there is none), as
produced by main.py. The result never claims "this person is alive"; it says
how much the frames look like a known presentation attack:

    status       LOW_RISK | ELEVATED | HIGH_RISK | INSUFFICIENT_QUALITY
    spoof_score  0.0 (looks genuine) .. 1.0 (looks like an attack), median over frames
    confidence   0..1, how much the scored frames agree and how many there were
    attack_type  None, or "presentation_attack" (the current model is binary and
                 cannot tell print from screen)

What it covers: re-captured faces - printed photos, photos/videos shown on a
phone, tablet or monitor - through texture/moire/screen artifacts.
What it does NOT cover: video injected through a virtual camera, or a
deepfake fed in digitally. Those frames were never re-captured, so they carry
no presentation artifacts (a plain digital photo scores "genuine"). Those
need stream-integrity checks, which are a separate module/workstream.

Swapping the model: write a class with `name`, `version` and
`live_probabilities(frames, faces)` and put it in BACKENDS. Nothing outside
this file changes.
"""

import hashlib
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from skimage.transform import SimilarityTransform

# ---------------------------------------------------------------------------
# Decision thresholds - PLACEHOLDERS, NOT CALIBRATED.
# Calibrate with tools/evaluate_pad.py on your own bona fide + attack videos.
# ---------------------------------------------------------------------------
SPOOF_ELEVATED = 0.20           # = the model's own default operating point (live >= 0.8)
SPOOF_HIGH = 0.50
MIN_SCORED_FRAMES = 3           # fewer good frames -> INSUFFICIENT_QUALITY

# Frame quality gate: a bad frame is skipped, never scored as an attack.
MIN_FACE_PX = 60                # face box width in pixels
MIN_SHARPNESS = 15.0            # variance of Laplacian on the 112x112 face crop
MIN_BRIGHTNESS = 40.0           # mean grey level of the face crop
MAX_BRIGHTNESS = 225.0


# ---------------------------------------------------------------------------
# Backend 1: InsightFace RGB liveness model (80x80, 5-point aligned).
# Pretrained weights: InsightFace states they are for NON-COMMERCIAL research
# use only - check licensing before production (see README).
# ---------------------------------------------------------------------------
class InsightFaceLiveness:
    name = "insightface_liveness_80"
    url = "https://github.com/deepinsight/insightface-model-addons/releases/download/addons/liveness.onnx"
    sha256 = "87a9ac1dbb16a61eec212957e5095e62a8769c1e188af9b0198f253302c4afdb"
    size = 80
    max_out_of_bounds = 0.30    # skip faces too close to the image border
    dst = np.array([[32.03, 38.06], [47.89, 37.98], [40.01, 47.08],
                    [33.50, 56.36], [46.63, 56.29]], dtype=np.float32)

    def __init__(self, model_path=None):
        self.model_path = Path(model_path or os.getenv(
            "PAD_MODEL_PATH", "~/.insightface/addons/liveness.onnx")).expanduser()
        self.version = self.sha256[:12]
        self._session = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._session is None:
                if not self.model_path.exists():
                    self._download()
                digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
                if digest != self.sha256:
                    raise RuntimeError(f"PAD model checksum mismatch: {self.model_path}")
                self._session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"])
                self._input = self._session.get_inputs()[0].name
        return self._session

    def _download(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.model_path.with_suffix(".download")
        urllib.request.urlretrieve(self.url, tmp)
        os.replace(tmp, self.model_path)

    def _crop(self, frame, kps):
        """Aligned 80x80 crop, or None if too much of it falls outside the frame."""
        src = np.asarray(kps, dtype=np.float32)
        if hasattr(SimilarityTransform, "from_estimate"):
            tform = SimilarityTransform.from_estimate(src, self.dst)
            if not tform:
                return None
        else:
            tform = SimilarityTransform()
            if not tform.estimate(src, self.dst):
                return None
        m = tform.params[:2, :]
        inside = cv2.warpAffine(np.ones(frame.shape[:2], np.float32), m, (self.size, self.size))
        if 1.0 - float(inside.mean()) > self.max_out_of_bounds:
            return None
        return cv2.warpAffine(frame, m, (self.size, self.size), borderMode=cv2.BORDER_REPLICATE)

    def live_probabilities(self, frames, faces):
        """One live probability per (frame, face); None where it can't score."""
        session = self._load()
        crops = [self._crop(fr, f.kps) for fr, f in zip(frames, faces)]
        idx = [i for i, c in enumerate(crops) if c is not None]
        out = [None] * len(crops)
        if idx:
            batch = np.stack([crops[i][:, :, ::-1].transpose(2, 0, 1) for i in idx])
            probs = session.run(None, {self._input: batch.astype(np.float32) / 255.0})[0]
            for i, p in zip(idx, np.asarray(probs).reshape(-1)):
                out[i] = float(p)
        return out


BACKENDS = [InsightFaceLiveness()]


def warm_up():
    """Load/download the models now instead of on the first request."""
    for backend in BACKENDS:
        if hasattr(backend, "_load"):
            backend._load()


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------
def face_quality(frame, face):
    """Returns (ok, reason, metrics)."""
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def detect_spoof(frames, faces):
    skipped = {}
    good = []
    for i, (frame, face) in enumerate(zip(frames, faces)):
        if face is None:
            skipped["NO_FACE"] = skipped.get("NO_FACE", 0) + 1
            continue
        ok, reason, _ = face_quality(frame, face)
        if not ok:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        good.append(i)

    # live probability per good frame, averaged over the configured backends
    per_backend = {}
    live = np.full(len(good), np.nan)
    if good:
        sums, counts = np.zeros(len(good)), np.zeros(len(good))
        for backend in BACKENDS:
            probs = backend.live_probabilities([frames[i] for i in good], [faces[i] for i in good])
            valid = [p for p in probs if p is not None]
            per_backend[f"{backend.name}@{backend.version}"] = (
                round(1.0 - float(np.median(valid)), 4) if valid else None)
            for j, p in enumerate(probs):
                if p is not None:
                    sums[j] += p
                    counts[j] += 1
        live = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    unscored = int(np.isnan(live).sum())
    if unscored:
        skipped["FACE_AT_EDGE"] = skipped.get("FACE_AT_EDGE", 0) + unscored
    scores = 1.0 - live[~np.isnan(live)]

    result = {
        "status": "INSUFFICIENT_QUALITY",
        "spoof_score": None,
        "confidence": None,
        "attack_type": None,
        "frames_scored": int(len(scores)),
        "frames_skipped": skipped,
        "frame_scores": [round(float(s), 4) for s in scores],
        "models": per_backend,
        "thresholds": {"elevated": SPOOF_ELEVATED, "high": SPOOF_HIGH, "calibrated": False},
    }
    if len(scores) < MIN_SCORED_FRAMES:
        return result

    spoof = float(np.median(scores))
    if spoof >= SPOOF_HIGH:
        status = "HIGH_RISK"
    elif spoof >= SPOOF_ELEVATED:
        status = "ELEVATED"
    else:
        status = "LOW_RISK"
    # agreement: share of frames on the same side of the elevated threshold as
    # the median; scaled down when only a few frames could be scored
    agreement = float(np.mean((scores >= SPOOF_ELEVATED) == (spoof >= SPOOF_ELEVATED)))
    coverage = min(1.0, len(scores) / (2 * MIN_SCORED_FRAMES))

    result.update({
        "status": status,
        "spoof_score": round(spoof, 4),
        "confidence": round(agreement * coverage, 3),
        "attack_type": None if status == "LOW_RISK" else "presentation_attack",
    })
    return result
