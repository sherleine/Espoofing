"""Passive PAD: aggregation rules (model output stubbed) and model parity."""

import importlib.util

import numpy as np
import pytest

from app.verification import anti_spoof
from conftest import models

FACE = object()   # detect_spoof only passes faces through to the stubbed functions
FRAME = np.zeros((10, 10, 3), np.uint8)


@pytest.fixture
def stub(monkeypatch):
    """Make every face pass the quality gate and return the given live
    probabilities in order."""
    def use(live_probabilities, quality=None):
        monkeypatch.setattr(anti_spoof, "face_quality",
                            quality or (lambda frame, face: (True, None, {})))
        monkeypatch.setattr(anti_spoof, "live_probabilities",
                            lambda frames, faces: list(live_probabilities)[:len(frames)])
    return use


def test_status_follows_median_spoof_score(stub):
    for live, status in [([0.95, 0.9, 0.97], "LOW_RISK"),
                         ([0.7, 0.75, 0.7], "ELEVATED"),
                         ([0.1, 0.2, 0.9], "HIGH_RISK")]:
        stub(live)
        result = anti_spoof.detect_spoof([FRAME] * 3, [FACE] * 3)
        assert result["status"] == status, live
        assert result["spoof_score"] == round(1 - float(np.median(live)), 4)


def test_response_fields(stub):
    stub([0.9, 0.9, 0.9])
    result = anti_spoof.detect_spoof([FRAME] * 3, [FACE] * 3)
    assert set(result) == {"status", "spoof_score", "frames_scored", "frames_skipped", "frame_scores"}
    assert result["frames_scored"] == 3 and result["frame_scores"] == [0.1, 0.1, 0.1]


def test_skipped_frames_are_counted_not_scored(stub):
    blurry = object()
    stub([0.9, None, 0.9, 0.9],
         quality=lambda frame, face: (False, "BLURRY", {}) if face is blurry else (True, None, {}))
    result = anti_spoof.detect_spoof([FRAME] * 6, [FACE, None, blurry, FACE, FACE, FACE])
    # frame 1: no face, frame 2: blurry, one of the remaining 4 cannot be cropped
    assert result["frames_skipped"] == {"NO_FACE": 1, "BLURRY": 1, "FACE_AT_EDGE": 1}
    assert result["frames_scored"] == 3 and result["status"] == "LOW_RISK"


def test_too_few_usable_frames_is_insufficient_quality(stub):
    stub([0.1, 0.1])
    result = anti_spoof.detect_spoof([FRAME] * 2, [FACE] * 2)
    assert result["status"] == "INSUFFICIENT_QUALITY" and result["spoof_score"] is None
    assert anti_spoof.detect_spoof([FRAME] * 2, [FACE] * 2, min_scored_frames=2)["status"] == "HIGH_RISK"


def test_no_faces_never_calls_the_model(stub, monkeypatch):
    def fail(frames, faces):
        raise AssertionError("model called without usable faces")
    monkeypatch.setattr(anti_spoof, "live_probabilities", fail)
    result = anti_spoof.detect_spoof([FRAME] * 3, [None] * 3)
    assert result["frames_skipped"] == {"NO_FACE": 3} and result["status"] == "INSUFFICIENT_QUALITY"


@models
@pytest.mark.skipif(importlib.util.find_spec("insightface.addons") is None,
                    reason="reference wrapper needs insightface >= 2.0")
def test_scores_match_insightface_reference(group_photo):
    """Our onnxruntime-only preprocessing must give the same scores as
    InsightFace's own liveness wrapper for the same model."""
    from insightface.addons import Liveness, ensure_addon
    from app.verification.faces import analyze_frame
    detected = analyze_frame(group_photo, landmarks=False, embedding=False)
    reference = Liveness(ensure_addon("liveness"), providers=["CPUExecutionProvider"])
    ours = anti_spoof.live_probabilities([group_photo] * len(detected), detected)
    theirs = [reference.predict(group_photo, f.kps)["live_score"] for f in detected]
    assert np.allclose(ours, theirs, atol=1e-6)
