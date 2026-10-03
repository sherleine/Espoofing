"""The refactored registration path must return exactly what the original
service.py returned (new keys like `anti_spoof` excluded) when no challenge
is used and PAD enforcement is off."""

import importlib.util
from pathlib import Path

import pytest

from conftest import models

NEW_KEYS = {"anti_spoof", "challenge"}
CASES = [
    ("profile_a", "video_a"),          # same person, no head/face movement
    ("profile_b", "video_a"),          # different person
    ("profile_a", "video_group"),      # six people
    ("profile_a", "video_swap"),       # person changes mid-video
    ("profile_a", "video_short"),
    ("profile_a", "video_noface"),
    ("profile_a", "video_corrupt"),
    ("profile_blank", "video_a"),      # no face in profile
    ("missing.jpg", "video_a"),        # unreadable profile
]


@pytest.fixture(scope="module")
def legacy():
    path = Path(__file__).parent / "legacy" / "service_original.py"
    spec = importlib.util.spec_from_file_location("service_original", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@models
@pytest.mark.parametrize("profile,video", CASES)
def test_same_result_as_original(legacy, media, monkeypatch, profile, video):
    from app.verification import main, service
    monkeypatch.setattr(main, "PAD_ENFORCE_REGISTRATION", False)
    profile_path = media.get(profile, profile)
    old = legacy.verify_faces(profile_path, media[video])
    new = service.verify_faces(profile_path, media[video])
    assert {k: v for k, v in new.items() if k not in NEW_KEYS} == old


@models
def test_cases_cover_distinct_outcomes(legacy, media):
    codes = {legacy.verify_faces(media.get(p, p), media[v])["reason_code"] for p, v in CASES}
    assert len(codes) >= 6, codes
