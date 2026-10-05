"""/register -> .NET stores reference_embedding -> /verify and exam start."""

import json

import cv2
import numpy as np
import pytest

from conftest import jitter_frames, models


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from server import app
    return TestClient(app)


@pytest.fixture(scope="module")
def registered(media):
    """A successful registration of person A. Synthetic videos cannot turn a
    head, so only the movement check is switched off; identity, same-person
    and PAD checks run as normal."""
    from app.verification import main, registration_liveness
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(registration_liveness, "movement_check", lambda faces: ({}, []))
        result = main.verify_registration(media["profile_a"], media["video_a"])
    assert result["verified"], result
    return result


def verify(client, media, reference, video="video_a", nonce=None):
    data = {"email": "a@example.com", "examId": "1",
            "reference_embedding": reference if isinstance(reference, str) else json.dumps(reference)}
    if nonce:
        data["challenge_nonce"] = nonce
    files = {"video": ("v.mp4", open(media[video], "rb"), "video/mp4")}
    return client.post("/verify", data=data, files=files).json()


@models
def test_successful_registration_returns_reference_embedding(registered):
    from app.verification.faces import EMBEDDING_MODEL
    ref = registered["reference_embedding"]
    assert ref["model"] == EMBEDDING_MODEL and len(ref["vector"]) == 512
    assert abs(np.linalg.norm(ref["vector"]) - 1) < 1e-3


@models
def test_failed_registration_returns_no_embedding(media):
    from app.verification import main
    result = main.verify_registration(media["profile_b"], media["video_a"])
    assert not result["verified"] and "reference_embedding" not in result


@models
def test_register_requires_a_challenge(client, media):
    form = {"email": "a@example.com", "examId": "1"}
    files = lambda: {"profile_photo": ("p.jpg", open(media["profile_a"], "rb"), "image/jpeg"),
                     "video": ("v.mp4", open(media["video_a"], "rb"), "video/mp4")}
    assert client.post("/register", data=form, files=files()).status_code == 422
    body = client.post("/register", data={**form, "challenge_nonce": "  "}, files=files()).json()
    assert body["reason_code"] == "CHALLENGE_MISSING"

    nonce = client.get("/verification/challenge").json()["nonce"]
    body = client.post("/register", data={**form, "challenge_nonce": nonce}, files=files()).json()
    assert body["reason_code"] == "CHALLENGE_FAILED" and "reference_embedding" not in body


@models
def test_passive_verify_same_person(client, media, registered):
    body = verify(client, media, registered["reference_embedding"])
    assert body["verified"] and body["mode"] == "passive", body
    assert body["identity"]["similarity"] > 0.9


@models
def test_passive_verify_other_person(client, media, registered, people, tmp_path):
    from conftest import write_video
    other = write_video(tmp_path / "b.mp4", jitter_frames(people[1], 30, seed=7))
    body = verify(client, {**media, "video_b": other}, registered["reference_embedding"], video="video_b")
    assert body["reason_code"] == "IDENTITY_MISMATCH" and not body["identity_match"]


@models
def test_verify_with_challenge_needs_the_actions(client, media, registered):
    nonce = client.get("/verification/challenge").json()["nonce"]
    body = verify(client, media, registered["reference_embedding"], nonce=nonce)
    # a still video cannot perform the requested actions
    assert body["mode"] == "challenge" and body["reason_code"] == "CHALLENGE_FAILED"
    assert body["challenge"]["passed"] is False and body["identity_match"]
    assert verify(client, media, registered["reference_embedding"], nonce=nonce)["reason_code"] == "CHALLENGE_INVALID"


@models
def test_verify_rejects_bad_embeddings(client, media, registered):
    good = registered["reference_embedding"]
    assert verify(client, media, "not json")["reason_code"] == "INVALID_REFERENCE_EMBEDDING"
    assert verify(client, media, {**good, "vector": good["vector"][:100]})["reason_code"] == "INVALID_REFERENCE_EMBEDDING"
    assert verify(client, media, {**good, "vector": ["x"] * 512})["reason_code"] == "INVALID_REFERENCE_EMBEDDING"
    assert verify(client, media, {**good, "model": "other-model"})["reason_code"] == "REFERENCE_MODEL_MISMATCH"


@models
def test_exam_uses_reference_embedding(client, registered, people):
    start = client.post("/exam/session/start", data={
        "email": "a@example.com", "examId": "1",
        "reference_embedding": json.dumps(registered["reference_embedding"])}).json()
    assert start["started"] and start["reference"] == "registration_embedding"

    jpgs = [("frames", (f"{i}.jpg", cv2.imencode(".jpg", f)[1].tobytes(), "image/jpeg"))
            for i, f in enumerate(jitter_frames(people[0], 6))]
    window = client.post(f"/exam/session/{start['session_id']}/window", files=jpgs).json()
    assert window["signals"]["identity"] == "MATCH" and window["state"] == "NORMAL"
    client.post(f"/exam/session/{start['session_id']}/end")


@models
def test_exam_start_needs_a_reference(client):
    body = client.post("/exam/session/start", data={"email": "a@example.com", "examId": "1"}).json()
    assert not body["started"] and body["reason_code"] == "NO_REFERENCE"


@models
def test_photo_uploaded_as_video_is_reported_clearly(client, media, registered):
    from app.verification import main
    assert main.verify_registration(media["profile_a"], media["profile_a"])["reason_code"] == "VIDEO_IS_IMAGE"
    body = verify(client, {**media, "photo": media["profile_a"]}, registered["reference_embedding"], video="photo")
    assert body["reason_code"] == "VIDEO_IS_IMAGE"


@models
def test_exam_window_with_one_image_is_rejected(client, media, registered):
    start = client.post("/exam/session/start", data={
        "email": "a@example.com", "examId": "1",
        "reference_embedding": json.dumps(registered["reference_embedding"])}).json()
    sid = start["session_id"]
    one_frame = {"frames": ("p.jpg", open(media["profile_a"], "rb"), "image/jpeg")}
    photo_as_clip = {"clip": ("p.jpg", open(media["profile_a"], "rb"), "image/jpeg")}
    for files in (one_frame, photo_as_clip):
        body = client.post(f"/exam/session/{sid}/window", files=files).json()
        assert not body["ok"] and body["reason_code"] == "TOO_FEW_FRAMES", body
        assert "Received 1 frame" in body["reason"]
    client.post(f"/exam/session/{sid}/end")
