"""HTTP layer: auth, response compatibility, challenge flow, exam endpoints."""

import pytest

from conftest import jitter_frames, models


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from server import app
    return TestClient(app)


def files(media, profile="profile_a", video="video_a"):
    return {
        "email_photo": ("p.jpg", open(media[profile], "rb"), "image/jpeg"),
        "live_photo": ("v.mp4", open(media[video], "rb"), "video/mp4"),
    }


LEGACY_KEYS = {"verified", "is_match", "reason_code", "reason", "failed_checks", "live",
               "identity_match", "similarity", "identity", "liveness", "video"}


@models
def test_service_key_enforced(client, media, monkeypatch):
    from app.verification import router
    monkeypatch.setattr(router, "SERVICE_KEY", "secret")
    form = {"email": "a@example.com", "examId": "1"}
    assert client.post("/verify-email-photo", data=form, files=files(media)).status_code == 401
    r = client.post("/verify-email-photo", data=form, files=files(media), headers={"X-Service-Key": "secret"})
    assert r.status_code == 200


@models
def test_legacy_request_keeps_response_shape(client, media):
    r = client.post("/verify-email-photo", data={"email": "a@example.com", "examId": "1"}, files=files(media))
    body = r.json()
    assert LEGACY_KEYS <= set(body)
    assert "anti_spoof" in body and body["is_match"] == body["verified"]


@models
def test_uploads_are_deleted(client, media):
    from app.verification import view
    import os
    before = set(os.listdir(view.UPLOAD_DIR))
    client.post("/verify-email-photo", data={"email": "a@example.com", "examId": "1"}, files=files(media))
    assert set(os.listdir(view.UPLOAD_DIR)) == before


@models
def test_challenge_flow(client, media):
    c = client.get("/verification/challenge").json()
    form = {"email": "a@example.com", "examId": "1", "challenge_nonce": c["nonce"]}
    body = client.post("/verify-email-photo", data=form, files=files(media)).json()
    # a still photo cannot perform the requested actions
    assert body["reason_code"] == "CHALLENGE_FAILED" and body["live"] is False
    assert body["challenge"]["steps"] == c["steps"]
    # the nonce cannot be reused
    body = client.post("/verify-email-photo", data=form, files=files(media)).json()
    assert body["reason_code"] == "CHALLENGE_INVALID"


@models
def test_exam_endpoints(client, media, people, cv2):
    start = client.post("/exam/session/start", data={"email": "a@example.com", "examId": "1"},
                        files={"email_photo": ("p.jpg", open(media["profile_a"], "rb"), "image/jpeg")}).json()
    sid = start["session_id"]
    jpgs = [("frames", (f"{i}.jpg", cv2.imencode(".jpg", f)[1].tobytes(), "image/jpeg"))
            for i, f in enumerate(jitter_frames(cv2, people[0], 6))]
    r = client.post(f"/exam/session/{sid}/window", files=jpgs).json()
    assert r["ok"] and r["state"] == "NORMAL" and r["signals"]["identity"] == "MATCH"

    clip = {"clip": ("c.mp4", open(media["video_a"], "rb"), "video/mp4")}
    r = client.post(f"/exam/session/{sid}/window", files=clip).json()
    assert r["ok"] and r["window"] == 2

    assert client.get(f"/exam/session/{sid}").json()["windows"] == 2
    assert client.post(f"/exam/session/{sid}/end").json()["ok"]
    assert client.get(f"/exam/session/{sid}").json()["reason_code"] == "SESSION_NOT_FOUND"
