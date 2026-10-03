import time

from app.verification import registration_liveness as rl


def frames(*segments):
    """segments: (count, yaw, eye, mouth) -> measurement list."""
    out = []
    for count, yaw, eye, mouth in segments:
        out += [{"pitch": 0.0, "yaw": yaw, "eye": eye, "mouth": mouth}] * count
    return out


NEUTRAL = (6, 0.0, 0.30, 0.05)
LEFT = (3, 30.0, 0.30, 0.05)     # positive yaw = candidate's left (non-mirrored)
RIGHT = (3, -30.0, 0.30, 0.05)
BLINK = (1, 0.0, 0.10, 0.05)
MOUTH = (2, 0.0, 0.30, 0.60)


def challenge(*steps):
    return {"nonce": "n", "steps": list(steps)}


def test_requested_sequence_passes():
    m = frames(NEUTRAL, LEFT, NEUTRAL, BLINK, NEUTRAL, RIGHT, NEUTRAL)
    r = rl.evaluate_challenge(challenge("turn_left", "blink", "turn_right"), m)
    assert r["passed"], r
    assert [x["action"] for x in r["matched"]] == ["turn_left", "blink", "turn_right"]


def test_wrong_order_fails():
    m = frames(NEUTRAL, RIGHT, NEUTRAL, BLINK, NEUTRAL, LEFT, NEUTRAL)
    r = rl.evaluate_challenge(challenge("turn_left", "blink", "turn_right"), m)
    assert not r["passed"]
    # turn_left is only found at the end, so the blink after it is missing
    assert r["failure"] == "WRONG_ORDER:blink"


def test_missing_action_fails():
    m = frames(NEUTRAL, LEFT, NEUTRAL, NEUTRAL)
    r = rl.evaluate_challenge(challenge("turn_left", "open_mouth", "blink"), m)
    assert r["failure"] == "NOT_DETECTED:open_mouth"


def test_do_everything_recording_fails():
    """A clip that performs every action (to satisfy any challenge) is rejected
    because it contains head turns that were not requested."""
    m = frames(NEUTRAL, LEFT, NEUTRAL, RIGHT, NEUTRAL, BLINK, NEUTRAL, MOUTH, NEUTRAL,
               LEFT, NEUTRAL, RIGHT, NEUTRAL)
    r = rl.evaluate_challenge(challenge("blink", "open_mouth", "turn_left"), m)
    assert not r["passed"]
    assert r["failure"] == "UNREQUESTED_HEAD_TURN"


def test_spontaneous_blinks_are_tolerated():
    m = frames(NEUTRAL, BLINK, NEUTRAL, MOUTH, NEUTRAL, BLINK, NEUTRAL, RIGHT, NEUTRAL, BLINK, NEUTRAL)
    r = rl.evaluate_challenge(challenge("open_mouth", "turn_right", "blink"), m)
    assert r["passed"], r


def test_static_photo_fails():
    r = rl.evaluate_challenge(challenge("turn_left", "blink", "open_mouth"), frames((40, 0.0, 0.3, 0.05)))
    assert r["failure"] == "NOT_DETECTED:turn_left"


def test_mirrored_input_swaps_directions(monkeypatch):
    monkeypatch.setattr(rl, "MIRRORED_INPUT", True)
    m = frames(NEUTRAL, LEFT, NEUTRAL)
    r = rl.evaluate_challenge(challenge("turn_right"), m)
    assert r["passed"]


def test_frames_without_face_are_ignored():
    m = frames(NEUTRAL) + [None, None] + frames(LEFT, NEUTRAL)
    r = rl.evaluate_challenge(challenge("turn_left"), m, timestamps=[i * 0.1 for i in range(len(m))])
    assert r["passed"] and r["matched"][0]["at_sec"] == 0.8


def test_issue_challenge_shape():
    c = rl.issue_challenge()
    assert len(c["steps"]) == rl.CHALLENGE_STEPS == len(set(c["steps"]))
    assert any(s in rl.TURN_ACTIONS for s in c["steps"])
    assert len(c["prompts"]) == len(c["steps"]) and c["nonce"]


def test_nonce_is_single_use():
    c = rl.issue_challenge()
    first, err = rl.consume_challenge(c["nonce"])
    assert first and err is None
    assert rl.consume_challenge(c["nonce"]) == (None, "CHALLENGE_INVALID")
    assert rl.consume_challenge("made-up") == (None, "CHALLENGE_INVALID")


def test_expired_nonce_rejected(monkeypatch):
    c = rl.issue_challenge()
    real = time.time
    monkeypatch.setattr(rl.time, "time", lambda: real() + rl.CHALLENGE_TTL_SEC + 1)
    assert rl.consume_challenge(c["nonce"]) == (None, "CHALLENGE_INVALID")
