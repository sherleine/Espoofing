from app.verification.risk_engine import RiskState

OK = {"face_count": "ONE", "identity": "MATCH", "continuity": "STABLE",
      "pad": "LOW_RISK", "absent_windows": 0}


def w(**changes):
    return {**OK, **changes}


def run(*windows):
    state = RiskState()
    return [state.assess(x) for x in windows]


def test_all_good_is_normal():
    out = run(OK, OK, OK)
    assert [o["state"] for o in out] == ["NORMAL"] * 3
    assert out[-1]["recommended_action"] == "CONTINUE"


def test_single_weak_signal_is_only_suspicious():
    for bad in (w(pad="ELEVATED"), w(pad="HIGH_RISK"), w(face_count="MULTIPLE"),
                w(identity="MISMATCH"), w(continuity="CHANGED")):
        out = run(OK, bad)
        assert out[-1]["state"] == "SUSPICIOUS", bad


def test_suspicion_relaxes_after_history_clears():
    out = run(w(pad="ELEVATED"), *[OK] * 6)
    assert out[1]["state"] == "SUSPICIOUS"
    assert out[-1]["state"] == "NORMAL"


def test_persistent_pad_high_becomes_high_risk():
    out = run(w(pad="HIGH_RISK"), w(pad="HIGH_RISK"))
    assert out[-1]["state"] == "HIGH_RISK"
    assert "SPOOF_SUSPECTED" in out[-1]["reasons"]
    assert out[-1]["recommended_action"] == "FLAG_FOR_REVIEW"


def test_pad_high_counts_only_scored_windows():
    # an unscorable window in between does not dilute the evidence
    out = run(w(pad="HIGH_RISK"), w(pad="INSUFFICIENT_QUALITY"), w(pad="HIGH_RISK"))
    assert out[-1]["state"] == "HIGH_RISK"


def test_identity_mismatch_two_of_three_checked():
    out = run(w(identity="MISMATCH"), w(identity="UNKNOWN"), w(identity="MISMATCH"))
    assert out[-1]["state"] == "HIGH_RISK"


def test_two_independent_signals_in_one_window_is_high_risk():
    out = run(OK, w(identity="MISMATCH", continuity="CHANGED"))
    assert out[-1]["state"] == "HIGH_RISK"


def test_multiple_faces_needs_persistence():
    out = run(w(face_count="MULTIPLE"), w(face_count="MULTIPLE"), w(face_count="MULTIPLE"))
    assert [o["state"] for o in out] == ["SUSPICIOUS", "SUSPICIOUS", "HIGH_RISK"]


def test_absence_is_suspicious_not_high():
    none = w(face_count="NONE", identity="UNKNOWN", continuity="UNKNOWN", pad="NOT_RUN")
    out = run({**none, "absent_windows": 1}, {**none, "absent_windows": 2}, {**none, "absent_windows": 3})
    assert [o["state"] for o in out] == ["NORMAL", "SUSPICIOUS", "SUSPICIOUS"]
    assert out[-1]["reasons"] == ["CANDIDATE_ABSENT"]


def test_persistent_unscorable_video_is_suspicious():
    out = run(*[w(pad="INSUFFICIENT_QUALITY")] * 4)
    assert out[2]["state"] == "NORMAL"
    assert out[3]["state"] == "SUSPICIOUS" and out[3]["reasons"] == ["POOR_VIDEO_QUALITY"]


def test_next_window_is_sooner_when_suspicious():
    out = run(OK, w(pad="ELEVATED"))
    assert out[0]["next_window_sec"] > out[1]["next_window_sec"]
