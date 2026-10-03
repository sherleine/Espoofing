# app/verification/risk_engine.py
"""
Combines exam-time signals into NORMAL | SUSPICIOUS | HIGH_RISK with simple
rules a reviewer can follow. Rules look at the last HISTORY windows, so one
bad window alone never gives HIGH_RISK (unless several independent signals
fire together), and the state relaxes once the evidence ages out.
It recommends an action; the exam platform decides what happens.
"""

import random
from collections import deque

# Placeholders, calibrate on real exam sessions.
# (k, n) = signal seen in at least k of the last n windows.
HISTORY = 6
ABSENT_SUSPICIOUS_WINDOWS = 2           # consecutive windows with no face
MULTI_FACE_HIGH = (3, 6)
MISMATCH_HIGH = (2, 3)                  # counted over identity-checked windows only
PAD_HIGH = (2, 4)                       # counted over PAD-scored windows only
POOR_QUALITY_SUSPICIOUS_WINDOWS = 4     # face present but never scorable (evasion by darkness/blur?)
COMBINED_HIGH = 2                       # independent attack signals in one window -> HIGH_RISK

POLICY = {
    "NORMAL": "CONTINUE",
    "SUSPICIOUS": "INCREASE_MONITORING",
    "HIGH_RISK": "FLAG_FOR_REVIEW",
}

# Seconds until the next window, by state (server-driven, with jitter so the
# schedule can't be predicted).
NEXT_WINDOW_SEC = {"NORMAL": 30, "SUSPICIOUS": 10, "HIGH_RISK": 10}
NEXT_WINDOW_JITTER = 0.3

REASONS = {
    "CANDIDATE_ABSENT": "No face visible for several windows",
    "MULTIPLE_FACES": "More than one face in view",
    "IDENTITY_MISMATCH": "Face does not match the registered candidate",
    "IDENTITY_CHANGED": "Face changed compared to the previous check",
    "SPOOF_SUSPECTED": "Frames look like a photo/screen presented to the camera",
    "POOR_VIDEO_QUALITY": "Face could not be analysed (lighting/blur) for several windows",
}


def _count(history, key, values, n=None, only=None):
    """How many of the last n windows have history[key] in values.
    `only` restricts to windows where that key is in `only` (e.g. scored)."""
    windows = list(history)
    if only is not None:
        windows = [w for w in windows if w[key] in only]
    if n:
        windows = windows[-n:]
    return sum(w[key] in values for w in windows)


def _flags(window):
    """Attack-related signals raised by one window."""
    flags = set()
    if window["face_count"] == "MULTIPLE":
        flags.add("MULTIPLE_FACES")
    if window["identity"] == "MISMATCH":
        flags.add("IDENTITY_MISMATCH")
    if window["continuity"] == "CHANGED":
        flags.add("IDENTITY_CHANGED")
    if window["pad"] in ("ELEVATED", "HIGH_RISK"):
        flags.add("SPOOF_SUSPECTED")
    return flags


class RiskState:
    def __init__(self):
        self.history = deque(maxlen=HISTORY)
        self.state = "NORMAL"

    def assess(self, signals):
        """Feed one window's signals; returns the assessment dict."""
        self.history.append(signals)
        h = self.history
        window_flags = _flags(signals)

        # anything raised by a window still in the history keeps the state at
        # least SUSPICIOUS, so it relaxes only once the evidence has aged out
        suspicious = set().union(*(_flags(w) for w in h))
        if signals.get("absent_windows", 0) >= ABSENT_SUSPICIOUS_WINDOWS:
            suspicious.add("CANDIDATE_ABSENT")
        recent_poor = list(h)[-POOR_QUALITY_SUSPICIOUS_WINDOWS:]
        if (len(recent_poor) == POOR_QUALITY_SUSPICIOUS_WINDOWS
                and all(w["pad"] == "INSUFFICIENT_QUALITY" for w in recent_poor)):
            suspicious.add("POOR_VIDEO_QUALITY")

        high = set()
        k, n = MULTI_FACE_HIGH
        if _count(h, "face_count", {"MULTIPLE"}, n) >= k:
            high.add("MULTIPLE_FACES")
        k, n = MISMATCH_HIGH
        if _count(h, "identity", {"MISMATCH"}, n, only={"MATCH", "MISMATCH"}) >= k:
            high.add("IDENTITY_MISMATCH")
        k, n = PAD_HIGH
        if _count(h, "pad", {"HIGH_RISK"}, n, only={"LOW_RISK", "ELEVATED", "HIGH_RISK"}) >= k:
            high.add("SPOOF_SUSPECTED")
        if len(window_flags) >= COMBINED_HIGH:
            high.update(window_flags)

        if high:
            state, reasons = "HIGH_RISK", sorted(high)
        elif suspicious:
            state, reasons = "SUSPICIOUS", sorted(suspicious)
        else:
            state, reasons = "NORMAL", []

        changed = state != self.state
        self.state = state
        base = NEXT_WINDOW_SEC[state]
        return {
            "state": state,
            "state_changed": changed,
            "reasons": reasons,
            "reason_text": [REASONS[r] for r in reasons],
            "recommended_action": POLICY[state],
            "next_window_sec": round(base * random.uniform(1 - NEXT_WINDOW_JITTER, 1 + NEXT_WINDOW_JITTER), 1),
        }
