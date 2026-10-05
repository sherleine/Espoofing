# ESpoofing

Face verification and presentation-attack detection service for remote exam registration and passive exam monitoring.

## Authentication

The .NET backend supplies the **trusted profile photo** from its own records, so only the .NET backend may call this service.
Set `VERIFICATION_SERVICE_KEY` on the Python service and send it as the `X-Service-Key` header on every call.
Without the variable, the endpoints are open (development only; a warning is logged). Keep the service off the public internet too.

## 3.1 Registration — `POST /verify-email-photo` (existing, unchanged fields)

Registration is designed as **one API call**. The integrating application sends the candidate details, trusted profile photo and live registration video to `/verify-email-photo`. The service performs the registration checks internally and returns one result.

| Form field | |
|---|---|
| `email`, `examId` | as before |
| `email_photo` | trusted profile photo (**from the .NET database, never from the browser**) |
| `live_photo` | the selfie **video** (name kept for compatibility) |
| `challenge_nonce` | optional — only needed when the client chooses the prompted challenge flow |

The normal one-call flow does **not** require a preliminary challenge request. Without a nonce, the uploaded video is checked for face presence/consistency, natural facial movement, identity match and passive presentation-attack detection in the same request.

The response keeps every existing key (`verified`, `is_match`, `reason_code`, `reason`, `failed_checks`, `live`,
`identity_match`, `similarity`, `identity`, `liveness`, `video`) and adds:

* `anti_spoof` — passive PAD result: `status`, `spoof_score`, `frames_scored`, `frames_skipped`, `frame_scores`.
  There is no `confidence`: the scores are not calibrated yet, so a confidence number would claim more than we know.
* `challenge` — only when a nonce was sent: steps, `passed`, `failure`, matched/detected actions with times

New `reason_code` values (the frontend should show `reason` for unknown codes):

| Code | When |
|---|---|
| `CHALLENGE_FAILED` | requested actions not seen, wrong order, or unrequested head turns |
| `CHALLENGE_INVALID` | nonce unknown, expired (3 min) or already used — nonces are single use |
| `CHALLENGE_MISSING` | no nonce while `VERIFICATION_REQUIRE_CHALLENGE=1` |
| `SPOOF_SUSPECTED` | PAD status `HIGH_RISK` (switch off with `PAD_ENFORCE_REGISTRATION=0`) |

Without a nonce the service uses the existing natural movement check. This keeps registration to one API call while still providing a liveness signal and PAD result. This movement check is weaker than a prompted challenge because a replay video can contain natural movement.

### 3.2 Optional prompted liveness — `GET /verification/challenge`

The challenge endpoint is retained for compatibility and for clients that want a stronger active-liveness flow. It is **not required** for the normal registration integration.

```json
{
  "nonce": "Q2x4…",
  "steps": ["turn_left", "blink", "turn_right"],
  "prompts": ["Turn your head to your left, then back to the centre",
              "Close your eyes for a moment, then open them",
              "Turn your head to your right, then back to the centre"],
  "step_seconds": 2.5,
  "issued_at": 1790000000.0,
  "expires_at": 1790000180.0
}
```

If the client uses this optional flow, it requests the challenge first, shows each prompt while recording, then uploads the video with the returned `nonce` to the same `/verify-email-photo` endpoint.

**Left/right:** "left" means the candidate's own left. This assumes the recorded video is **not mirrored**,
which is what `MediaRecorder` produces even when the on-screen preview is mirrored with CSS. If your client records a mirrored stream,
set `VERIFICATION_MIRRORED_INPUT=1`. The webcam demo shows the live yaw so you can check this.

### 3.3 Exam monitoring (passive)

```text
POST /exam/session/start             email, examId, email_photo   -> session_id + capture hints
POST /exam/session/{id}/window       frames=<jpg> (repeat, in order)  OR  clip=<short video>
GET  /exam/session/{id}              state, highest state, event log
POST /exam/session/{id}/end          final summary, frees the session
```

Suggested client cadence (returned by `start`, all configurable): **8 frames at ~250 ms intervals (a 2 s window)**.
After each window, wait `next_window_sec` from the response before sending the next one. It is about 30 s when NORMAL and
about 10 s when SUSPICIOUS, with ±30 % random jitter so the schedule can't be predicted.

Window response (shortened):

```json
{
  "ok": true, "window": 12,
  "state": "SUSPICIOUS", "state_changed": true,
  "reasons": ["SPOOF_SUSPECTED"],
  "reason_text": ["Frames look like a photo/screen presented to the camera"],
  "recommended_action": "INCREASE_MONITORING",
  "next_window_sec": 9.4,
  "signals": {
    "face_count": "ONE", "identity": "MATCH", "continuity": "STABLE",
    "reference_similarity": 0.71, "previous_similarity": 0.83,
    "face_returned": false, "absent_windows": 0,
    "pad": {"status": "ELEVATED", "spoof_score": 0.27, "frames_scored": 8, "frames_skipped": {}}
  },
  "faces": {"frames_sampled": 8, "frames_with_face": 8, "face_ratio": 1.0, "multi_face_ratio": 0.0},
  "processing_ms": 1180
}
```

`recommended_action`: `CONTINUE` → `INCREASE_MONITORING` → `FLAG_FOR_REVIEW`. What happens next (proctor alert, re-verification
with the registration flow, review after the exam) is **exam policy, decided by the platform**, not by this service.

Sessions live in memory in a single process. With several uvicorn/gunicorn workers, either route each session to one worker
or move `_sessions` and the challenge store to Redis (both are small dicts, marked in the code).

---

## 4. What each layer protects against

| Attack | Type | Main defence here | Limitations |
