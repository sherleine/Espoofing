# Face verification & anti-spoofing (prototype)

Refactor and extension of the existing `app/verification` module (router → view → service).

| Situation | What runs | Candidate has to… |
|---|---|---|
| **Registration / initial verification** | Active liveness (random ordered challenge) + face match + passive PAD | follow 3 short prompts |
| **Exam** | Passive PAD + face tracking + identity continuity → risk engine | nothing — just sit the exam |

> **No approach here gives 100% protection.** The system collects evidence and reports a risk level.
> It never fails a candidate on its own during the exam; a person reviews flagged sessions.
>
> **Stream integrity / virtual-camera injection detection is not currently implemented.** Video fed in through a
> virtual camera (a recording or a real-time deepfake) bypasses passive PAD, which only detects faces re-captured
> from a photo or screen. Deepfake detection is not implemented either. See section 4.

---

## 1. Quick start

```bash
pip install -r requirements.txt          # Python 3.10+; CPU only is fine
python -m pytest -q                      # 69 tests, ~10-15 min on CPU (models download on first run)
python -m uvicorn server:app --port 8000 # standalone API (see section 3), docs at /docs
python tools/webcam_demo.py pad          # live anti-spoofing score on your webcam
```

On first use InsightFace downloads `buffalo_l` (~280 MB) and the PAD model `liveness.onnx` (1.5 MB) to `~/.insightface/`.
For offline servers, copy those folders across (set `PAD_MODEL_PATH` if the PAD model is elsewhere).

**Integrating into the real application:** copy `app/verification/` across, `pip install` the requirements, and keep including
`app.verification.router.router` as today. **Do not copy `app/monitor/session_store.py`**: it is a stand-in so this prototype runs on its own.
`server.py` exists only for standalone testing.

> **Run one worker process.** Challenges and exam sessions are kept in process memory. With several uvicorn/gunicorn
> workers, a challenge issued by one worker is unknown to the next (`CHALLENGE_INVALID`) and exam windows can land on a
> worker that has no session. Moving both stores to Redis fixes this; it is not done yet.

---

## 2. Architecture

```text
router.py ─► view.py ─► service.py (compatibility wrapper: verify_faces)
                │
                ▼
             main.py  ── verification flow: which checks run, in what order
                │
             faces.py ── decode input, detect faces ONCE per frame (InsightFace)
                │
   ┌────────────┼──────────────┬─────────────────┬──────────────┐
   ▼            ▼              ▼                 ▼              ▼
face_verification  registration_liveness  face_tracking   anti_spoof    risk_engine
identity only      active challenge       face count,     passive PAD   NORMAL /
(embeddings)       (registration only)    continuity      (one model)   SUSPICIOUS /
                                                                         HIGH_RISK
```

| Module | Responsibility | Not responsible for |
|---|---|---|
| `main.py` | Registration flow, exam flow, exam session store | Detection, PAD, tracking or risk details |
| `faces.py` | Loads InsightFace once, decodes video/images, detects faces, embeddings, face quality, head pose | Decisions |
| `face_verification.py` | Cosine similarity live vs profile, match decision | Liveness |
| `registration_liveness.py` | Challenge issue/consume (single-use nonce), ordered action check, legacy movement check | Exam time (never used there) |
| `face_tracking.py` | Face presence/count, same-person consistency, exam continuity (`ExamTracker`) | Spoofing |
| `anti_spoof.py` | Passive PAD: `detect_spoof(frames, faces)` with one ONNX model | Injection / virtual cameras |
| `risk_engine.py` | Explainable k-of-n rules over recent windows → state, reasons, recommended action | Final decisions (policy is the exam platform's) |

**Replacing the PAD model:** change `load_model()` and `live_probabilities()` in `anti_spoof.py`.
`detect_spoof()` and everything that calls it stay the same.

**Entry points** (Python): `verify_registration(profile_photo_path, video_path, challenge_nonce)` for registration;
`start_exam_session`, `process_exam_window`, `get_exam_status`, `end_exam_session` for the exam. `service.verify_faces()`
is kept for existing callers.

---

## 3. API (called by the .NET backend)

### Authentication
The .NET backend supplies the **trusted profile photo** from its own records, so only the .NET backend may call this service.
Set `VERIFICATION_SERVICE_KEY` on the Python service and send it as the `X-Service-Key` header on every call.
Without the variable, the endpoints are open (development only; a warning is logged). Keep the service off the public internet too.

### Flow

```text
1. GET  /verification/challenge   -> nonce + steps; the frontend shows the prompts while recording
2. POST /register                 profile photo + challenge video + nonce
                                  -> verified + reference_embedding          (.NET stores it)
3. POST /verify                   reference_embedding + short video (+ optional nonce)
                                  -> verified: is this still the registered person?
4. POST /exam/session/start       reference_embedding -> session_id, then exam windows (3.5)
```

The reference embedding comes from the **live registration video**, not the official photo: same camera and similar
lighting as later checks, so matches are more reliable, and later checks compare against the face that passed liveness
and anti-spoofing.

### 3.1 Register — `POST /register`

| Form field | |
|---|---|
| `email`, `examId` | candidate and exam |
| `profile_photo` | trusted profile photo (**from the .NET database, never from the browser**) |
| `video` | the challenge video |
| `challenge_nonce` | **required**, from `GET /verification/challenge` |

Same checks and response as `/verify-email-photo` below, but the challenge is mandatory: a missing or empty nonce gives
`CHALLENGE_MISSING` instead of falling back to the weak movement check. When `verified` is true the response also holds:

```json
"reference_embedding": {"model": "insightface/buffalo_l/w600k_r50", "vector": [0.0123, -0.0456, "… 512 numbers"]}
```

**.NET stores this object as-is** (e.g. a JSON column next to the candidate and exam) and sends it back unchanged as
the `reference_embedding` form field. It is biometric data: protect and delete it like the photo. The `model` name
matters: if the face model is ever replaced, old embeddings are rejected with `REFERENCE_MODEL_MISMATCH` and the
candidate registers again.

### 3.2 Verify — `POST /verify`

| Form field | |
|---|---|
| `email`, `examId` | candidate and exam |
| `reference_embedding` | the JSON object saved from `/register`, as text |
| `video` | short video of the candidate (a few seconds) |
| `challenge_nonce` | *optional*: with a nonce the candidate performs the actions (**challenge** mode); without, they only look at the camera (**passive** mode) |

Both modes check: exactly one consistent face, identity against the saved embedding, and passive PAD. Challenge mode also
checks the requested actions. Response: `verified`, `reason_code`, `reason`, `failed_checks`, `mode`
(`passive` / `challenge`), `identity_match`, `similarity`, `identity`, `video`, `anti_spoof`, and `challenge` when a
nonce was sent. A different person gives `IDENTITY_MISMATCH`.

### 3.3 Legacy registration — `POST /verify-email-photo` (existing, unchanged fields)

| Form field | |
|---|---|
| `email`, `examId` | as before |
| `email_photo` | trusted profile photo (**from the .NET database, never from the browser**) |
| `live_photo` | the selfie **video** (name kept for compatibility) |
| `challenge_nonce` | *new, optional* — from `GET /verification/challenge` |

The response keeps every existing key (`verified`, `is_match`, `reason_code`, `reason`, `failed_checks`, `live`,
`identity_match`, `similarity`, `identity`, `liveness`, `video`) and adds:

* `anti_spoof` — passive PAD result: `status`, `spoof_score`, `frames_scored`, `frames_skipped`, `frame_scores`.
  There is no `confidence`: the scores are not calibrated yet, so a confidence number would claim more than we know.
* `challenge` — only when a nonce was sent: steps, `passed`, `failure`, matched/detected actions with times
* `reference_embedding` — only when `verified` is true (see 3.1)

New `reason_code` values (the frontend should show `reason` for unknown codes):

| Code | When |
|---|---|
| `CHALLENGE_FAILED` | requested actions not seen, wrong order, or unrequested head turns |
| `CHALLENGE_INVALID` | nonce unknown, expired (3 min) or already used — nonces are single use |
| `CHALLENGE_MISSING` | no nonce while `VERIFICATION_REQUIRE_CHALLENGE=1` |
| `SPOOF_SUSPECTED` | PAD status `HIGH_RISK` (switch off with `PAD_ENFORCE_REGISTRATION=0`) |
| `IDENTITY_MISMATCH` | `/verify`: the video does not match the saved embedding |
| `INVALID_REFERENCE_EMBEDDING` | `reference_embedding` is not valid JSON / not 512 numbers |
| `REFERENCE_MODEL_MISMATCH` | `reference_embedding` was made with a different face model |
| `NO_REFERENCE` | exam start without `reference_embedding` or photo |
| `VIDEO_IS_IMAGE` | a photo was uploaded in the video field |

Without a nonce the old "did the head/face move at all" check runs, with the same results as before
(verified by `tests/test_legacy_compat.py` against a frozen copy of the original `service.py`).
That check is **weak**: any recording of the candidate passes. Turn on `VERIFICATION_REQUIRE_CHALLENGE=1` once the frontend shows prompts.

### 3.4 Challenge — `GET /verification/challenge`

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

Frontend: show each prompt for `step_seconds` while recording (≈ 8 s in total), then upload the video with the `nonce`.
The steps are 3 distinct actions out of {turn_left, turn_right, blink, open_mouth}, in random order, and there is always at least one head turn.

**Head turns don't need a precise angle.** A gentle turn of about 10° counts in the challenge, and 5° in the simple
check; a still photo moves the angle estimate by less than 1°. Turning far to the side is fine too: frames turned more
than 35° are left out of the identity and same-person checks, because face recognition is unreliable on side profiles.

**Left/right:** "left" means the candidate's own left. This assumes the recorded video is **not mirrored**,
which is what `MediaRecorder` produces even when the on-screen preview is mirrored with CSS. If your client records a mirrored stream,
set `VERIFICATION_MIRRORED_INPUT=1`. The webcam demo shows the live yaw so you can check this.

### 3.5 Exam monitoring (passive)

```text
POST /exam/session/start             email, examId, reference_embedding -> session_id + capture hints
                                     (email_photo instead: fallback for candidates registered earlier)
POST /exam/session/{id}/window       frames=<jpg> (repeat, in order)  OR  clip=<short video>
GET  /exam/session/{id}              state, highest state, event log
POST /exam/session/{id}/end          final summary, frees the session
```

A window needs at least 3 frames (or a clip): fewer gives `TOO_FEW_FRAMES`, because anti-spoofing
needs several frames.
Suggested client cadence (returned by `start`, all configurable): **8 frames at ~250 ms intervals (a 2 s window)**.
After each window, wait `next_window_sec` from the response before sending the next one. It is about 30 s when NORMAL and
about 10 s when SUSPICIOUS, with ±30 % random jitter so the schedule can't be predicted.
The start response says which reference is used: `"reference": "registration_embedding"` or `"profile_photo"`.

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
|---|---|---|---|
| Printed photo | physical presentation | registration challenge; passive PAD | high-quality prints/cut-outs can beat a single PAD model |
| Photo/video on phone, tablet, monitor | physical replay | passive PAD (screen artefacts); challenge at registration | large high-DPI screens at the right distance are hard |
| Prerecorded video of the candidate | physical replay **or** injection | challenge (random order + single-use nonce + no unrequested turns); PAD if shown on a screen | a recording of every possible sequence, played through a virtual camera, gets past both |
| Deepfake / face swap | almost always **injection** (virtual camera) | identity checks partly; **not covered yet** | needs stream integrity + deepfake detection (planned) |
| Virtual camera / injected stream | injection | **not covered yet** | PAD cannot see it: the frames were never re-captured. A digitally generated video of a photo scores *genuine* (spoof ≈ 0.01) |
| Second person in view | proctoring | face count, persistence rule; small background faces (<4 % area) ignored | a person outside the camera view |
| Someone else takes over mid-exam | proctoring | identity vs reference every window; continuity vs previous window; `face_returned` flag after absence | twins / very similar faces; very poor lighting |

**PAD vs injection:** passive PAD analyses a *physical scene* filmed by a camera. Injection attacks (OBS Virtual Camera,
real-time face-swap apps) replace the camera feed itself, so PAD has nothing to detect. Stream-integrity checks are a separate workstream
(section 9). The risk engine has no integrity signal yet; one will be added when the checks exist.

---

## 5. Testing

### Automated
```bash
python -m pytest -q
```
* `test_legacy_compat.py`: the new code gives **identical** results to the original `service.py` on 9 cases
  (same person, other person, group, mid-video swap, short, no face, corrupt video, blank and missing profile).
* `test_registration_liveness.py`: ordering, wrong order, missing action, "do every movement" recording, spontaneous blinks,
  mirroring, nonce single-use and expiry.
* `test_risk_engine.py`: one weak signal → only SUSPICIOUS; persistence → HIGH_RISK; relaxing back to NORMAL.
* `test_anti_spoof.py`: status from the median spoof score, quality-skipped frames are counted but never scored as attacks,
  too few usable frames → INSUFFICIENT_QUALITY, and scores identical to InsightFace's own liveness wrapper.
* `test_register_verify.py`: /register needs a challenge, a verified registration returns the embedding, passive and
  challenge /verify against it, wrong person, broken or wrong-model embeddings, exam start with the embedding.
* `test_registration_flow.py`: side-profile frames are not reported as a different person.
* `test_exam_flow.py`, `test_api.py`: end to end through the HTTP API, auth, upload deletion.

Synthetic videos are generated from InsightFace's sample photos, so the tests need no personal data. They prove the **logic**, not
the detection quality. Detection quality needs real recordings (below).

### Webcam demo — try attacks live
```bash
python tools/webcam_demo.py pad                         # per-frame spoof score + face count + yaw
python tools/webcam_demo.py exam --profile me.jpg       # silent exam monitoring, window every 5 s
python tools/webcam_demo.py register --profile me.jpg   # SPACE = guided challenge, then full verification
```
Press **R** to use the current webcam frame as the profile photo.
Press **B / P / M / S / O** to save a 3 s labelled clip (bona_fide / print / mobile / screen / other) into `dataset/`.

### Attack test matrix (record each with several people, devices and lighting)
1. **Bona fide:** different laptops and webcams; dark, back-lit and bright rooms; glasses; beards; head coverings; varied skin tones; normal exam behaviour (looking down, writing, drinking)
2. **Print:** matte and glossy, A4/A5, held still, moving, bent, eyes cut out
3. **Phone:** photo, then video of the candidate, at several brightness levels and distances
4. **Tablet / monitor:** photo and video replay
5. **Second person:** beside, behind, leaning in; a photo on the wall
6. **Person swap:** leaves and someone else sits down
7. **Registration:** a generic recording, the right actions in the wrong order, a reused nonce
8. *(next workstream)* OBS Virtual Camera playing a recording; real-time face swap (lab only, with consent)

### Measure (ISO/IEC 30107-3)
```bash
python tools/evaluate_pad.py dataset --csv results.csv
```
Reports **APCER per attack species** and the worst case, **BPCER**, ACER (for comparison only), EER, and
**BPCER at worst-case APCER ≤ 1 % / 5 %**, plus how many presentations were unscorable. For the exam, also track
**false alerts per exam-hour**, time-to-detect per attack, and results per device/lighting/demographic group.

### Calibrating thresholds
Every threshold is a **placeholder** and marked as such in the code:
`anti_spoof.SPOOF_ELEVATED / SPOOF_HIGH`, the quality gate, `risk_engine` k-of-n rules, and the challenge angles/ratios.
1. Record bona fide sessions on real devices and run `evaluate_pad.py`. This shows the genuine score distribution.
2. Pick `SPOOF_HIGH` for an acceptable BPCER (false flags), then read off the APCER per species at that threshold.
3. Run exam monitoring in "report only" mode (log the states, act on nothing) on real exams and count false alerts per hour.
   Then tune the risk rules.
4. Re-calibrate whenever the model, frame sampling or client capture settings change.

---

## 6. Configuration

| Environment variable | Default | Meaning |
|---|---|---|
| `VERIFICATION_SERVICE_KEY` | *(empty)* | required `X-Service-Key`; empty = no auth (dev only) |
| `VERIFICATION_REQUIRE_CHALLENGE` | `0` | `1` = registration without a nonce fails |
| `VERIFICATION_MIRRORED_INPUT` | `0` | `1` if the client records a mirrored stream |
| `PAD_ENFORCE_REGISTRATION` | `1` | PAD `HIGH_RISK` fails registration with `SPOOF_SUSPECTED` |
| `PAD_MODEL_PATH` | `~/.insightface/addons/liveness.onnx` | PAD model location (downloaded and checksum-verified if missing) |
| `VERIFICATION_KEEP_UPLOADS` | `0` | `1` keeps uploaded photo/video files (e.g. to build a test set) |

Other settings are constants at the top of each module.

---

## 7. Performance (measured: i5-13450HX laptop, CPU only)

| Step | Time |
|---|---|
| Face detection, 640 px | ~80 ms / frame |
| 3D landmarks + pose | ~67 ms / face |
| Identity embedding (ArcFace r50) | ~220 ms / face |
| PAD model | ~11 ms / face |
| **Registration**, legacy (25 frames) | ~13 s (about the same as before) |
| **Registration** with challenge (40 frames) | ~16 s |
| **Exam window** (8 frames: detection + PAD on all, 1 embedding) | ~1.2 s → ≈ 4 % of a core per candidate at one window / 30 s |

Expensive models run only where they are needed: no landmarks during the exam, and one embedding per window.
For many concurrent candidates, run exam windows in a worker queue rather than in the API process.

---

## 8. ⚠ Licensing — must be checked before production

InsightFace's code is MIT, but its documentation says its **pretrained models are for non-commercial research only**.
That covers `buffalo_l`, **which the current production code already uses**, and the PAD model `liveness.onnx`.
Options: buy a commercial licence from InsightFace, or swap the models. The PAD model can be swapped in `anti_spoof.py` alone.
A candidate is MiniFASNet (Silent-Face-Anti-Spoofing, Apache-2.0 per its repository — verify), which would need converting to ONNX.
Public anti-spoofing datasets (CelebA-Spoof, OULU-NPU, SiW, Replay-Attack) are mostly academic licences too,
which matters if you fine-tune on them.

## Privacy

* Logs contain no email, images or embeddings: registrations are logged by exam ID, exam windows by session ID,
  and only windows that change or raise the risk state are logged.
* The registration response returns the candidate's face embedding (512 numbers) for .NET to store. Treat it as
  biometric data: encrypt at rest, restrict access, and delete it with the candidate's other data.
* Uploaded photos and videos are now **deleted after verification**. They used to be kept forever in `uploads/verification/`.
* Exam frames sent as images are processed in memory. A window sent as a video clip is written to a temporary file
  for decoding and deleted straight away. Sessions keep only one reference embedding, the risk history and an event log.
* Face images and embeddings are biometric data (GDPR Art. 9 where it applies). You need a legal basis and a notice to candidates,
  short retention periods, and **human review before any consequence**.

---

## 9. Not done yet (next phases)

1. **Supporting PAD signals:** a phone/screen object detector (with "device around the face" logic), face-vs-background
   motion coherence, planar-motion test (photos), loop/duplicate-frame detection, frequency/moiré features, and optionally rPPG.
   Each would be one more backend or signal; add them one at a time and measure with `evaluate_pad.py`.
2. **Stream integrity / injection (separate workstream):** browser telemetry (`enumerateDevices` labels such as "OBS Virtual Camera",
   track settings and capabilities, frame-timing jitter, `devicechange` events), camera consistency with registration, and a
   screen-colour reflection check on escalation. Browser-side signals can be faked; a desktop/lockdown client is the only way to get
   much stronger guarantees.
3. **Deepfake detection** on escalated windows only (generalises poorly; use as a weak signal).
4. A stronger or fine-tuned PAD model trained on your own recorded data, then score calibration and learned fusion in the risk engine.
5. Redis for sessions/challenges, a worker queue, and a reviewer view of the event log with evidence thumbnails.
