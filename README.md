# Espoofing

Face verification and presentation-attack detection service for exam registration and monitoring.

## Production API

POST /verify-email-photo

Form fields:
- email — candidate email
- examId — exam identifier
- email_photo — trusted profile/reference photo
- live_photo — selfie video; the field name is kept for frontend compatibility

Registration performs profile face detection, video sampling, face presence and consistency checks, identity matching, liveness/movement checks, passive presentation-attack detection, and the final verification decision.

Other endpoints:
- GET /verification/challenge — random ordered liveness actions and a single-use nonce
- POST /register — profile_photo, video and a required challenge_nonce; returns reference_embedding when verified, for the caller to store
- POST /verify — reference_embedding (JSON from /register) and video; passive, or with a challenge_nonce for an active check
- POST /exam/session/start, POST /exam/session/{id}/window, GET /exam/session/{id}, POST /exam/session/{id}/end — passive exam monitoring with a NORMAL / SUSPICIOUS / HIGH_RISK risk state

The production registration sampler uses 10 frames. This was selected after 100-request benchmarks covering genuine, wrong-person, and spoof cases. All 300 tested requests produced the expected decision, with roughly 20% lower average latency than the previous 15-frame configuration.

## Project structure

server.py
requirements.txt
app/
  monitor/
    session_store.py
  verification/
    anti_spoof.py
    face_tracking.py
    face_verification.py
    faces.py
    main.py
    registration_liveness.py
    risk_engine.py
    router.py
    service.py
    view.py

The repository contains production code only. Benchmark scripts, benchmark output, temporary profilers, demo utilities, and test-only files have been removed.

## Running locally

Install dependencies:

    pip install -r requirements.txt

Start the API:

    python -m uvicorn server:app --port 8000

Run a single worker: liveness challenges and exam sessions are kept in process memory.

Endpoint:

    http://127.0.0.1:8000/verify-email-photo

## Configuration

- VERIFICATION_SERVICE_KEY — optional service-key authentication.
- VERIFICATION_REQUIRE_CHALLENGE — set to 1 to require a liveness challenge.
- VERIFICATION_MIRRORED_INPUT — set to 1 when the recorded video is mirrored.
- PAD_ENFORCE_REGISTRATION — set to 0 to report PAD without rejecting high-risk results.
- PAD_MODEL_PATH — optional path to the PAD ONNX model.
- VERIFICATION_KEEP_UPLOADS — set to 1 only when uploaded biometric files must be retained.

Uploads are deleted after processing by default.

## Response

The endpoint returns verified, is_match, reason_code, reason, failed_checks, identity/similarity, liveness, video, PAD, and a reference_embedding on successful registration.

Important failure codes include FACE_MISMATCH, SPOOF_SUSPECTED, VIDEO_TOO_SHORT, VIDEO_IS_IMAGE, and NO_FACE_IN_PROFILE.

## Limitations

Stream integrity / virtual-camera injection detection is not currently implemented. Passive PAD detects faces re-captured from a photo or screen; a recording or real-time deepfake fed in through a virtual camera bypasses it. Deepfake detection is not implemented either.

## Production note

The implementation uses InsightFace pretrained models and an InsightFace liveness model. Check the applicable model licences before commercial deployment.
