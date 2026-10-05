# Face Verification & Anti-Spoofing

Prototype verification service for remote exam registration and passive exam monitoring.

## Integration rule

The existing API routes are kept unchanged. Registration is designed so the integrating application can complete the normal verification flow with **one API call**:

```text
POST /verify-email-photo
```

The request contains:

- `email`
- `examId`
- `email_photo` — trusted profile photo from the backend/database
- `live_photo` — registration selfie video (field name kept for compatibility)
- `challenge_nonce` — optional

The service performs the verification internally and returns one result.

## Registration flow

```text
POST /verify-email-photo
        |
        v
Read profile photo + live video
        |
        v
Reference face detection
        |
        v
Registration video frame extraction
        |
        +--------------------+
        |                    |
        v                    v
Identity matching       Liveness check
        |                    |
        +---------+----------+
                  |
                  v
            Passive PAD
                  |
                  v
          Final verification
                  |
                  v
             One response
```

The normal one-call flow does **not** require a preliminary challenge request. Without a nonce, the service uses the existing natural facial-movement check together with identity verification and passive presentation-attack detection.

The optional challenge endpoint is retained for clients that want prompted active liveness:

```text
GET /verification/challenge
```

If a client uses the challenge, it sends the returned `nonce` with the same `/verify-email-photo` request. The route is not required for the normal one-call integration.

## Existing routes

Do not change these routes when integrating the service:

```text
POST /verify-email-photo
GET  /verification/challenge
POST /exam/session/start
POST /exam/session/{session_id}/window
GET  /exam/session/{session_id}
POST /exam/session/{session_id}/end
```

Exam monitoring is intentionally different from registration. It operates on short webcam windows during the exam, so multiple window requests are expected.

## Internal modules

```text
app/verification/
├── router.py                 HTTP routes only
├── view.py                   request/file handling
├── service.py                compatibility functions for existing callers
├── main.py                   verification and exam orchestration
├── faces.py                  image/video decoding and face/model operations
├── face_verification.py      identity comparison
├── registration_liveness.py  registration liveness/challenge logic
├── anti_spoof.py             passive presentation-attack detection
├── face_tracking.py          face presence and identity continuity
└── risk_engine.py            exam risk assessment
```

The important design rule is **one external API call, multiple small internal functions**. The individual checks remain separate so they can be tested and maintained independently, while the existing API remains easy to integrate.

## Identity vs liveness vs PAD

These checks answer different questions:

- **Identity:** does the face match the registered profile?
- **Registration liveness:** does the video contain natural/live facial movement, or optionally the requested challenge actions?
- **PAD:** does the camera input look like a physical photo/screen presentation attack?
- **Exam tracking:** does the candidate remain present and consistent across exam windows?
- **Risk engine:** what does the combination of recent exam signals indicate?

A face match by itself is not sufficient. For example, a photo of the correct candidate may match the profile but should still be rejected when PAD identifies it as a presentation attack.

## Important limitation

Passive PAD does not provide stream integrity. If an attacker replaces the webcam feed using a virtual camera, real-time face swap, or another injection technique, the PAD model may never see the physical presentation attack. Deepfake and virtual-camera injection detection are separate future work.

No part of this prototype should be described as providing 100% spoofing protection.

## Running locally

```bash
pip install -r requirements.txt
python -m pytest -q
uvicorn server:app --port 8000
```

The models are loaded at application startup. The service currently keeps exam sessions and challenge state in process memory, so production deployments using multiple workers need a shared store such as Redis or worker/session affinity.

## Compatibility

Existing function entry points are intentionally preserved where the surrounding application may already depend on them. In particular, `service.verify_faces()` remains available as a compatibility wrapper around the registration verification flow.

The refactor should change implementation details only; existing API routes, request fields and consumed response fields should remain compatible with the integrating application.
