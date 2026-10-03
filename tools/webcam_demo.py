"""
Live webcam demo + test-data recorder. Runs the same code as the API.

    python tools/webcam_demo.py pad                              live PAD score per frame
    python tools/webcam_demo.py exam     [--profile photo.jpg]   silent exam monitoring
    python tools/webcam_demo.py register [--profile photo.jpg]   challenge + full verification

Keys (all modes):
    R          use the current webcam frame as the profile photo
    SPACE      register mode: start the challenge recording
    B P M S O  record a 3 s labelled clip into the dataset folder:
               B = bona_fide (real face)   P = print (paper photo)
               M = mobile (phone screen)   S = screen (monitor/tablet)
               O = other attack
    Q / ESC    quit

Recorded clips go to dataset/<label>/... - evaluate them with
    python tools/evaluate_pad.py dataset
"""

import argparse
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.verification import anti_spoof, face_verification, main, registration_liveness  # noqa: E402

LABEL_KEYS = {ord("b"): "bona_fide", ord("p"): "print", ord("m"): "mobile",
              ord("s"): "screen", ord("o"): "other_attack"}
RECORD_SEC = 3.0
COLORS = {"NORMAL": (60, 180, 60), "LOW_RISK": (60, 180, 60), "SUSPICIOUS": (0, 165, 255),
          "ELEVATED": (0, 165, 255), "HIGH_RISK": (40, 40, 220), "INSUFFICIENT_QUALITY": (160, 160, 160)}


def text(img, s, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def pad_status(score):
    if score is None:
        return "INSUFFICIENT_QUALITY"
    if score >= anti_spoof.SPOOF_HIGH:
        return "HIGH_RISK"
    return "ELEVATED" if score >= anti_spoof.SPOOF_ELEVATED else "LOW_RISK"


class Background:
    """Run one job at a time off the UI thread."""

    def __init__(self):
        self.thread, self.result = None, None

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def run(self, fn, *args):
        def job():
            self.result = fn(*args)
        self.thread = threading.Thread(target=job, daemon=True)
        self.thread.start()


class Recorder:
    def __init__(self, root):
        self.root, self.writer, self.until, self.path = Path(root), None, 0.0, None

    def start(self, label, frame, fps):
        d = self.root / label
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        h, w = frame.shape[:2]
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self.until = time.time() + RECORD_SEC

    def write(self, frame):
        if self.writer is None:
            return
        self.writer.write(frame)
        if time.time() >= self.until:
            self.writer.release()
            self.writer = None
            print(f"saved {self.path}")


def main_loop(args):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {args.camera}")
    # Saved clips use the loop's real rate (per-frame analysis makes it slower
    # than the camera), so their timestamps and duration stay correct.
    loop_fps, last_frame_at = 10.0, None

    profile = cv2.imread(args.profile) if args.profile else None
    if args.profile and profile is None:
        sys.exit(f"cannot read {args.profile}")
    profile_emb = None
    session_id = None
    work = Background()
    recorder = Recorder(args.dataset)

    window, next_window = [], time.time() + 3
    banner, banner_color, detail = "", (255, 255, 255), []
    challenge, reg_writer, reg_path, step_started = None, None, None, 0.0

    def set_profile(img):
        nonlocal profile, profile_emb, session_id
        faces = main.analyze_frame(main.resize(img), landmarks=False)
        if not faces:
            print("no face in profile image")
            return
        profile, profile_emb = img.copy(), face_verification.largest(faces).normed_embedding
        if args.mode == "exam":
            if session_id:
                main.end_exam_session(session_id)
            session_id = main.start_exam_session("demo@example.com", 0, profile)["session_id"]
        print("profile set")

    if profile is not None:
        set_profile(profile)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if last_frame_at is not None and now > last_frame_at:
            loop_fps = 0.9 * loop_fps + 0.1 * (1.0 / (now - last_frame_at))
        last_frame_at = now
        recorder.write(frame)
        view = frame.copy()

        # --- live per-frame view: faces + PAD score of the main face ---------
        faces = main.analyze_frame(main.resize(frame), landmarks=False, embedding=False,
                                   det_size=main.EXAM_DET_SIZE)
        small = main.resize(frame)
        scale = frame.shape[1] / small.shape[1]
        if faces:
            face = face_verification.largest(faces)
            good, why, _ = anti_spoof.face_quality(small, face)
            score = None
            if good:
                p = anti_spoof.BACKENDS[0].live_probabilities([small], [face])[0]
                score = None if p is None else 1.0 - p
            status = pad_status(score)
            for f in faces:
                x1, y1, x2, y2 = (f.bbox * scale).astype(int)
                cv2.rectangle(view, (x1, y1), (x2, y2), COLORS[status] if f is face else (200, 200, 200), 2)
            _, yaw = registration_liveness.pose(face)
            text(view, f"faces={len(faces)}  spoof={'-' if score is None else f'{score:.3f}'}  "
                       f"[{status if good else why}]  yaw~{yaw:+.0f}", 25, COLORS[status])
        else:
            text(view, "no face", 25, (160, 160, 160))

        # --- exam mode: send a window every --interval seconds ----------------
        if args.mode == "exam" and session_id:
            if now >= next_window and len(window) < main.EXAM_WINDOW_FRAMES:
                if not window or now - window[-1][0] >= main.EXAM_FRAME_INTERVAL_MS / 1000:
                    window.append((now, frame.copy()))
            if len(window) >= main.EXAM_WINDOW_FRAMES and not work.busy():
                work.run(main.process_exam_window, session_id, [f for _, f in window])
                window, next_window = [], now + args.interval
            if work.result:
                r, work.result = work.result, None
                banner, banner_color = f"EXAM: {r['state']}", COLORS[r["state"]]
                s = r["signals"]
                detail = [f"identity={s['identity']} sim={s['reference_similarity']} continuity={s['continuity']}",
                          f"faces={s['face_count']} pad={s['pad']['status']} spoof={s['pad']['spoof_score']}",
                          "reasons: " + (", ".join(r["reasons"]) or "-"),
                          f"window {r['window']} in {r['processing_ms']} ms"]
                print(json.dumps({k: r[k] for k in ("window", "state", "reasons", "signals")}, default=str))

        # --- register mode: guided challenge recording -------------------------
        if args.mode == "register" and challenge:
            reg_writer.write(frame)
            step = int((now - step_started) // registration_liveness.STEP_SECONDS)
            if step < len(challenge["steps"]):
                left = registration_liveness.STEP_SECONDS - (now - step_started) % registration_liveness.STEP_SECONDS
                banner, banner_color = f"{step + 1}/{len(challenge['steps'])}: {challenge['prompts'][step]}", (0, 220, 255)
                detail = [f"{left:.1f}s"]
            else:
                reg_writer.release()
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
                    profile_path = t.name
                cv2.imwrite(profile_path, profile)
                work.run(lambda p=profile_path, v=str(reg_path), n=challenge["nonce"]:
                         main.verify_candidate(p, v, "registration", challenge_nonce=n))
                challenge, banner, detail = None, "verifying...", []
        if args.mode == "register" and not challenge and work.result:
            r, work.result = work.result, None
            banner = f"REGISTRATION: {r['reason_code']}"
            banner_color = (60, 180, 60) if r["verified"] else (40, 40, 220)
            detail = [f"live={r.get('live')} match={r.get('identity_match')} sim={r.get('similarity')}",
                      f"pad={(r.get('anti_spoof') or {}).get('status')} spoof={(r.get('anti_spoof') or {}).get('spoof_score')}",
                      f"challenge: {(r.get('challenge') or {}).get('failure') or 'passed'}"]
            print(json.dumps(r, indent=2, default=str))

        if banner:
            text(view, banner, 55, banner_color, 0.7)
        for i, line in enumerate(detail):
            text(view, line, 80 + 22 * i)
        if profile is None:
            text(view, "press R to use this frame as the profile photo", view.shape[0] - 15, (0, 220, 255))
        if recorder.writer is not None:
            text(view, f"REC {recorder.path.parent.name}", view.shape[0] - 40, (40, 40, 220))
        cv2.imshow("verification demo", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            set_profile(frame)
        elif key in LABEL_KEYS and recorder.writer is None:
            recorder.start(LABEL_KEYS[key], frame, loop_fps)
        elif key == ord(" ") and args.mode == "register" and profile is not None and not challenge and not work.busy():
            challenge = registration_liveness.issue_challenge()
            reg_path = Path(tempfile.gettempdir()) / f"challenge_{int(now)}.mp4"
            h, w = frame.shape[:2]
            reg_writer = cv2.VideoWriter(str(reg_path), cv2.VideoWriter_fourcc(*"mp4v"), loop_fps, (w, h))
            step_started = now
            print("challenge:", challenge["steps"])

    cap.release()
    cv2.destroyAllWindows()
    if session_id:
        print(json.dumps(main.end_exam_session(session_id), indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["pad", "exam", "register"])
    ap.add_argument("--profile", help="profile photo (otherwise press R)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--interval", type=float, default=5.0, help="exam: seconds between windows (demo default 5)")
    ap.add_argument("--dataset", default="dataset", help="where labelled clips are saved")
    main_loop(ap.parse_args())
