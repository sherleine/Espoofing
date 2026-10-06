"""
Temporary standalone profiler for the registration verification pipeline.

Run:
    python timing_profiler.py

It uses the real verification code and measures each major stage without
changing the application code. Delete this file after profiling.
"""

import sys
import time
from pathlib import Path

# Keep the profiler itself independent of the API/router.
# It imports the real verification modules and wraps their existing functions.


class Timer:
    def __init__(self):
        self.times = {}
        self._starts = {}

    def start(self, name):
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        elapsed = time.perf_counter() - self._starts.pop(name)
        self.times[name] = self.times.get(name, 0.0) + elapsed
        return elapsed


timing = Timer()


# ---------------------------------------------------------------------------
# Wrap the real functions before running verify_registration().
# ---------------------------------------------------------------------------

from app.verification import anti_spoof
from app.verification import face_verification
from app.verification import faces
from app.verification import registration_liveness


_original_read_video = faces.read_video
_original_analyze_frame = faces.analyze_frame
_original_embed = faces.embed
_original_detect_spoof = anti_spoof.detect_spoof
_original_movement_check = registration_liveness.movement_check
_original_compare = face_verification.compare
_original_build_template = face_verification.build_template


def timed_read_video(*args, **kwargs):
    timing.start("Video decode / frame sampling")
    try:
        return _original_read_video(*args, **kwargs)
    finally:
        timing.stop("Video decode / frame sampling")


def timed_analyze_frame(*args, **kwargs):
    # analyze_frame is the actual InsightFace face-analysis call used by the
    # application. It can include detection, landmarks and/or embedding
    # depending on the arguments supplied by the caller.
    timing.start("Face detection / analysis")
    try:
        return _original_analyze_frame(*args, **kwargs)
    finally:
        timing.stop("Face detection / analysis")


def timed_embed(*args, **kwargs):
    timing.start("Identity embedding")
    try:
        return _original_embed(*args, **kwargs)
    finally:
        timing.stop("Identity embedding")


def timed_detect_spoof(*args, **kwargs):
    timing.start("PAD / anti-spoof")
    try:
        return _original_detect_spoof(*args, **kwargs)
    finally:
        timing.stop("PAD / anti-spoof")


def timed_movement_check(*args, **kwargs):
    timing.start("Liveness / movement check")
    try:
        return _original_movement_check(*args, **kwargs)
    finally:
        timing.stop("Liveness / movement check")


def timed_compare(*args, **kwargs):
    timing.start("Identity comparison")
    try:
        return _original_compare(*args, **kwargs)
    finally:
        timing.stop("Identity comparison")


def timed_build_template(*args, **kwargs):
    timing.start("Reference template")
    try:
        return _original_build_template(*args, **kwargs)
    finally:
        timing.stop("Reference template")


faces.read_video = timed_read_video
faces.analyze_frame = timed_analyze_frame
faces.embed = timed_embed
anti_spoof.detect_spoof = timed_detect_spoof
registration_liveness.movement_check = timed_movement_check
face_verification.compare = timed_compare
face_verification.build_template = timed_build_template


# Import AFTER wrapping the shared modules so verify_registration uses the
# wrapped functions above.
from app.verification.main import verify_registration


def ask_path(label):
    while True:
        value = input(f"{label}: ").strip().strip('"')
        path = Path(value)

        if path.is_file():
            return str(path)

        print(f"File not found: {path}")
        print("Please enter the full path to the file.\n")


def print_report(result, total_time):
    print()
    print("=" * 64)
    print("VERIFICATION PERFORMANCE")
    print("=" * 64)
    print()

    for name, seconds in timing.times.items():
        print(f"{name:<36}: {seconds:>8.3f}s")

    print("-" * 64)
    print(f"{'TOTAL VERIFICATION':<36}: {total_time:>8.3f}s")
    print("=" * 64)

    print()
    print("RESULT")
    print("-" * 64)
    print(f"Verified       : {result.get('verified')}")
    print(f"Reason         : {result.get('reason_code')}")
    print(f"Identity match : {result.get('identity_match')}")
    print(f"Similarity     : {result.get('similarity')}")

    pad = result.get("anti_spoof") or {}
    print(f"PAD status     : {pad.get('status')}")
    print(f"PAD score      : {pad.get('spoof_score')}")

    video = result.get("video") or {}
    print(f"Frames sampled : {video.get('frames_sampled')}")
    print(f"Video frames   : {video.get('total_frames')}")
    print()


def main():
    print()
    print("=" * 64)
    print("TEMPORARY VERIFICATION TIMING PROFILER")
    print("=" * 64)
    print()
    print("This runs the REAL registration verification pipeline once.")
    print("Nothing in the application is modified.")
    print()

    profile_path = ask_path("Profile photo path")
    video_path = ask_path("Live video path")

    print()
    print("Running verification...")
    print("The first run may include model initialization.")
    print()

    total_start = time.perf_counter()

    try:
        result = verify_registration(profile_path, video_path)
    except Exception as exc:
        total_time = time.perf_counter() - total_start
        print()
        print("Verification failed with an exception:")
        print(repr(exc))
        print()
        print(f"Time until failure: {total_time:.3f}s")
        raise

    total_time = time.perf_counter() - total_start

    print_report(result, total_time)


if __name__ == "__main__":
    main()
