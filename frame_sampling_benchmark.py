"""Standalone A/B benchmark for registration frame sampling.

This script does NOT modify production files or API behavior.

It runs the real registration verification pipeline against the same profile
photo and live video at four sampling levels:

    25, 20, 15, 10 frames

Usage:

    python frame_sampling_benchmark.py <profile_photo> <live_video>

Example:

    python frame_sampling_benchmark.py profile.jpg live.mp4

The production TARGET_FRAMES value is only changed in memory inside this
temporary Python process. Nothing is written back to the repository.
"""

from __future__ import annotations

import argparse
import time


SAMPLE_COUNTS = (25, 20, 15, 10)


def run_benchmark(profile_path: str, video_path: str) -> None:
    from app.verification import anti_spoof
    from app.verification import face_verification
    from app.verification import faces
    from app.verification import main
    from app.verification import registration_liveness

    # Models are already loaded by app.verification.main at import time.
    original_target = main.TARGET_FRAMES

    original_read_video = faces.read_video
    original_analyze_frame = faces.analyze_frame
    original_embed = faces.embed
    original_compare = face_verification.compare
    original_movement_check = registration_liveness.movement_check
    original_detect_spoof = anti_spoof.detect_spoof

    try:
        for target in SAMPLE_COUNTS:
            timings = {
                "video decode / sampling": 0.0,
                "face detection / analysis": 0.0,
                "identity embedding": 0.0,
                "identity comparison": 0.0,
                "liveness / movement check": 0.0,
                "PAD / anti-spoof": 0.0,
            }
            calls = {
                "analyze_frame": 0,
                "embed": 0,
            }

            def timed_read_video(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original_read_video(*args, **kwargs)
                finally:
                    timings["video decode / sampling"] += time.perf_counter() - started

            def timed_analyze_frame(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original_analyze_frame(*args, **kwargs)
                finally:
                    timings["face detection / analysis"] += time.perf_counter() - started
                    calls["analyze_frame"] += 1

            def timed_embed(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original_embed(*args, **kwargs)
                finally:
                    timings["identity embedding"] += time.perf_counter() - started
                    calls["embed"] += 1

            def timed_compare(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original_compare(*args, **kwargs)
                finally:
                    timings["identity comparison"] += time.perf_counter() - started

            def timed_movement_check(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original_movement_check(*args, **kwargs)
                finally:
                    timings["liveness / movement check"] += time.perf_counter() - started

            def timed_detect_spoof(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original_detect_spoof(*args, **kwargs)
                finally:
                    timings["PAD / anti-spoof"] += time.perf_counter() - started

            faces.read_video = timed_read_video
            faces.analyze_frame = timed_analyze_frame
            faces.embed = timed_embed
            face_verification.compare = timed_compare
            registration_liveness.movement_check = timed_movement_check
            anti_spoof.detect_spoof = timed_detect_spoof

            # This is an in-memory override only. It affects the real
            # verify_registration() call made below and disappears when the
            # script exits.
            main.TARGET_FRAMES = target

            started = time.perf_counter()
            result = main.verify_registration(profile_path, video_path)
            total = time.perf_counter() - started

            video = result.get("video") or {}
            pad = result.get("anti_spoof") or {}
            liveness = result.get("liveness") or {}

            print()
            print("=" * 78)
            print(f"FRAME SAMPLING TEST: {target} FRAMES")
            print("=" * 78)

            print("VIDEO")
            print("-" * 78)
            print(f"Total video frames            : {video.get('total_frames')}")
            print(f"FPS                           : {video.get('fps')}")
            print(f"Duration                      : {video.get('duration_sec')}s")
            print(f"Frames sampled                : {video.get('frames_sampled')}")
            print(f"Frames with face              : {video.get('frames_with_face')}")
            print(f"Face ratio                    : {video.get('face_ratio')}")
            print(f"Multi-face ratio              : {video.get('multi_face_ratio')}")
            print(f"Minimum consistency           : {video.get('min_consistency')}")

            print()
            print("VERIFICATION")
            print("-" * 78)
            print(f"Identity similarity           : {result.get('similarity')}")
            print(f"Identity match                : {result.get('identity_match')}")
            print(f"Liveness passed               : {result.get('live')}")
            print(f"Liveness details              : {liveness}")
            print(f"PAD status                    : {pad.get('status')}")
            print(f"PAD spoof score              : {pad.get('spoof_score')}")
            print(f"PAD frames scored             : {pad.get('frames_scored')}")
            print(f"PAD frames skipped            : {pad.get('frames_skipped')}")
            print(f"Final verified                : {result.get('verified')}")
            print(f"Reason code                   : {result.get('reason_code')}")
            print(f"Failed checks                 : {result.get('failed_checks')}")

            print()
            print("TIMING")
            print("-" * 78)
            for name, elapsed in timings.items():
                print(f"{name:<30}: {elapsed:.3f}s")
            print(f"analyze_frame calls            : {calls['analyze_frame']}")
            print(f"embed calls                    : {calls['embed']}")
            print(f"TOTAL PROCESSING               : {total:.3f}s")
            print("=" * 78)

            # Restore functions after every run so wrappers never stack.
            faces.read_video = original_read_video
            faces.analyze_frame = original_analyze_frame
            faces.embed = original_embed
            face_verification.compare = original_compare
            registration_liveness.movement_check = original_movement_check
            anti_spoof.detect_spoof = original_detect_spoof

    finally:
        main.TARGET_FRAMES = original_target
        faces.read_video = original_read_video
        faces.analyze_frame = original_analyze_frame
        faces.embed = original_embed
        face_verification.compare = original_compare
        registration_liveness.movement_check = original_movement_check
        anti_spoof.detect_spoof = original_detect_spoof


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the registration pipeline at 25, 20, 15 and 10 sampled frames."
    )
    parser.add_argument("profile_photo", help="Path to the trusted profile photo")
    parser.add_argument("live_video", help="Path to the live verification video")
    args = parser.parse_args()

    print("=" * 78)
    print("FRAME SAMPLING A/B BENCHMARK")
    print("=" * 78)
    print(f"Profile photo : {args.profile_photo}")
    print(f"Live video    : {args.live_video}")
    print("Sampling      : 25, 20, 15, 10 frames")
    print()
    print("Production code is not modified. TARGET_FRAMES is overridden only")
    print("in memory for each benchmark run.")
    print("=" * 78)

    run_benchmark(args.profile_photo, args.live_video)


if __name__ == "__main__":
    main_cli()
