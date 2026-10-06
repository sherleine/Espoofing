"""Temporary frame-level verification profiler.

Run this file from the repository root:

    python frame_timing_profiler.py

It starts the same API as server.py, but adds temporary diagnostics around
the existing registration pipeline. It does not modify production files.

Send the normal Postman request to:

    POST http://127.0.0.1:8000/verify-email-photo

The output reports:
- total video frames
- sampled frames
- frames with a detected face
- frames used for identity embeddings
- frames scored by PAD
- time spent in each stage
- total request time

Press Ctrl+C when finished.
"""

import time

from fastapi import UploadFile
from starlette.requests import Request

from server import app


def _wrap(name, func, timings, counters=None):
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - started
        timings[name] = timings.get(name, 0.0) + elapsed

        if counters is not None:
            counters["calls:" + name] = counters.get("calls:" + name, 0) + 1

        return result

    return wrapper


def install_hooks():
    from app.verification import anti_spoof, faces, registration_liveness
    from app.verification import face_verification, view

    timings = {}
    counters = {}

    # Keep references to the original functions so this file is safe to run
    # repeatedly in a fresh Python process.
    original_read_video = faces.read_video
    original_analyze_frame = faces.analyze_frame
    original_embed = faces.embed
    original_compare = face_verification.compare
    original_movement_check = registration_liveness.movement_check
    original_detect_spoof = anti_spoof.detect_spoof
    original_build_template = face_verification.build_template
    original_verify_faces = view.verify_faces
    original_write = view._write
    original_remove = view._remove
    original_upload_read = UploadFile.read

    def timed_read_video(*args, **kwargs):
        started = time.perf_counter()
        result = original_read_video(*args, **kwargs)
        timings["video decode / frame sampling"] = (
            timings.get("video decode / frame sampling", 0.0)
            + time.perf_counter() - started
        )

        frames = result[0] if isinstance(result, tuple) and result else []
        counters["sampled frames"] = len(frames)

        # read_video returns metadata after the frames. Capture it without
        # assuming exact metadata names.
        if isinstance(result, tuple):
            if len(result) > 1:
                counters["video metadata 1"] = result[1]
            if len(result) > 2:
                counters["video metadata 2"] = result[2]

        return result

    def timed_analyze_frame(*args, **kwargs):
        started = time.perf_counter()
        result = original_analyze_frame(*args, **kwargs)
        timings["face detection / analysis"] = (
            timings.get("face detection / analysis", 0.0)
            + time.perf_counter() - started
        )

        counters["analyzed frames"] = counters.get("analyzed frames", 0) + 1

        # analyze_frame returns a face list/detection result. Count a frame as
        # face-positive when the returned value is non-empty.
        try:
            if result:
                counters["frames with face"] = counters.get("frames with face", 0) + 1
        except (TypeError, ValueError):
            pass

        return result

    def timed_embed(*args, **kwargs):
        started = time.perf_counter()
        result = original_embed(*args, **kwargs)
        timings["identity embedding"] = (
            timings.get("identity embedding", 0.0)
            + time.perf_counter() - started
        )
        counters["embedding calls"] = counters.get("embedding calls", 0) + 1
        return result

    def timed_detect_spoof(*args, **kwargs):
        started = time.perf_counter()
        result = original_detect_spoof(*args, **kwargs)
        timings["PAD / anti-spoof"] = (
            timings.get("PAD / anti-spoof", 0.0)
            + time.perf_counter() - started
        )

        # anti_spoof.detect_spoof receives the selected frames as its first
        # positional argument in the current implementation.
        if args:
            try:
                counters["PAD input frames"] = len(args[0])
            except TypeError:
                pass

        return result

    def timed_compare(*args, **kwargs):
        started = time.perf_counter()
        result = original_compare(*args, **kwargs)
        timings["identity comparison"] = (
            timings.get("identity comparison", 0.0)
            + time.perf_counter() - started
        )
        return result

    def timed_movement_check(*args, **kwargs):
        started = time.perf_counter()
        result = original_movement_check(*args, **kwargs)
        timings["liveness / movement check"] = (
            timings.get("liveness / movement check", 0.0)
            + time.perf_counter() - started
        )
        return result

    def timed_build_template(*args, **kwargs):
        started = time.perf_counter()
        result = original_build_template(*args, **kwargs)
        timings["reference template"] = (
            timings.get("reference template", 0.0)
            + time.perf_counter() - started
        )
        return result

    def timed_verify_faces(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_verify_faces(*args, **kwargs)
        finally:
            timings["core verification"] = time.perf_counter() - started

    def timed_write(path, data):
        started = time.perf_counter()
        try:
            return original_write(path, data)
        finally:
            timings["temporary file writes"] = (
                timings.get("temporary file writes", 0.0)
                + time.perf_counter() - started
            )

    def timed_remove(*paths):
        started = time.perf_counter()
        try:
            return original_remove(*paths)
        finally:
            timings["temporary file cleanup"] = (
                timings.get("temporary file cleanup", 0.0)
                + time.perf_counter() - started
            )

    async def timed_upload_read(self, *args, **kwargs):
        started = time.perf_counter()
        data = await original_upload_read(self, *args, **kwargs)
        elapsed = time.perf_counter() - started

        name = self.filename or "unnamed file"
        timings["upload read: " + name] = elapsed
        counters["upload bytes: " + name] = len(data)
        return data

    faces.read_video = timed_read_video
    faces.analyze_frame = timed_analyze_frame
    faces.embed = timed_embed
    face_verification.compare = timed_compare
    registration_liveness.movement_check = timed_movement_check
    anti_spoof.detect_spoof = timed_detect_spoof
    face_verification.build_template = timed_build_template

    view.verify_faces = timed_verify_faces
    view._write = timed_write
    view._remove = timed_remove
    UploadFile.read = timed_upload_read

    app.state.frame_timing = {
        "timings": timings,
        "counters": counters,
    }


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    started = time.perf_counter()
    content_length = request.headers.get("content-length", "unknown")

    try:
        return await call_next(request)
    finally:
        total = time.perf_counter() - started
        state = getattr(app.state, "frame_timing", {})
        timings = state.get("timings", {})
        counters = state.get("counters", {})

        print()
        print("=" * 72)
        print("FRAME-LEVEL VERIFICATION PROFILER")
        print("=" * 72)
        print(f"Method / path                 : {request.method} {request.url.path}")
        print(f"Content-Length                : {content_length}")
        print()

        print("FRAME COUNTS")
        print("-" * 72)

        for key in (
            "sampled frames",
            "analyzed frames",
            "frames with face",
            "embedding calls",
            "PAD input frames",
        ):
            if key in counters:
                print(f"{key:<30}: {counters[key]}")

        for key, value in counters.items():
            if key.startswith("upload bytes:"):
                print(f"{key:<30}: {value / (1024 * 1024):.2f} MB")

        for key in ("video metadata 1", "video metadata 2"):
            if key in counters:
                print(f"{key:<30}: {counters[key]}")

        print()
        print("TIMINGS")
        print("-" * 72)

        for key, value in timings.items():
            print(f"{key:<30}: {value:.3f}s")

        print("-" * 72)
        print(f"TOTAL HTTP REQUEST            : {total:.3f}s")
        print("=" * 72)
        print()


install_hooks()


if __name__ == "__main__":
    import uvicorn

    print("Starting temporary frame-level verification profiler...")
    print("Send the normal Postman request to:")
    print("  POST http://127.0.0.1:8000/verify-email-photo")
    print("Press Ctrl+C when finished.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
