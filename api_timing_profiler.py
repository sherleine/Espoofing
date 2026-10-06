"""Temporary API timing profiler.

Run this file instead of server.py while diagnosing slow Postman requests:

    python api_timing_profiler.py

Then send the normal POST request to:
    http://127.0.0.1:8000/verify-email-photo

This file only measures the existing application. It does not change the
production verification code.
"""

import time

from fastapi import FastAPI, UploadFile
from starlette.requests import Request

from server import app


def _timed(name, func, timings):
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - started

    return wrapper


def install_timing_hooks() -> None:
    from app.verification import anti_spoof, faces, main, registration_liveness
    from app.verification import view
    from app.verification import face_verification

    timings = {}

    # The registration pipeline calls these functions directly.
    faces.read_video = _timed("video decode / frame sampling", faces.read_video, timings)
    faces.analyze_frame = _timed("face detection / analysis", faces.analyze_frame, timings)
    faces.embed = _timed("identity embedding", faces.embed, timings)

    face_verification.compare = _timed(
        "identity comparison", face_verification.compare, timings
    )
    registration_liveness.movement_check = _timed(
        "liveness / movement check", registration_liveness.movement_check, timings
    )
    anti_spoof.detect_spoof = _timed(
        "PAD / anti-spoof", anti_spoof.detect_spoof, timings
    )
    face_verification.build_template = _timed(
        "reference template", face_verification.build_template, timings
    )

    # view.verify_email_photo imported verify_faces directly, so patch the
    # reference held by the view module rather than only patching the service.
    original_verify_faces = view.verify_faces

    def timed_verify_faces(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_verify_faces(*args, **kwargs)
        finally:
            timings["core verification"] = time.perf_counter() - started

    view.verify_faces = timed_verify_faces

    # Temporary-file write/delete timing.
    original_write = view._write
    original_remove = view._remove

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

    view._write = timed_write
    view._remove = timed_remove

    # UploadFile.read() is where the endpoint reads the uploaded files after
    # Starlette has parsed the multipart request.
    original_upload_read = UploadFile.read

    async def timed_upload_read(self, *args, **kwargs):
        started = time.perf_counter()
        data = await original_upload_read(self, *args, **kwargs)
        elapsed = time.perf_counter() - started
        timings[f"upload read: {self.filename or 'unnamed file'}"] = elapsed
        timings[f"upload bytes: {self.filename or 'unnamed file'}"] = len(data)
        return data

    UploadFile.read = timed_upload_read

    # Store the latest timing dictionary on the app so middleware can print
    # the same measurements after the request finishes.
    app.state.timing_data = timings


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    started = time.perf_counter()
    content_length = request.headers.get("content-length", "unknown")

    try:
        response = await call_next(request)
        return response
    finally:
        total = time.perf_counter() - started
        timings = getattr(app.state, "timing_data", {})

        print()
        print("=" * 64)
        print("API REQUEST TIMING")
        print("=" * 64)
        print(f"Method / path                 : {request.method} {request.url.path}")
        print(f"Content-Length                : {content_length}")

        for name, value in timings.items():
            if name.startswith("upload bytes:"):
                print(f"{name:<30}: {value / (1024 * 1024):.2f} MB")
            else:
                print(f"{name:<30}: {value:.3f}s")

        print("-" * 64)
        print(f"TOTAL HTTP REQUEST            : {total:.3f}s")
        print("=" * 64)
        print()


install_timing_hooks()


if __name__ == "__main__":
    import uvicorn

    print("Starting temporary API timing profiler...")
    print("Send the normal Postman request to:")
    print("  POST http://127.0.0.1:8000/verify-email-photo")
    print("Press Ctrl+C when finished.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
