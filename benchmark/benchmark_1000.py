import statistics
import time
import requests

URL = "http://127.0.0.1:8000/verify-email-photo"

PROFILE_PHOTO = r"D:\espoofing\uploads\verification\WIN_20261005_17_55_58_Pro.jpg"
LIVE_VIDEO = r"D:\espoofing\uploads\verification\WIN_20261005_17_56_19_Pro.mp4"

EMAIL = "test@example.com"
EXAM_ID = "1"

REQUESTS_TO_SEND = 10   # First test: change to 1000 after this works


def send_request():
    with open(PROFILE_PHOTO, "rb") as profile:
        with open(LIVE_VIDEO, "rb") as video:
            files = {
                "email_photo": (
                    "profile.jpg",
                    profile,
                    "image/jpeg",
                ),
                "live_photo": (
                    "live.mp4",
                    video,
                    "video/mp4",
                ),
            }

            data = {
                "email": EMAIL,
                "examId": EXAM_ID,
            }

            start = time.perf_counter()

            response = requests.post(
                URL,
                data=data,
                files=files,
                timeout=120,
            )

            elapsed = time.perf_counter() - start

    return response, elapsed


def main():
    latencies = []
    verified = 0
    failed = 0

    total_start = time.perf_counter()

    print(f"Starting benchmark: {REQUESTS_TO_SEND} requests")
    print("-" * 60)

    for i in range(1, REQUESTS_TO_SEND + 1):
        try:
            response, elapsed = send_request()

            latencies.append(elapsed)

            try:
                result = response.json()
                reason = result.get("reason_code", "UNKNOWN")
                is_verified = result.get("verified", False)
            except Exception:
                reason = "INVALID_JSON"
                is_verified = False

            if response.ok and is_verified:
                verified += 1
            else:
                failed += 1

            print(
                f"{i:4}/{REQUESTS_TO_SEND} | "
                f"{elapsed:6.2f}s | "
                f"HTTP {response.status_code} | "
                f"{reason}"
            )

        except Exception as e:
            failed += 1
            print(f"{i:4}/{REQUESTS_TO_SEND} | ERROR | {e}")

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"Total requests : {REQUESTS_TO_SEND}")
    print(f"Verified       : {verified}")
    print(f"Failed         : {failed}")
    print(f"Total time     : {total_time:.2f}s")

    if latencies:
        print(f"Average        : {statistics.mean(latencies):.3f}s")
        print(f"Median         : {statistics.median(latencies):.3f}s")
        print(f"Minimum        : {min(latencies):.3f}s")
        print(f"Maximum        : {max(latencies):.3f}s")

        sorted_times = sorted(latencies)
        p95_index = int(len(sorted_times) * 0.95) - 1
        p95_index = max(0, min(p95_index, len(sorted_times) - 1))

        print(f"P95            : {sorted_times[p95_index]:.3f}s")

    print("=" * 60)


if __name__ == "__main__":
    main()