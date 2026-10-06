#!/usr/bin/env python3
"""Run repeated API verification requests and capture server resource usage.

This is a standalone benchmark. It does not modify the production API.

Example (Windows):
    python benchmark_100_requests.py \
        --url http://127.0.0.1:8000/verify-email-photo \
        --profile "C:\\path\\profile.jpg" \
        --video "C:\\path\\live.mp4" \
        --email test@example.com \
        --exam-id 1 \
        --pid 12345

If --pid is omitted, the script still records request latency and can record
system-level CPU/RAM when psutil is available. For per-server-process CPU/RAM,
pass the PID of the uvicorn worker/process handling the requests.

GPU metrics are reported when NVIDIA NVML is available. GPU utilization is
device-level during the request; GPU memory is reported for the target process
when NVML can associate it with the process.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    import requests
except ImportError:
    requests = None


SAMPLE_INTERVAL_SEC = 0.10


class ResourceSampler:
    def __init__(self, pid: int | None):
        self.pid = pid
        self.samples: list[dict] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.gpu_ready = False
        self.gpu_handles = []
        self.before = None

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(pynvml.nvmlDeviceGetCount())
                ]
                self.gpu_ready = bool(self.gpu_handles)
            except Exception:
                self.gpu_ready = False
        self.before = self._snapshot()

    def _processes(self):
        if psutil is None or self.pid is None:
            return []

        try:
            root = psutil.Process(self.pid)
            processes = [root]
            processes.extend(root.children(recursive=True))
            return [p for p in processes if p.is_running()]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

    def _snapshot(self):
        ram_mb = None
        if psutil is not None and self.pid is not None:
            current = self._processes()
            if current:
                rss = 0
                for p in current:
                    try:
                        rss += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                ram_mb = rss / (1024 * 1024)

        gpu_util = None
        gpu_mem_mb = None
        if self.gpu_ready:
            utils = []
            mem_values = []
            for handle in self.gpu_handles:
                try:
                    utils.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    mem_values.append(
                        pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)
                    )
                except Exception:
                    continue
            if utils:
                gpu_util = max(utils)
                gpu_mem_mb = max(mem_values)

        return {
            "ram_mb": ram_mb,
            "gpu_util_percent": gpu_util,
            "gpu_memory_mb": gpu_mem_mb,
        }

    def _run(self):
        processes = self._processes()
        for p in processes:
            try:
                p.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        while not self.stop_event.wait(SAMPLE_INTERVAL_SEC):
            cpu = None
            rss_mb = None

            current = self._processes()
            if current:
                cpu_values = []
                rss = 0
                for p in current:
                    try:
                        cpu_values.append(p.cpu_percent(None))
                        rss += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                if cpu_values:
                    cpu = sum(cpu_values)
                    rss_mb = rss / (1024 * 1024)

            gpu_util = None
            gpu_mem_mb = None
            if self.gpu_ready:
                utils = []
                mem_values = []
                for handle in self.gpu_handles:
                    try:
                        utils.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                        mem_values.append(
                            pynvml.nvmlDeviceGetMemoryInfo(handle).used
                            / (1024 * 1024)
                        )
                    except Exception:
                        continue
                if utils:
                    gpu_util = max(utils)
                    gpu_mem_mb = max(mem_values)

            self.samples.append({
                "cpu_percent": cpu,
                "ram_mb": rss_mb,
                "gpu_util_percent": gpu_util,
                "gpu_memory_mb": gpu_mem_mb,
            })

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def summary(self):
        def values(key):
            return [x[key] for x in self.samples if x[key] is not None]

        result = {}
        for key in (
            "cpu_percent",
            "ram_mb",
            "gpu_util_percent",
            "gpu_memory_mb",
        ):
            vals = values(key)
            result[f"{key}_avg"] = statistics.fmean(vals) if vals else None
            result[f"{key}_peak"] = max(vals) if vals else None

        after = self._snapshot()
        result["ram_before_mb"] = self.before["ram_mb"] if self.before else None
        result["ram_after_mb"] = after["ram_mb"]
        result["gpu_memory_before_mb"] = (
            self.before["gpu_memory_mb"] if self.before else None
        )
        result["gpu_memory_after_mb"] = after["gpu_memory_mb"]
        return result

    def close(self):
        if self.gpu_ready:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def fmt(value, digits=3):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="100-request verification benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000/verify-email-photo")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--exam-id", required=True, type=int)
    parser.add_argument("--pid", type=int, help="PID of the API server/worker")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--output-dir", default="benchmark_results")
    args = parser.parse_args()

    if requests is None:
        raise SystemExit("Install benchmark dependencies first: pip install -r requirements-benchmark.txt")

    profile = Path(args.profile)
    video = Path(args.video)
    if not profile.is_file():
        raise SystemExit(f"Profile file not found: {profile}")
    if not video.is_file():
        raise SystemExit(f"Video file not found: {video}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_100_requests.csv"
    summary_path = output_dir / "benchmark_100_summary.txt"

    rows = []
    latencies = []

    for request_no in range(1, args.requests + 1):
        sampler = ResourceSampler(args.pid)
        sampler.start()

        started = time.perf_counter()
        status_code = None
        payload = {}
        error = ""

        try:
            with profile.open("rb") as profile_file, video.open("rb") as video_file:
                files = {
                    "email_photo": (profile.name, profile_file, "image/jpeg"),
                    "live_photo": (video.name, video_file, "video/mp4"),
                }
                data = {
                    "email": args.email,
                    "examId": str(args.exam_id),
                }
                response = requests.post(
                    args.url,
                    data=data,
                    files=files,
                    timeout=120,
                )
                status_code = response.status_code
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                    error = "Response was not JSON"
        except Exception as exc:
            error = str(exc)

        elapsed = time.perf_counter() - started
        sampler.stop()
        resources = sampler.summary()
        sampler.close()

        latencies.append(elapsed)
        rows.append({
            "request": request_no,
            "http_status": status_code,
            "latency_sec": elapsed,
            "verified": payload.get("verified"),
            "reason_code": payload.get("reason_code"),
            "is_match": payload.get("is_match", payload.get("identity_match")),
            "cpu_avg_percent": resources["cpu_percent_avg"],
            "cpu_peak_percent": resources["cpu_percent_peak"],
            "ram_before_mb": resources["ram_before_mb"],
            "ram_peak_mb": resources["ram_mb_peak"],
            "ram_after_mb": resources["ram_after_mb"],
            "gpu_util_avg_percent": resources["gpu_util_percent_avg"],
            "gpu_util_peak_percent": resources["gpu_util_percent_peak"],
            "gpu_memory_before_mb": resources["gpu_memory_before_mb"],
            "gpu_memory_peak_mb": resources["gpu_memory_mb_peak"],
            "gpu_memory_after_mb": resources["gpu_memory_after_mb"],
            "error": error,
        })

        print(
            f"{request_no:03d}/{args.requests}: "
            f"{elapsed:.3f}s status={status_code} "
            f"verified={payload.get('verified')} "
            f"reason={payload.get('reason_code')}"
        )

    latency_values = latencies
    verified_count = sum(row["verified"] is True for row in rows)
    failed_count = args.requests - verified_count

    reason_counts = {}
    for row in rows:
        reason = row["reason_code"] or "REQUEST_ERROR"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    def metric(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return (statistics.fmean(vals), max(vals)) if vals else (None, None)

    cpu_avg, cpu_peak = metric("cpu_avg_percent")
    ram_avg, ram_peak = metric("ram_peak_mb")
    gpu_avg, gpu_peak = metric("gpu_util_avg_percent")
    gpu_mem_avg, gpu_mem_peak = metric("gpu_memory_peak_mb")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = f"""Frame sampling: 15
Requests: {args.requests}

Latency:
Average: {statistics.fmean(latency_values):.3f} sec
Median: {statistics.median(latency_values):.3f} sec
P95: {percentile(latency_values, 0.95):.3f} sec
Min: {min(latency_values):.3f} sec
Max: {max(latency_values):.3f} sec

CPU (target process and child processes):
Average: {fmt(cpu_avg)} %
Peak: {fmt(cpu_peak)} %

RAM (target process and child processes):
Average peak-per-request: {fmt(ram_avg)} MB
Peak: {fmt(ram_peak)} MB

GPU:
Average utilization: {fmt(gpu_avg)} %
Peak utilization: {fmt(gpu_peak)} %
Average peak memory: {fmt(gpu_mem_avg)} MB
Peak memory: {fmt(gpu_mem_peak)} MB

Verification results:
Verified: {verified_count}/{args.requests}
Failed: {failed_count}/{args.requests}

Reason codes:
"""
    for reason, count in sorted(reason_counts.items()):
        summary += f"{reason}: {count}\n"

    summary += f"""
Notes:
- Input video frame rate is unchanged; production sampling is 15 frames.
- CPU/RAM are measured for --pid and its child processes when psutil is available.
- GPU utilization and memory are device-level when NVIDIA NVML is available.
- CPU/RAM values require --pid; without it they are unavailable rather than system-wide estimates.
- Missing metrics are reported as unavailable; no values are fabricated.
- Detailed per-request results: {csv_path}
"""
    summary_path.write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
