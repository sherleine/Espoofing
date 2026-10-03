"""
Measure passive PAD on your own labelled recordings (ISO/IEC 30107-3 style).

    python tools/evaluate_pad.py dataset [--frames 16] [--csv results.csv]

Folder layout - one folder per presentation type, videos or images inside:

    dataset/
      bona_fide/        real candidates (aliases: real, live, genuine)
      print/            printed photos
      mobile/           photo/video on a phone
      screen/           photo/video on a monitor or tablet
      <anything>/       every other folder = one attack species

`tools/webcam_demo.py` records clips straight into this layout.

Reported (per attack species, as ISO/IEC 30107-3 requires):
    APCER  attacks accepted as bona fide          (per species + worst case)
    BPCER  bona fide presentations flagged as attacks
    ACER   (worst APCER + BPCER) / 2  - kept for comparison with papers; the
           current ISO edition discourages it as a headline number
    EER, and BPCER at the threshold where worst-case APCER <= 1% / 5%

A presentation counts as "flagged as attack" when its spoof_score >= the
threshold. Unscorable presentations (INSUFFICIENT_QUALITY) are listed
separately: for bona fide they are a usability problem, for attacks a
possible evasion.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.verification import anti_spoof, face_tracking, main  # noqa: E402

BONA_FIDE = {"bona_fide", "real", "live", "genuine"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def score_file(path, n_frames):
    if path.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(path))
        frames = [main.resize(img)] if img is not None else []
    else:
        frames, _, _ = main.read_video(str(path), n_frames)
    if not frames:
        return None
    faces = [main.analyze_frame(f, landmarks=False, embedding=False) for f in frames]
    primary = face_tracking.summarize(faces)["primary"]
    # a single image cannot reach MIN_SCORED_FRAMES: score it on its own
    min_frames = anti_spoof.MIN_SCORED_FRAMES
    if len(frames) == 1:
        anti_spoof.MIN_SCORED_FRAMES = 1
    try:
        return anti_spoof.detect_spoof(frames, primary)
    finally:
        anti_spoof.MIN_SCORED_FRAMES = min_frames


def rates(bona, attacks, thr):
    bpcer = float(np.mean(bona >= thr)) if len(bona) else float("nan")
    apcer = {sp: float(np.mean(s < thr)) for sp, s in attacks.items() if len(s)}
    worst = max(apcer.values()) if apcer else float("nan")
    return apcer, worst, bpcer


def main_eval(args):
    root = Path(args.dataset)
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS)
        for f in files:
            r = score_file(f, args.frames)
            rows.append({
                "file": str(f), "label": folder.name.lower(),
                "bona_fide": folder.name.lower() in BONA_FIDE,
                "status": "UNREADABLE" if r is None else r["status"],
                "spoof_score": None if r is None else r["spoof_score"],
                "frames_scored": 0 if r is None else r["frames_scored"],
                "skipped": "" if r is None else r["frames_skipped"],
            })
            print(f"{rows[-1]['label']:>12}  {rows[-1]['status']:>20}  "
                  f"{'' if rows[-1]['spoof_score'] is None else rows[-1]['spoof_score']:>8}  {f.name}")
    if not rows:
        sys.exit(f"no media found under {root}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    scored = [r for r in rows if r["spoof_score"] is not None]
    bona = np.array([r["spoof_score"] for r in scored if r["bona_fide"]])
    attacks = {}
    for r in scored:
        if not r["bona_fide"]:
            attacks.setdefault(r["label"], []).append(r["spoof_score"])
    attacks = {k: np.array(v) for k, v in attacks.items()}

    print("\n=== presentations ===")
    for label in sorted({r["label"] for r in rows}):
        group = [r for r in rows if r["label"] == label]
        unscored = sum(r["spoof_score"] is None for r in group)
        print(f"{label:>12}: {len(group)} total, {unscored} unscorable")

    if not len(bona) or not attacks:
        print("\nNeed scored bona fide AND attack presentations for APCER/BPCER.")
        return

    print("\n=== operating points (attack if spoof_score >= threshold) ===")
    for name, thr in [("ELEVATED", anti_spoof.SPOOF_ELEVATED), ("HIGH", anti_spoof.SPOOF_HIGH)]:
        apcer, worst, bpcer = rates(bona, attacks, thr)
        per = "  ".join(f"{k}={v:.1%}" for k, v in sorted(apcer.items()))
        print(f"{name:>8} thr={thr:.2f}  APCER[{per}]  worst={worst:.1%}  BPCER={bpcer:.1%}  "
              f"ACER={(worst + bpcer) / 2:.1%}")

    thresholds = np.unique(np.concatenate([bona, *attacks.values(), [0.0, 1.0 + 1e-9]]))
    curve = [(t, *rates(bona, attacks, t)[1:]) for t in thresholds]   # (thr, worst APCER, BPCER)
    eer_t, eer_a, eer_b = min(curve, key=lambda c: abs(c[1] - c[2]))
    print(f"\nEER ~ {(eer_a + eer_b) / 2:.1%} at threshold {eer_t:.3f} (worst-case APCER vs BPCER)")
    for target in (0.01, 0.05):
        ok = [c for c in curve if c[1] <= target]
        if ok:
            t, a, b = min(ok, key=lambda c: c[2])
            print(f"BPCER @ worst APCER <= {target:.0%}: {b:.1%} (threshold {t:.3f})")
        else:
            print(f"BPCER @ worst APCER <= {target:.0%}: not reachable")
    print("\nSmall datasets give unstable numbers - report the counts with every rate.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--frames", type=int, default=16, help="frames sampled per video")
    ap.add_argument("--csv", help="write per-file results here")
    main_eval(ap.parse_args())
