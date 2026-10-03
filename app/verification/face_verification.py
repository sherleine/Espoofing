# app/verification/face_verification.py
"""
Identity only: does the live face belong to the registered candidate?

Works on embeddings already computed by main.py (InsightFace buffalo_l /
ArcFace, L2-normalised), so cosine similarity is a plain dot product.
No liveness or anti-spoofing logic lives here: a perfect identity match says
nothing about whether the face is physically present.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Tunable thresholds (log the returned metrics on real users and adjust)
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.45     # profile photo vs video face (cosine)
MIN_MATCH_RATIO = 0.6           # fraction of video frames that must match


def largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def similarity(embedding, reference_embedding):
    """Cosine similarity of two normed embeddings."""
    return float(np.dot(embedding, reference_embedding))


def compare(embeddings, reference_face):
    """Compare a stack of live embeddings (n, 512) with the reference face.

    Returns (identity, matched): `identity` is the response block (rounded
    values), `matched` is decided on the unrounded values.
    """
    sims = embeddings @ reference_face.normed_embedding
    median = float(np.median(sims))
    match_ratio = float(np.mean(sims >= SIMILARITY_THRESHOLD))
    identity = {
        "similarity": round(median, 4),
        "best_similarity": round(float(sims.max()), 4),
        "match_ratio": round(match_ratio, 3),
        "threshold": SIMILARITY_THRESHOLD,
        "profile_det_score": round(float(reference_face.det_score), 3),
    }
    matched = median >= SIMILARITY_THRESHOLD and match_ratio >= MIN_MATCH_RATIO
    return identity, matched
