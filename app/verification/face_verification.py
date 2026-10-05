# app/verification/face_verification.py
"""
Identity only: does the live face belong to the registered candidate?
A perfect match says nothing about whether the face is physically present.
Embeddings are L2-normalised, so cosine similarity is a plain dot product.
"""

import numpy as np

from app.verification.faces import EMBEDDING_MODEL, face_quality, pose

# Log the returned similarities on real users and adjust.
SIMILARITY_THRESHOLD = 0.45     # profile photo vs video face (cosine)
MIN_MATCH_RATIO = 0.6           # fraction of video frames that must match
MAX_IDENTITY_YAW_DEG = 35.0     # profile views give unreliable embeddings


def largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def pick_identity_face(frames, faces):
    """The (frame, face) in a window best suited for an identity sample:
    good quality, roughly frontal, then largest and most confident.
    `faces` holds the main face per frame (None = no face). Returns None if
    no frame qualifies."""
    best, best_key = None, -1.0
    for frame, face in zip(frames, faces):
        if face is None:
            continue
        ok, _, _ = face_quality(frame, face)
        _, yaw = pose(face)
        if not ok or abs(yaw) > MAX_IDENTITY_YAW_DEG:
            continue
        area = float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        key = float(face.det_score) * area
        if key > best_key:
            best, best_key = (frame, face), key
    return best


def build_template(embeddings):
    """One reference embedding from several frames: the normalised mean."""
    mean = embeddings.mean(axis=0)
    return mean / np.linalg.norm(mean)


def export_template(template):
    """JSON-safe form returned at registration and stored by the caller."""
    return {"model": EMBEDDING_MODEL, "vector": [round(float(x), 6) for x in template]}


def import_template(data):
    """Reverse of export_template. Returns (template, None) or (None, reason_code)."""
    if not isinstance(data, dict) or not isinstance(data.get("vector"), list):
        return None, "INVALID_REFERENCE_EMBEDDING"
    if data.get("model") != EMBEDDING_MODEL:
        return None, "REFERENCE_MODEL_MISMATCH"
    try:
        vector = np.asarray(data["vector"], dtype=np.float64)
    except (TypeError, ValueError):
        return None, "INVALID_REFERENCE_EMBEDDING"
    norm = np.linalg.norm(vector) if vector.shape == (512,) else 0.0
    if not np.isfinite(norm) or norm == 0:
        return None, "INVALID_REFERENCE_EMBEDDING"
    return vector / norm, None


def similarity(embedding, reference_embedding):
    """Cosine similarity of two normed embeddings."""
    return float(np.dot(embedding, reference_embedding))


def compare(embeddings, reference_embedding):
    """Compare a stack of live embeddings (n, 512) with one reference embedding.

    Returns (identity, matched): `identity` is the response block (rounded
    values), `matched` is decided on the unrounded values.
    """
    sims = embeddings @ reference_embedding
    median = float(np.median(sims))
    match_ratio = float(np.mean(sims >= SIMILARITY_THRESHOLD))
    identity = {
        "similarity": round(median, 4),
        "best_similarity": round(float(sims.max()), 4),
        "match_ratio": round(match_ratio, 3),
        "threshold": SIMILARITY_THRESHOLD,
    }
    matched = median >= SIMILARITY_THRESHOLD and match_ratio >= MIN_MATCH_RATIO
    return identity, matched
