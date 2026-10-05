# app/verification/face_tracking.py
"""
Face presence, face count and same-person continuity, for one registration
video (summarize, check_presence, consistency) and across exam windows
(ExamTracker).
"""

from app.verification.face_verification import (
    SIMILARITY_THRESHOLD, build_template, largest, similarity)

SAME_PERSON_THRESHOLD = 0.50    # every video face vs the video's mean face
MIN_FACE_FRAME_RATIO = 0.6      # face must be visible in 60% of frames
MAX_MULTI_FACE_RATIO = 0.2      # >20% frames with 2+ faces -> reject
MIN_FACE_FRAMES = 5

# Exam - placeholders, calibrate on real exam data
EXAM_MIN_FACE_RATIO = 0.3       # below this a window counts as "no face"
EXAM_MULTI_FACE_RATIO = 0.3     # at/above this a window counts as "multiple faces"
EXAM_MIN_REL_FACE_AREA = 0.04   # ignore extra faces smaller than 4% of the main face
                                # (posters, photos far in the background)


def summarize(faces_per_frame, min_rel_area=0.0):
    """Per-window face statistics. `faces_per_frame` is a list of face lists.

    Returns a dict with the response fields plus `primary`: the largest face
    of every frame (None when the frame has no face).
    """
    primary, multi = [], 0
    for faces in faces_per_frame:
        if not faces:
            primary.append(None)
            continue
        main = largest(faces)
        primary.append(main)
        if min_rel_area > 0:
            main_area = _area(main)
            faces = [f for f in faces if f is main or _area(f) >= min_rel_area * main_area]
        if len(faces) > 1:
            multi += 1

    n = len(faces_per_frame)
    with_face = sum(f is not None for f in primary)
    return {
        "frames_sampled": n,
        "frames_with_face": with_face,
        "face_ratio": with_face / n if n else 0.0,
        "multi_face_ratio": multi / n if n else 0.0,
        "primary": primary,
    }


def check_presence(summary):
    """Registration gate. Returns a reason code or None."""
    if summary["multi_face_ratio"] > MAX_MULTI_FACE_RATIO:
        return "MULTIPLE_FACES_IN_VIDEO"
    if (summary["face_ratio"] < MIN_FACE_FRAME_RATIO
            or summary["frames_with_face"] < MIN_FACE_FRAMES):
        return "FACE_NOT_CONSISTENT"
    return None


def consistency(embeddings):
    """Similarity of every embedding to the mean embedding (same person check)."""
    return embeddings @ build_template(embeddings)


def _area(face):
    return float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


class ExamTracker:
    """Per exam session: turns each window into face_count (NONE | ONE |
    MULTIPLE), identity vs the reference and continuity vs the last window."""

    def __init__(self, reference_embedding):
        self.reference = reference_embedding
        self.last_embedding = None      # embedding of the last identity sample
        self.absent_windows = 0         # consecutive windows without a face

    def face_count(self, summary):
        if summary["face_ratio"] < EXAM_MIN_FACE_RATIO:
            return "NONE"
        if summary["multi_face_ratio"] >= EXAM_MULTI_FACE_RATIO:
            return "MULTIPLE"
        return "ONE"

    def update(self, face_count, embedding):
        """Feed one window. `embedding` is the identity sample taken in this
        window (or None if none was taken). Returns the window facts."""
        returned = face_count != "NONE" and self.absent_windows > 0
        self.absent_windows = self.absent_windows + 1 if face_count == "NONE" else 0

        identity, continuity, ref_sim, prev_sim = "UNKNOWN", "UNKNOWN", None, None
        if embedding is not None:
            ref_sim = similarity(embedding, self.reference)
            identity = "MATCH" if ref_sim >= SIMILARITY_THRESHOLD else "MISMATCH"
            if self.last_embedding is not None:
                prev_sim = similarity(embedding, self.last_embedding)
                continuity = "STABLE" if prev_sim >= SAME_PERSON_THRESHOLD else "CHANGED"
            self.last_embedding = embedding

        return {
            "face_count": face_count,
            "identity": identity,
            "continuity": continuity,
            "reference_similarity": None if ref_sim is None else round(ref_sim, 4),
            "previous_similarity": None if prev_sim is None else round(prev_sim, 4),
            "face_returned": returned,
            "absent_windows": self.absent_windows,
        }
