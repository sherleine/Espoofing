import numpy as np

from app.verification import face_tracking as ft


class FakeFace(dict):
    def __init__(self, x, size):
        super().__init__(bbox=np.array([x, 0, x + size, size], float))

    def __getattr__(self, name):
        return self.get(name)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_summarize_counts_and_primary():
    big, small = FakeFace(0, 100), FakeFace(200, 50)
    s = ft.summarize([[big], [small, big], [], [big]])
    assert s["frames_with_face"] == 3
    assert s["multi_face_ratio"] == 0.25
    assert s["primary"] == [big, big, None, big]


def test_tiny_background_faces_ignored_in_exam_mode():
    big, poster = FakeFace(0, 200), FakeFace(400, 20)   # 1% of the area
    assert ft.summarize([[big, poster]])["multi_face_ratio"] == 1.0
    assert ft.summarize([[big, poster]], min_rel_area=ft.EXAM_MIN_REL_FACE_AREA)["multi_face_ratio"] == 0.0


def test_check_presence():
    assert ft.check_presence({"multi_face_ratio": 0.3, "face_ratio": 1, "frames_with_face": 20}) == "MULTIPLE_FACES_IN_VIDEO"
    assert ft.check_presence({"multi_face_ratio": 0, "face_ratio": 0.5, "frames_with_face": 20}) == "FACE_NOT_CONSISTENT"
    assert ft.check_presence({"multi_face_ratio": 0, "face_ratio": 1, "frames_with_face": 4}) == "FACE_NOT_CONSISTENT"
    assert ft.check_presence({"multi_face_ratio": 0, "face_ratio": 1, "frames_with_face": 20}) is None


def test_exam_tracker_identity_and_continuity():
    ref = unit([1, 0, 0])
    t = ft.ExamTracker(ref)
    same = t.update("ONE", unit([1, 0.1, 0]))
    assert same["identity"] == "MATCH" and same["continuity"] == "UNKNOWN"
    other = t.update("ONE", unit([0, 1, 0]))
    assert other["identity"] == "MISMATCH" and other["continuity"] == "CHANGED"


def test_exam_tracker_absence_forces_identity_check():
    t = ft.ExamTracker(unit([1, 0]))
    t.update("ONE", None)
    assert not t.needs_identity_check()
    t.update("NONE", None)
    assert t.needs_identity_check()
    back = t.update("ONE", unit([1, 0]))
    assert back["face_returned"] and not t.needs_identity_check()
