import pytest
from limelight.face_detection.haar_cascade import detect_faces_hc
from limelight.face_detection.retinanet import detect_faces_ret, load_retinanet


@pytest.fixture(scope='function')
def retinanet_model():
    return load_retinanet()


def test_face_detection_ret(retinanet_model, frames):
    faces = detect_faces_ret(frames, retinanet_model)

    assert len(faces) == len(frames)
    assert len(faces[0]) == 3
