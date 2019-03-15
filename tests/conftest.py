import pytest
import numpy as np
import cv2
import face_recognition

@pytest.fixture(scope='function')
def frames():
    '''
    Returns a Numpy array containing one test image
    with a batch dimension.
    '''
    img = cv2.imread(f'tests/data/img1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)
    assert len(img) == 1
    return img


@pytest.fixture(scope='function')
def faces():
    '''
    Returns the coordinates of faces detected using
    RetinaNet.
    '''

    faces = [
        np.array([
            [292, 422, 286, 378],
            [1422, 342, 250, 313],
            [2407, 508, 248, 311],
        ])
    ]

    faces = [face.astype(np.uint) for face in faces]

    return faces


@pytest.fixture
def subject_face_enc():
    '''
    Returns the face encoding of a reference subject ('The Rock').
    '''
    img = cv2.imread('data/rock.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(img)
    return enc
