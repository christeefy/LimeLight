import numpy as np
import face_recognition

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from pkg_utils import std2facerec, facerec2std


def embed_face(rgb, bboxes):
    '''
    Returns a list of 128-dim face embeddings.

    Inputs:
        rgb:    cv2 RGB image
        bboxes: List of bounding box coordinates
                (std: x1, y1, w, h).

    Returns:
        A list of 128-dim face embeddings.
        Has the same length as `bboxes`.
    '''
    return face_recognition.face_encodings(rgb, std2facerec(bboxes))


def recognize_face(new, known):
    '''
    Compares whether a list of new face encodings matches
    a list of known face encodings.

    Inputs:
        new: A list of new face encodings
        known: A list of known face encodings

    Return a NumPy boolean with the same length as `new`.
    '''
    assert isinstance(new, list), 'Ensure `new` argument is a list.'
    return np.max([face_recognition.compare_faces(known, new_) for new_ in new], axis=1)
