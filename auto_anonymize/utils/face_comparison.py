import numpy as np
import face_recognition

def compare_face_encs(new, known):
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