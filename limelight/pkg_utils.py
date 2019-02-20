import cv2
import numpy as np


def to_rgb(bgr):
    '''
    Convert an image's channels from 
    BGR to RGB.
    '''
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rgb_read(img_path):
    '''
    Read an image file as a RGB image. 
    '''
    return to_rgb(cv2.imread(img_path))


def facerec2std(bboxes):
    '''
    Convert bounding boxes parameters of 
    face_recognition package (t, r, b, l)
    to (x1, y1, w, h).
    '''
    return np.array([(bbox[3], bbox[0], bbox[1] - bbox[3], bbox[2] - bbox[0]) for bbox in bboxes])


def std2facerec(bboxes):
    '''
    Convert bounding boxes parameters of 
    (x1, y1, w, h) to face_recognition 
    package (t, r, b, l).
    '''
    return np.array([(bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0]) for bbox in bboxes])


def expand_bboxes(frame, bboxes, margins=(1, 1)):
    '''
    Expand the bounding boxes coordinates by
    a ratio specificed by margins. 

    Inputs:
        frame:   A NumPy array of the image
        bboxes:  A list of bounding boxes of 
                 schema (x1, y1, w, h).
        margins: Ratio to expand margins by. 
                 Accepts tuple of (w_ratio, h_ratio).
                 If only only a float is specified, 
                 that will map to both w_ratio and h_ratio.

                 E.g. To expand both width and height of the
                 bounding boxes by 10%, specify (1.1, 1.1).

    Returns:
        A list of expanded bounding boxes.
    '''
    if isinstance(margins, float):
        margins = (margins, margins)
    assert isinstance(margins, tuple), 'margins has to be a tuple or a float.'

    frame_h, frame_w, _ = frame.shape

    # Performs expansion while ensuring bbox specifications 
    # do not exceed frame dimensions
    bboxes[..., 2] = np.clip(margins[0] * bboxes[..., 2], 
                             1, frame_w - bboxes[..., 0])
    bboxes[..., 3] = np.clip(margins[1] * bboxes[..., 3],
                             1, frame_h - bboxes[..., 1])

    return bboxes.astype(int)