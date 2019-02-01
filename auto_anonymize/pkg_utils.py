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
