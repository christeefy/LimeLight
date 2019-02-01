import cv2
import numpy as np
from pathlib import Path

HAAR_CASCADE_XML_FILE_NAME = 'haarcascade_frontalface_default.xml'

assert (Path(__file__).parent / HAAR_CASCADE_XML_FILE_NAME).is_file(), \
       'Haar Cascade xml file does not exists in directory.'

def detect_faces_hc(img, haarcascade_xml_src=str(Path(__file__).parent / HAAR_CASCADE_XML_FILE_NAME)):
    '''
    Detect faces using the Viola-Jones face detection 
    framework using the openCV implementation of 
    Harr Cascades.
    
    Inputs:
        img: cv2 RGB image
        haarcascade_xml_src: XML model file for face 
                             detector
    
    Returns a numpy array denoting the bounding box
    coordinates (x1, y1, w, h) sorted by x1.
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    face_cascade = cv2.CascadeClassifier(haarcascade_xml_src)
    bboxes = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len((bboxes)):
        # Sort bboxes in increasing order of x1
        bboxes = bboxes[np.argsort(bboxes[:, 0])]
    
    return bboxes