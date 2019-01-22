import cv2
from pathlib import Path

# assert (Path.cwd() / 'haarcascade_frontalface_default.xml').is_file(), \
#        'Haar Casecade xml file does not exists in directory.'

def detect_faces_haar_cascade(img, haarcascade_xml_src='haarcascade_frontalface_default.xml'):
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
        bboxes.sort(axis=0)
    
    return bboxes