def facereg2std(bboxes):
    '''
    Convert bounding boxes parameters of 
    face_recognition package (t, r, b, l)
    to (x1, y1, w, h).
    '''
    return [(bbox[3], bbox[0], bbox[1] - bbox[3], bbox[2] - bbox[0]) for bbox in bboxes]

def std2facereg(bboxes):
    '''
    Convert bounding boxes parameters of 
    (x1, y1, w, h) to face_recognition 
    package (t, r, b, l).
    '''
    return [(bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0]) for bbox in bboxes]