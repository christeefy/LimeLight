import numpy as np

def facereg2std(bboxes):
    '''
    Convert bounding boxes parameters of 
    face_recognition package (t, r, b, l)
    to (x1, y1, w, h).
    '''
    return np.array([(bbox[3], bbox[0], bbox[1] - bbox[3], bbox[2] - bbox[0]) for bbox in bboxes])


def std2facereg(bboxes):
    '''
    Convert bounding boxes parameters of 
    (x1, y1, w, h) to face_recognition 
    package (t, r, b, l).
    '''
    return np.array([(bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0]) for bbox in bboxes])


def generate_label_map(frame, bboxes, rf, perturb=False):
    '''
    Map the bounding boxes onto frame coordinates.
    
    Inputs:
        frame: An H x W x 3 NumPy array
        bboxes: A list of bounding boxes of schema (x1, y1, w, h)
        rf: Reduction factor
        perturb: Perturb the label map assignment to account for 
                 slight discrepancies of ground truth label to 
                 predictions.
    
    Returns:
        A NumPy array label map of shape (H/f x W/f x 5) where f is 
        the reduction factor. Schema follows (objectness, x1, y1, w, h).
    '''
    def has_collisions(bboxes):
        assert len(np.array(bboxes).shape) == 2
        
        if len(bboxes[:, :2]) == len(np.unique(bboxes[:, :2], axis=1)):
            return False
        else:
            return True
    
    img_h, img_w, _ = frame.shape
    
    # Convert bboxes coordinates to (x_c, y_c, w, h)
    bboxes = np.array(bboxes)
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] // 2
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] // 2
    
    # Normalize scales
    _img_h = img_h // rf
    _img_w = img_w // rf
    _bboxes = bboxes // rf

    # Check if there are location collisions amongst normalized 
    # bounding boxes
    if has_collisions(_bboxes):
        raise Exception('Collision occured.')

    # Populates label map
    label_map = np.zeros((_img_h, _img_w, 5))

    for _bbox in _bboxes: 
        perturb_range = [-1, 0, 1] if perturb else [0]
        # Perturb the insertion
        for i in perturb_range:
            for j in perturb_range:               
                label_map[np.clip(0, _bbox[1] + j, _img_w), 
                          np.clip(0, _bbox[0] + i, _img_h), ...] = np.append([1], _bbox)  

    return label_map
