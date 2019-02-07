import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parents[1]))
from pkg_utils import facerec2std

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image


def load_retinanet():
    MODEL_PATH = Path(__file__).parents[2] / 'models/retinanet.h5'

    assert MODEL_PATH.is_file(), f'Inference model is missing. ' + \
        f'Expected to find model at {MODEL_PATH.resolve()}'

    model = models.load_model(str(MODEL_PATH), backbone_name='resnet50')
    return model


def detect_faces_ret(frame, model, threshold=0.5, std_coord=True):
    '''
    Detect faces using a pre-trained RetinaNet model.

    Inputs:
        frame: cv2 RGB image
        model: RetinaNet Keras Model
        threshold: Minimum threshold to consider an object to 
                   positively be a face
        std_coord: Boolean on whether output coordinates should be
                   'standardized' if True (i.e. of schema (x1, y1, w, h)).
                   Otherwise, output is of form (x1, y1, x2, y2).

    Returns:
        A NumPy array of bounding box predictions of shape (N x 4)
        where N is the number of positive predictions.
    '''
    # Preprocess frame and perform inference on frame
    image = preprocess_image(frame)
    image, scale = resize_image(image)
    boxes, scores, _ = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale    

    # Keep significant bounding boxes
    boxes = boxes[scores > threshold]

    # Convert bounding box coordinates (see docstring)
    if std_coord:
        std_boxes = np.zeros(boxes.shape)
        std_boxes[..., :2] = boxes[..., :2]
        std_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        std_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        boxes = std_boxes

    # Sort detect faces from left to right
    boxes = boxes[np.argsort(boxes[:, 0])]

    return boxes.astype(int)