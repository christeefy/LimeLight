import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parents[1]))
from pkg_utils import facerec2std

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image, \
    compute_resize_scale


def load_retinanet():
    MODEL_PATH = Path(__file__).parents[2] / 'models/retinanet.h5'

    assert MODEL_PATH.is_file(), f'Inference model is missing. ' + \
        f'Expected to find model at {MODEL_PATH.resolve()}'

    model = models.load_model(str(MODEL_PATH), backbone_name='resnet50')
    return model


def resize_image_batch(frames):
    '''
    Wrapper function to resize a batch of images.
    Assumption is that images are of the same shape.
    '''
    scale = compute_resize_scale(frames[0].shape)
    frames = np.array(list(map(lambda x: x[0],
                               [resize_image(frame) for frame in frames])))
    return frames, scale


def threshold_factor_map(frame_shape, min_factor=0.5,
                         pad=0.1, interp='linear'):
    '''
    Create a map that scales the threshold
    on the border of the image.

    Values at the frame borders will have the lowest factor of `min_factor`.
    These are scaled to 1 using `interp` method and this scaling on each
    side of the frame up to an amount specified by `pad` (see below).

    Inputs:
        frame_shape: Input frame shape (h, w, c)
        min_factor: Lower bound of threshold factor.
        pad: Percentage of largest image dimension to perform scaling.
        interp: Interpolation method. Available methods: {'linear'}

    Returns:
        A NumPy factor map with the same width and height of the
        input frame.
    '''
    assert isinstance(frame_shape, tuple), \
        'Ensure that the image shape is provided, not the image itself.'
    assert interp in ['linear'], 'Invalid interpolation method.'

    # Absolute value, relative to largest frame dimension
    pad_abs = int(pad * max(frame_shape[:2]))

    # Initialize placeholders for map, the edge and corner objects
    map = np.ones(frame_shape[:2])
    edge_factor = np.linspace(min_factor, 1, pad_abs)
    corner_factor = np.zeros((pad_abs, pad_abs))

    # Fill in corner_factor
    for i in range(pad_abs):
        corner_factor[i, :(i + 1)] = edge_factor[:(i + 1)]
    corner_factor += corner_factor.T
    for i in range(len(corner_factor)):
        corner_factor[i, i] /= 2

    # Apply corner factor to all four corners
    # in the following order: (UL, UR, LL, LR)
    map[:pad_abs, :pad_abs] = corner_factor
    map[:pad_abs, -pad_abs:] = np.fliplr(corner_factor)
    map[-pad_abs:, :pad_abs] = np.flipud(corner_factor)
    map[-pad_abs:, -pad_abs:] = np.flipud(np.fliplr(corner_factor))

    # Apply edge factors to remaining edges
    # in the following order: (T, L, R, B)
    map[:pad_abs, pad_abs:-pad_abs] = edge_factor[:, np.newaxis]
    map[pad_abs:-pad_abs, :pad_abs] = edge_factor[np.newaxis, :]
    map[pad_abs:-pad_abs, -pad_abs:] = np.fliplr(edge_factor[np.newaxis, :])
    map[-pad_abs:, pad_abs:-pad_abs] = np.flipud(edge_factor[:, np.newaxis])

    return map


def is_above_threshold_map(bboxes, scores, threshold_map):
    '''
    Check whether each prediction is above the threshold map.

    Inputs:
        bboxes: An array of bounding boxes of schema (x1, y1, x2, y2).
        scores: An array of scores with the same length as `boxes`.
        threshold_map: A factor map.

    Returns:
        A Boolean np.ndarray of same shape as scores.
    '''
    assert len(bboxes) == len(scores), \
        'Boxes and scores need to have the same length.'

    # Calculate the centers (y, x) of each bounding box
    centers = (
        ((bboxes[..., 3] + bboxes[..., 1]) // 2).astype(int),
        ((bboxes[..., 2] + bboxes[..., 0]) // 2).astype(int)
    )

    # Map localized scores to threshold map
    return scores > threshold_map[centers]


def detect_faces_ret(frames, model, threshold=0.5,
                     std_coord=True, apply_threshold_map=True):
    '''
    Detect faces using a pre-trained RetinaNet model.

    Inputs:
        frames: A NumPy array of shape (b, w, h, c)
        model: RetinaNet Keras Model
        threshold: Minimum threshold to consider an object to
                   positively be a face
        std_coord: Boolean on whether output coordinates should be
                   'standardized' if True (i.e. of schema (x1, y1, w, h)).
                   Otherwise, output is of form (x1, y1, x2, y2).
        apply_threshold_map: Boolean on whether to apply a threshold factor
                             map. Defaults to True.

    Returns:
        A NumPy array of bounding box predictions of shape (N x 4)
        where N is the number of positive predictions.
    '''
    # Preprocess frames and perform inference on frames
    images = preprocess_image(frames)
    images, scale = resize_image_batch(images)
    boxes, scores, _ = model.predict_on_batch(images)
    boxes /= scale

    # Process outputs for each frame prediction
    boxes_list = []
    for box, score in zip(boxes, scores):
        # Filter out null-ish values in prediction output
        box = box[score >= 0]
        score = score[score >= 0]

        # Keep significant bounding boxes
        if apply_threshold_map:
            threshold_map = threshold * threshold_factor_map(frames[0].shape)
            sig_idx = is_above_threshold_map(box, score, threshold_map)
            box = box[sig_idx]
        else:
            box = box[scores > threshold]

        # Convert bounding box coordinates (see docstring)
        if std_coord:
            std_box = np.zeros(box.shape)
            std_box[..., :2] = box[..., :2]
            std_box[:, 2] = box[:, 2] - box[:, 0]
            std_box[:, 3] = box[:, 3] - box[:, 1]
            box = std_box

        # Sort detect faces from left to right
        box = box[np.argsort(box[:, 0])].astype(int)
        boxes_list.append(box)

    return boxes_list
