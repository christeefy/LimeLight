import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K


def to_rgb(bgr):
    '''
    Convert an image's channels from
    BGR to RGB. Batch processing compatible.
    '''
    if bgr.shape[-1] != 3:
        raise AssertionError('Image can only have three colour channels.')
    return bgr[..., (2, 1, 0)]


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
    res = np.empty(shape=bboxes.shape, dtype=np.uint)

    res[..., 0] = bboxes[..., 3]
    res[..., 1] = bboxes[..., 0]
    res[..., 2] = bboxes[..., 1] - bboxes[..., 3]
    res[..., 3] = bboxes[..., 2] - bboxes[..., 0]

    return res


def std2facerec(bboxes):
    '''
    Convert bounding boxes parameters of
    (x1, y1, w, h) to face_recognition
    package (t, r, b, l).
    '''
    res = np.empty(shape=bboxes.shape, dtype=np.uint)

    res[..., 0] = bboxes[..., 1]
    res[..., 1] = bboxes[..., 0] + bboxes[..., 2]
    res[..., 2] = bboxes[..., 1] + bboxes[..., 3]
    res[..., 3] = bboxes[..., 0]

    return res


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
        assert margins > 0, 'margins must be greater than 0.'
        margins = (margins, margins)
    assert isinstance(margins, tuple), 'margins has to be a tuple or a float.'
    assert all(margin > 0 for margin in margins), \
        'margins must be greater than 0.'

    frame_h, frame_w, _ = frame.shape

    # Performs expansion while ensuring bbox specifications
    # do not exceed frame dimensions
    bboxes[..., 2] = np.clip(margins[0] * bboxes[..., 2],
                             1, frame_w - bboxes[..., 0])
    bboxes[..., 3] = np.clip(margins[1] * bboxes[..., 3],
                             1, frame_h - bboxes[..., 1])

    return bboxes.astype(int)


def reset_session():
    '''
    Clear existing backend session and create
    a new one with dynamic GPU memory allocation.
    '''
    K.clear_session()

    # Create new session configuration
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)


def frame_gen(cap, batch_size=1):
    '''
    Create a generator that outputs batch of video frames.

    Arguments:
        cap -- cv2 video capture object
        batch_size {int} -- Number of frames per batch
    '''

    while True:
        frames = []
        for _ in range(batch_size):
            if cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if len(frames):
                        yield np.array(frames)
                    return
                frames.append(frame)
        yield np.array(frames)


if __name__ == '__main__':
    cap = cv2.VideoCapture('demo/examples/short/dwayne_short_super.mp4')
    gen = frame_gen(cap, 4)
    for i, f in enumerate(gen):
        print(i, len(f))
