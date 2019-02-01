import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def mark_faces_plt(rgb, bboxes):
    '''
    Apply a bounding box around faces in the image.
    
    Inputs:
        img_path: File path to image
        bboxes: NumPy array of bounding boxes.
   
    Output:
        An annotated Matplotlib figure.
    '''
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    
    # Annotate image with bounding boxes
    for bbox in bboxes:
        x1, y1, w, h, _ = np.split(bbox, [1, 2, 3, 4])
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
    return fig


def mark_faces_cv2(frame, bboxes, recognized_faces=None):
    '''
    Apply bounding boxes around faces using cv2.

    Note: Not a pure function.

    Inputs:
        img: cv2 image
        bboxes: NumPy array of bounding boxes.
        recognized_faces: A Boolean list of whether a face
                          is recognized.
    '''
    if not len(bboxes): return

    if recognized_faces is None:
        recognized_faces = [False] * len(bboxes)

    for bbox, recognized in zip(bboxes, recognized_faces):
        x1, y1, w, h, _ = np.split(bbox, [1, 2, 3, 4])
        cv2.rectangle(frame,
                      (x1, y1), 
                      (x1 + w, y1 + h), 
                      (0, 255, 0) if recognized else (255, 0, 0), 
                      2)


def blur_faces(rgb, bboxes, recognized_faces=None, blur_mode='gaussian', *args):
    '''
    Blur faces in a cv2 image. 

    Inputs:
        rgb:              A cv2 RGB image
        bboxes:           Bounding boxes coordinates 
                          for faces (x1, y1, w, h)
        recognized_faces: A Boolean list of whether a face
                          is recognized.
        blur_mode:        A string denoting blurring method.
                          Valid values: {'gaussian', 'median'}

    Returns:
        A cv2 RGB image with the same properties of `rgb` 
        with faces blurred.
    '''
    blur_func_key = {
        'gaussian': cv2.GaussianBlur,
        'median': cv2.medianBlur
    }

    assert blur_mode in blur_func_key.keys()
    
    if not len(bboxes): return rgb
    if recognized_faces is None:
        recognized_faces = [False] * len(bboxes)

    for bbox, recognized in zip(bboxes, recognized_faces):
        if recognized: continue

        x_slice = slice(bbox[0], bbox[0] + bbox[2])
        y_slice = slice(bbox[1], bbox[1] + bbox[3])

        face = rgb[y_slice, x_slice]
        face = blur_func_key[blur_mode](face, *args)

        rgb[y_slice, x_slice] = face

    return rgb