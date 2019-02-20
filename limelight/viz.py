import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

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


def blur_faces(rgb, bboxes, recognized_faces=None, blur_mode='pixelate', *args):
    '''
    Blur faces in a cv2 image. 

    Inputs:
        rgb:              A cv2 RGB image
        bboxes:           Bounding boxes coordinates 
                          for faces (x1, y1, w, h)
        recognized_faces: A Boolean list of whether a face
                          is recognized.
        blur_mode:        A string denoting blurring method.
                          Valid values: {'pixelate', 'gaussian', 'median'}

    Returns:
        A NumPy array representing the RGB image with the same properties
        of `rgb` with faces blurred.
    '''
    def pixelate(img, squeeze_ratio=16):
        '''
        Pixelates a img.

        Note: 
        Pixelation is done by smoothly shrinking the image
        and then rescaling the image to its original size 
        using a nearest neighbour interpolation of the pixels.

        Inputs:
            img: A cv2 image.
            squeeze_ratio: Length ratio of the shrunken image.
        '''
        # Swap H and W dimensions 
        *img_shape, _ = img.shape
        swapped_shape = (img_shape[1], img_shape[0])

        # Shrink image smoothly
        shrunk = (
            Image
            .fromarray(img)
            .resize(tuple(x // squeeze_ratio for x in swapped_shape), 
                    Image.BILINEAR)
        )

        # Resize image with NEAREST interpolation
        pixelated = shrunk.resize(swapped_shape, Image.NEAREST)

        return np.asarray(pixelated)

    # Create a mappiny for blur functions
    blur_func_key = {
        'pixelate': pixelate,
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