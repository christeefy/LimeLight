import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

def mark_faces_plt(img_path, bboxes):
    '''
    Apply a bounding box around faces in the image.
    
    Inputs:
        img_path: File path to image
        bboxes: NumPy array of bounding boxes.
   
    Output:
        An annotated Matplotlib figure.
    '''
    fig, ax = plt.subplots()
    
    # Parse image channels from BGR to RGB and display image
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    
    # Annotate image with bounding boxes
    for bbox in bboxes:
        x1, y1, w, h, _ = np.split(bbox, [1, 2, 3, 4])
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
    return fig


def mark_faces_cv2(frame, bboxes):
    '''
    Apply bounding boxes around faces using cv2.

    Note: Not a pure function.

    Inputs:
        img: cv2 image
        bboxes: NumPy array of bounding boxes.
    '''
    if not len(bboxes): return
    for bbox in bboxes:
        x1, y1, w, h, _ = np.split(bbox, [1, 2, 3, 4])
        cv2.rectangle(frame,
                      (x1, y1), 
                      (x1 + w, y1 + h), 
                      (255, 0, 0), 
                      2)


def blur_faces(frame, bboxes, blur_mode='gaussian', *args):
    '''
    Blur faces in a cv2 image. 

    Inputs:
        frame: A cv2 image
        bboxes: Bounding boxes coordinates 
                for faces (x1, y1, w, h)
        blur_mode: A string denoting blurring method.
                   Valid values: {'gaussian', 'median'}

    Returns:
        A cv2 image with the same properties of `frame` 
        with faces blurred.
    '''
    blur_func_key = {
        'gaussian': cv2.GaussianBlur,
        'median': cv2.medianBlur
    }

    assert blur_mode in blur_func_key.keys()

    if not len(bboxes): return frame

    for bbox in bboxes:
        x_slice = slice(bbox[0], bbox[0] + bbox[2])
        y_slice = slice(bbox[1], bbox[1] + bbox[3])

        face = frame[y_slice, x_slice]
        face = blur_func_key[blur_mode](face, *args)

        frame[y_slice, x_slice] = face

    return frame