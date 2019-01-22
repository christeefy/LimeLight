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


def mark_faces_cv2(img, bboxes):
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
        cv2.rectangle(img,
                      (x1, y1), 
                      (x1 + w, y1 + h), 
                      (255, 0, 0), 
                      2)