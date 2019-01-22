from pathlib import Path
import pandas as pd
import numpy as np
from .data_config import WIDER_FACE_IMGS_DIR, WIDER_FACE_LABELS_DIR

def _load_widerface_images(src):
    '''
    Loads the images from the WIDER FACE dataset.
    Returns a pd DataFrame with columns ['filepath', 'file_name'].
    '''
    
    filepaths = [str(x) for x in Path(src).glob('**/*.jpg')]
    
    assert len(filepaths), 'Directory contains no JPG images.'
    
    df = pd.DataFrame(filepaths, columns=['filepath'])
    df['file_name'] = df['filepath'].apply(lambda x: x.split('/')[-1])
    
    return df


def _parse_widerface_labels(src, include_metalabels=False):
    '''
    Parses the ground truths values of the WIDER FACE dataset from the 
    annotated text files into a pd DataFrame. DataFrame contains 'bboxes' which 
    is a numpy array of bounding boxes coordinates of 
    (x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose).
    
    Inputs:
        src: Path to label text file.
        include_metalabels: Boolean to include additional labels other than 
                            bounding box coordinates.
    
    Output:
        Pandas DataFrame with columns ['file_name', 'bboxes'].
    '''
    assert src[-4:].lower() == '.txt', 'Please provide a text file of annotations.'
    
    with open(src, 'r+') as f:
        # Create collector dataframe
        _columns = ['file_name', 'bboxes']
        df = pd.DataFrame(columns=_columns)
        
        # Parse ground truths for each image
        for line in f:
            file_name = line[:-1].split('/')[-1]
            
            assert isinstance(file_name, str), 'File name string may contain more than one "/".'
            
            # Parse and store bounding boxes as a 2D numpy array
            bboxes = np.empty((0, 10 if include_metalabels else 4), dtype=int)
            for _ in range(int(f.readline())):
                values = np.array([int(x) for x in f.readline().split(' ')[:-1]], dtype=int, ndmin=2)
                
                if not include_metalabels:
                    values = values[:, :4]
                
                bboxes = np.append(bboxes, values, axis=0)
            
            # Append to collector dataframe
            _df = pd.DataFrame([(file_name, bboxes)], columns=_columns)
            df = df.append(_df)
            
        return df
    

def create_wider_face_dataframe(image_src=WIDER_FACE_IMGS_DIR, 
                                label_src=WIDER_FACE_LABELS_DIR):
    '''
    Loads the WIDER FACE dataset as a pd DataFrame, containing the
    file paths to the images, the image names and the bounding box
    labels.
    '''

    assert Path(image_src).is_dir(), \
           f'Images directory does not exist.\nProvided directory is {image_src}.'
    assert Path(label_src).is_file(), \
           f'Labels file does not exists.\nProvided path is {label_src}.'

    images = _load_widerface_images(image_src)
    labels = _parse_widerface_labels(label_src)
    
    assert 'file_name' in images.columns and 'file_name' in labels.columns, \
        'Ensure that both dataframes contain the column "file_name".'
    
    return images.join(labels.set_index('file_name'), on='file_name')