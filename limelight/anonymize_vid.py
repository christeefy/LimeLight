import time
import argparse
import warnings
from pathlib import Path
from functools import partial

import cv2
import numpy as np
import face_recognition

import keras.backend as K

import sys
sys.path.append(str(Path(__file__).parent))

from pkg_utils import rgb_read, to_rgb, expand_bboxes
from viz import mark_faces_cv2, blur_faces
from face_detection.haar_cascade import detect_faces_hc
from face_detection.retinanet import detect_faces_ret, load_retinanet
from recognize_face import embed_face, recognize_face

def anonymize_vid(src, dst=None, known_faces_loc=None, 
                  use_retinanet=True, threshold=0.5,
                  lower_border_thresholds=True,
                  mark_faces=False, profile=False,
                  batch_size=1, expand_bbox=1.0):
    '''
    Anonymize a video by blurring unrecognized faces. 
    Writes a processed video to `dst`.

    Inputs:
        src:             Path to video.
        dst:             Path to save processsed video. If None, 
                         append 'mod' to src filename. 
        known_faces_loc: Directory containing JPG images of 
                         recognized faces not to blur.
        use_retinanet:   Use RetinaNet (True) or 
                         Viola Jones algorithm (False).
        threshold:       Threshold to consider an object a face. 
                         Applies to RetinaNet only.
        lower_border_thresholds:
                         Reduce the thresholds at the border of the images.
                         Applies to RetinaNet only.
        mark_faces:      Mark faces with bounding boxes. Default to False.
        profile:         Profiles code execution time (Boolean). 
        expand_bbox:     Expand bounding boxes height and width by a
                         factor of (w_factor, h_factor).
        batch_size:      Process these number of images per batches. 

    Returns nothing.
    '''
    assert src.split('.')[-1].lower() in ['mov', 'mp4'], \
           f'{src.split(".")[-1]} is not a valid video file format.'
    if dst is None:
        dst = '_mod.'.join(src.rsplit('.', 1))
    else:
        assert src.split('.')[-1].lower() == dst.split('.')[-1].lower(), \
            'Input and output file formats do not match .'

    # Record initial execution time
    if profile:
        start_time = time.time()

    # Define face detection function
    if use_retinanet:
        K.clear_session()
        retinanet = load_retinanet()
        detect_fn = partial(detect_faces_ret, 
                            model=retinanet, 
                            threshold=threshold,
                            apply_threshold_map=lower_border_thresholds)
    else:
        detect_fn = detect_faces_hc

    # Load video and its metadata
    cap = cv2.VideoCapture(src)
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    VID_FPS = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer stream
    Path(dst).parent.mkdir(exist_ok=True, parents=True)
    out = cv2.VideoWriter(dst, 
          cv2.VideoWriter_fourcc(*"mp4v"), VID_FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    # Get encodings of known faces
    perform_face_rec = False      # Boolean to perform facial recognition
    if known_faces_loc is not None:
        known_faces = [str(x) for x in Path(known_faces_loc).rglob('*.jpg')]
        if len(known_faces) == 0:
            warnings.warn('[WARNING] known_faces_loc contains no faces. JPG files only.')
        else:
            perform_face_rec = True
            # Create face encodings
            for filepath in known_faces:
                face_rgb = rgb_read(filepath)
                face_loc = detect_fn(face_rgb)
                recognized_face_encs = embed_face(face_rgb, face_loc)

    # Craete collector variables to store execution time
    if profile:
        detect_times = []
        recog_times = []

    # Video processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb = to_rgb(frame)

        # Detect faces
        if profile:
            _detect_start = time.time()
            preds = detect_fn(rgb)
            detect_times.append(time.time() - _detect_start)
        else:
            preds = detect_fn(rgb)

        if len(preds):
            # Expand bounding boxes to fully capture the face and head
            if expand_bbox is not None:
                preds = expand_bboxes(rgb, preds, expand_bbox)
                
            # Perform facial recognition
            # `recognized` is a boolean of recognized faces
            if perform_face_rec:
                if profile:
                    _recog_start = time.time()
                    face_encs = embed_face(rgb, preds)
                    recognized = recognize_face(face_encs, recognized_face_encs)
                    recog_times.append(time.time() - _recog_start)
                else:
                    face_encs = embed_face(rgb, preds)
                    recognized = recognize_face(face_encs, recognized_face_encs)

            # Add annotations on screen
            rgb = blur_faces(rgb=rgb, 
                             bboxes=preds, 
                             recognized_faces=recognized if perform_face_rec else None, 
                             blur_mode='pixelate')
            if mark_faces:
                mark_faces_cv2(rgb, preds, recognized if perform_face_rec else None)

        # Save frame
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    K.clear_session()
    cap.release()
    out.release()

    # Summarize execution times
    if profile:
        end_time = time.time()
        print(f'\nTotal processing runtime: {end_time - start_time:.1f} sec for ' + \
              f'{FRAME_COUNT} frames ({FRAME_COUNT / (end_time - start_time):.1f} fps).')
        print(f'Average detection time: {np.mean(detect_times):.4f} sec per frame ' + \
              f'for {len(detect_times)} frames ({1/np.mean(detect_times):.1f} fps).')
        print(f'Average recognition time: {np.mean(recog_times):.4f} sec per frame' + \
              f'for {len(recog_times)} frames ({1/np.mean(recog_times):.1f} fps).')

    print(f'\nAnonymized video saved at {dst}.')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Script to anonymize faces in a video.')
    parser.add_argument('src', help='Path to video.')
    parser.add_argument('--dst', 
                        help='[Optional] Path to save processed video.' + \
                             'If not specified, \'mod\' is append to `src` filename.')
    parser.add_argument('--known-faces-loc',
                        help='Directory containing JPG images of faces to not blur.')
    parser.add_argument('--batch-size',
                        help='Batch process video frames for increased computation speed. ' +
                             'Recommended for GPU only.', 
                        nargs='?', const=1, default=1, type=int)
    parser.add_argument('--vj',
                        help='Use Viola Jones algorithm in lieu of RetinaNet' +
                             'for faster but less accurate face detection.',
                        dest='retinanet', action='store_false')
    parser.add_argument('--mark-faces', 
                        help='Mark faces with bounding boxes. Default to False.', 
                        action='store_true')
    parser.add_argument('--profile', help='Boolean to profile code execution time.', 
                        action='store_true')
    parser.add_argument('--threshold', help='Threshold to consider an object a face. ' + 
                                            'Applies to RetinaNet only.',
                        nargs='?', default=0.5, const=0.5, type=int)
    parser.add_argument('--disable_border_thresholds', 
                        help='Disable threshold-lowering at the frame borders.',
                        action='store_false', dest='lower_border_thresholds')
    parser.add_argument('--expand_bbox', 
                        help='Expand bounding boxes height and width by a factor of' + 
                             '(w_factor, h_factor).',
                        nargs='?', type=int, default=1, const=1)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    anonymize_vid(src=args.src, dst=args.dst, 
                  known_faces_loc=args.known_faces_loc,
                  use_retinanet=args.retinanet,
                  batch_size=args.batch_size,
                  mark_faces=args.mark_faces,
                  profile=args.profile)


if __name__ == '__main__':
    main()