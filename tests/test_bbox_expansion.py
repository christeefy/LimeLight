import numpy as np
from itertools import product

from limelight.pkg_utils import expand_bboxes

from hypothesis import given, assume
from bbox_test_utils import generate_std_coordinates
from bbox_test_utils import generate_frame_shape, generate_margins


def test_correctness_expand_bboxes():
    frames = [np.empty(shape=(1080, 1920, 3))]
    margins = [1.1, 0.2, (2, 2), (4, 0.1)]
    bboxes = [[
        [10, 10, 10, 10],
        [1800, 1050, 80, 21]
    ]]

    answers = [
        [
            [10, 10, 11, 11],
            [1800, 1050, 88, 23]
        ],
        [
            [10, 10, 2, 2],
            [1800, 1050, 16, 4]
        ],
        [
            [10, 10, 20, 20],
            [1800, 1050, 120, 30]
        ],
        [
            [10, 10, 40, 1],
            [1800, 1050, 120, 2]
        ]
    ]

    # Convert values into numpy arrays
    bboxes = [np.array(bbox) for bbox in bboxes]
    answers = [np.array(ans) for ans in answers]

    for (f, m, bbox), ans in zip(product(frames, margins, bboxes), answers):
        assert np.all(ans == expand_bboxes(frame=f,
                                           bboxes=bbox.copy(),
                                           margins=m))


@given(shape=generate_frame_shape(),
       bboxes=generate_std_coordinates(),
       margins=generate_margins())
def test_fuzz_expand_bboxes(shape, bboxes, margins):
    bboxes = np.array(bboxes)
    assume(all(bboxes[..., 0] < shape[1]))
    assume(all(bboxes[..., 1] < shape[0]))

    expand_bboxes(frame=np.empty(shape),
                  bboxes=bboxes,
                  margins=margins)
