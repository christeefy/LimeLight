import numpy as np
from limelight.pkg_utils import facerec2std, std2facerec

from hypothesis import given
from bbox_test_utils import generate_facerec_coordinates
from bbox_test_utils import generate_std_coordinates


def test_correctness_facerec_coords_to_std_coords():
    '''
    Test correctness of facerec to std coordinate conversion.
    '''
    data = np.array([(100, 200, 120, 60), (500, 2500, 5100, 625)])
    ans = np.array([(60, 100, 140, 20), (625, 500, 1875, 4600)])
    assert np.allclose(facerec2std(data), ans)


def test_correctness_std_coords_to_facerec_coords():
    '''
    Test correctness of std to facerec coordinate conversion.
    '''
    data = np.array([(200, 130, 52, 40), (150, 3000, 500, 137)])
    ans = np.array([(130, 252, 170, 200), (3000, 650, 3137, 150)])
    assert np.allclose(std2facerec(data), ans)


@given(facerec=generate_facerec_coordinates())
def test_facerec2std2facerec_invariance(facerec):
    '''
    Test that facerec to/from std coordinates is invariant.
    '''
    facerec = np.array(facerec)
    assert np.allclose(std2facerec(facerec2std(facerec)), facerec)


@given(std=generate_std_coordinates())
def test_std2facerec2std_invariance(std):
    '''
    Test that facerec to/from std coordinates is invariant.
    '''
    std = np.array(std)
    assert np.allclose(facerec2std(std2facerec(std)), std)


@given(generate_facerec_coordinates())
def test_fuzz_facerec2std(bboxes):
    bboxes = np.array(bboxes)
    facerec2std(bboxes)


@given(generate_std_coordinates())
def test_fuzz_std2facerec(bboxes):
    bboxes = np.array(bboxes)
    std2facerec(bboxes)
