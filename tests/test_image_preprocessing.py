from hypothesis import given
from hypothesis import strategies as st
import hypothesis.extra.numpy as hnp

import numpy as np
from limelight.pkg_utils import to_rgb


def generate_hypothesis_numpy_array():
    shape = st.tuples(
        st.integers(min_value=1, max_value=2**6),
        st.integers(min_value=1, max_value=2**6),
        st.integers(min_value=1, max_value=2**6),
        st.integers(min_value=3, max_value=3))

    int_arr = (
        hnp.arrays(np.int,
                   shape=shape)
    )

    float_arr = (
        hnp.arrays(np.float,
                   shape=st.tuples(
                       st.integers(min_value=1, max_value=2**6),
                       st.integers(min_value=1, max_value=2**6),
                       st.integers(min_value=1, max_value=2**6),
                       st.integers(min_value=3, max_value=3)),
                   elements=st.floats(allow_nan=False,
                                      allow_infinity=False))
    )

    return int_arr | float_arr


@given(frame=generate_hypothesis_numpy_array())
def test_correctness_to_rgb(frame):
    ans = to_rgb(frame.copy())

    assert np.all(frame[..., 1] == ans[..., 1])
    assert np.all(frame[..., 2] == ans[..., 0])
    assert np.all(frame[..., 0] == ans[..., 2])
