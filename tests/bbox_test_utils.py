import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp


def _generate_sorted_tuples():
    return (
        st.tuples(st.integers(min_value=1, max_value=2**31),
                  st.integers(min_value=1, max_value=2**31),
                  st.integers(min_value=1, max_value=2**31),
                  st.integers(min_value=1, max_value=2**31))
        .map(sorted)
    )


def generate_facerec_coordinates():
    return (
        st.lists(
            _generate_sorted_tuples()
            .map(lambda x: (x[0], x[3], x[1], x[2])),
            min_size=1
        )
    )


def generate_std_coordinates():
    return st.lists(_generate_sorted_tuples(), min_size=1)


def generate_frame():
    return st.tuples(
        st.integers(min_value=1, max_value=2**16),
        st.integers(min_value=1, max_value=2**16),
        st.integers(min_value=3, max_value=3))


def generate_margins():
    unit_margin = st.floats(min_value=0, exclude_min=True, width=32,
                            allow_nan=False, allow_infinity=False)
    tuple_margin = st.tuples(unit_margin, unit_margin)

    return unit_margin | tuple_margin
