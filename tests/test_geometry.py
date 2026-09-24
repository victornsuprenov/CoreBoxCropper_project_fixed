from coreboxcropper.geometry import (
    calculate_angle,
    calculate_perspective,
    expand_roi,
    is_parallel,
    is_perpendicular,
    is_valid_quadrilateral,
    line_intersection,
    order_points,
)


def test_order_points_returns_clockwise_corners():
    assert order_points([(90, 90), (10, 10), (90, 10), (10, 90)]) == [
        (10.0, 10.0),
        (90.0, 10.0),
        (90.0, 90.0),
        (10.0, 90.0),
    ]


def test_geometry_predicates():
    assert round(calculate_angle((0, 0), (0, 1), (1, 1))) == 90
    assert is_parallel(((0, 0), (10, 0)), ((1, 2), (10, 1)))
    assert is_perpendicular(((0, 0), (10, 0)), ((1, 0), (1, 10)))
    assert line_intersection(((0, 0), (10, 10)), ((0, 10), (10, 0))) == (5.0, 5.0)


def test_expand_roi_clamps_to_image():
    assert expand_roi((10, 10, 90, 90), (100, 100, 3), 0.1) == (2, 2, 98, 98)


def test_order_points_accepts_numpy_array():
    import numpy as np

    points = np.array([[90, 90], [10, 10], [90, 10], [10, 90]], dtype=np.int32)
    assert order_points(points) == [
        (10.0, 10.0),
        (90.0, 10.0),
        (90.0, 90.0),
        (10.0, 90.0),
    ]


def test_order_points_matches_top_left_top_right_bottom_right_bottom_left():
    points = [(30, 30), (200, 40), (210, 180), (10, 200)]
    assert order_points(points) == [
        (30.0, 30.0),
        (200.0, 40.0),
        (210.0, 180.0),
        (10.0, 200.0),
    ]


def test_calculate_perspective_margin_expands_output():
    _, size = calculate_perspective([(10, 10), (110, 10), (110, 60), (10, 60)], 0.1)
    assert size == (120, 60)


def test_order_points_repairs_cyclic_order_for_rotated_quad():
    points = [(10, 70), (90, 10), (110, 30), (30, 90)]
    ordered = order_points(points)
    assert is_valid_quadrilateral(ordered)
    assert ordered[0] == (90.0, 10.0)
    assert ordered[1] == (110.0, 30.0)
