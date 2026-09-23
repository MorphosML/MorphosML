import math

import pytest

from morphosml import limit
from morphosml.calculus import evaluate_limit


def test_limit_sin_x_over_x():
    # lim_{x -> 0} sin(x)/x = 1.0
    f = lambda x: math.sin(x) / x
    val = limit(f, 0.0)
    assert pytest.approx(1.0, rel=1e-5) == val


def test_limit_removable_singularity():
    # lim_{x -> 2} (x^2 - 4)/(x - 2) = 4.0
    f = lambda x: (x**2 - 4.0) / (x - 2.0)
    val = limit(f, 2.0)
    assert pytest.approx(4.0, rel=1e-5) == val


def test_limit_one_sided_infinite():
    # f(x) = 1/x -> right is +inf, left is -inf
    f = lambda x: 1.0 / x

    res_right = evaluate_limit(f, 0.0, direction="right")
    assert res_right.is_infinite
    assert res_right.value > 0

    res_left = evaluate_limit(f, 0.0, direction="left")
    assert res_left.is_infinite
    assert res_left.value < 0

    # Two-sided should raise RuntimeError
    with pytest.raises(RuntimeError):
        _ = limit(f, 0.0, direction="both")


def test_limit_continuous():
    f = lambda x: 2.0 * x + 5.0
    assert pytest.approx(11.0, rel=1e-5) == limit(f, 3.0)
