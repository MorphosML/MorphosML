import math

import pytest

from morphosml import derivative, differentiable


def test_first_derivative_polynomial():
    # f(x) = x^3 - 2x + 5 -> f'(x) = 3x^2 - 2
    f = lambda x: x**3 - 2.0 * x + 5.0
    val = derivative(f, 2.0)
    expected = 3.0 * (2.0**2) - 2.0  # 10.0
    assert pytest.approx(expected, rel=1e-5) == val


def test_first_derivative_transcendental():
    # f(x) = sin(x) -> f'(x) = cos(x)
    assert pytest.approx(1.0, abs=1e-6) == derivative(math.sin, 0.0)
    assert pytest.approx(0.0, abs=1e-6) == derivative(math.sin, math.pi / 2.0)
    assert pytest.approx(-1.0, abs=1e-6) == derivative(math.sin, math.pi)

    # f(x) = exp(x) -> f'(1) = e
    assert pytest.approx(math.e, rel=1e-5) == derivative(math.exp, 1.0)


def test_second_derivative():
    # f(x) = x^4 -> f''(x) = 12x^2
    f = lambda x: x**4
    val = derivative(f, 3.0, order=2)
    expected = 12.0 * (3.0**2)  # 108.0
    assert pytest.approx(expected, rel=1e-4) == val

    # f(x) = cos(x) -> f''(0) = -cos(0) = -1.0
    assert pytest.approx(-1.0, abs=1e-5) == derivative(math.cos, 0.0, order=2)


def test_third_derivative():
    # f(x) = x^3 + 5x^2 -> f'''(x) = 6.0
    f = lambda x: x**3 + 5.0 * (x**2)
    val = derivative(f, 4.0, order=3)
    assert pytest.approx(6.0, rel=1e-3) == val


def test_differentiable_decorator():
    @differentiable
    def my_func(x):
        return x**2 + 3.0 * x + 1.0

    # Call directly
    assert my_func(2.0) == 11.0

    # First derivative: 2x + 3 = 7.0 at x=2
    assert pytest.approx(7.0, rel=1e-5) == my_func.derivative(2.0)

    # Second derivative: 2.0
    assert pytest.approx(2.0, rel=1e-4) == my_func.derivative(2.0, order=2)
