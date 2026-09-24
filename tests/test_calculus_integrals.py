import math

import pytest

from morphosml import integrate, integrate_2d


def test_integrate_sine():
    # integral_0^pi sin(x) dx = 2.0
    val_simpson = integrate(math.sin, 0.0, math.pi, method="simpson", n=500)
    assert pytest.approx(2.0, rel=1e-5) == val_simpson

    val_trap = integrate(math.sin, 0.0, math.pi, method="trapezoidal", n=2000)
    assert pytest.approx(2.0, rel=1e-4) == val_trap

    val_38 = integrate(math.sin, 0.0, math.pi, method="simpson_38", n=600)
    assert pytest.approx(2.0, rel=1e-5) == val_38

    val_gauss = integrate(math.sin, 0.0, math.pi, method="gauss_legendre")
    assert pytest.approx(2.0, rel=1e-4) == val_gauss


def test_integrate_polynomial():
    # integral_0^3 x^2 dx = 3^3 / 3 = 9.0
    f = lambda x: x**2
    assert pytest.approx(9.0, rel=1e-6) == integrate(
        f, 0.0, 3.0, method="simpson", n=100
    )


def test_integrate_exponential():
    # integral_0^1 exp(x) dx = e - 1 = 1.7182818...
    expected = math.e - 1.0
    assert pytest.approx(expected, rel=1e-5) == integrate(
        math.exp, 0.0, 1.0, method="simpson"
    )


def test_integrate_2d():
    # integral_0^1 integral_0^1 (x + y) dx dy = 1.0
    f = lambda x, y: x + y
    val = integrate_2d(f, 0.0, 1.0, 0.0, 1.0, nx=50, ny=50)
    assert pytest.approx(1.0, rel=1e-5) == val
