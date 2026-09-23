import numpy as np

from morphosml import gradient, hessian, jacobian


def test_gradient_quadratic():
    # f(x, y) = x^2 + 2y^2 -> grad f = [2x, 4y]
    def f(v):
        return v[0] ** 2 + 2.0 * (v[1] ** 2)

    grad = gradient(f, [3.0, -2.0])
    expected = np.array([6.0, -8.0])
    np.testing.assert_allclose(grad, expected, rtol=1e-5)


def test_gradient_rosenbrock():
    # Rosenbrock minimum at (1, 1) has grad = (0, 0)
    def rosenbrock(v):
        x, y = v[0], v[1]
        return (1.0 - x) ** 2 + 100.0 * ((y - x**2) ** 2)

    grad = gradient(rosenbrock, [1.0, 1.0])
    np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-5)


def test_jacobian():
    # f(x, y) = [x^2 + y, 3x - y^2]
    # J = [[2x, 1], [3, -2y]]
    def f(v):
        x, y = v[0], v[1]
        return [x**2 + y, 3.0 * x - y**2]

    J = jacobian(f, [2.0, 3.0])
    assert J.rows() == 2
    assert J.cols() == 2

    # Expected: [[4, 1], [3, -6]]
    expected = np.array([[4.0, 1.0], [3.0, -6.0]])
    J_np = np.asarray(J)
    np.testing.assert_allclose(J_np, expected, rtol=1e-5)


def test_hessian():
    # f(x, y) = x^2 + 3xy + 2y^2
    # d^2f/dx^2 = 2, d^2f/dy^2 = 4, d^2f/dxdy = 3
    def f(v):
        x, y = v[0], v[1]
        return x**2 + 3.0 * x * y + 2.0 * (y**2)

    H = hessian(f, [1.0, 1.0])
    assert H.rows() == 2
    assert H.cols() == 2

    expected = np.array([[2.0, 3.0], [3.0, 4.0]])
    H_np = np.asarray(H)
    np.testing.assert_allclose(H_np, expected, rtol=1e-4)
