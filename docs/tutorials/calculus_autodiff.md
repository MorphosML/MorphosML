# Calculus & Differentiable Programming Tutorial

MorphosML provides a compiled C++ numerical calculus engine designed for machine learning optimization, scientific computing, and algorithmic differentiation.

---

## 1. High-Precision Numerical Derivatives

MorphosML employs 5-point central difference stencils achieving $\mathcal{O}(h^4)$ truncation error.

### Mathematical Formulations
- **1st Order Derivative**:
  $$f'(x) \approx \frac{-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)}{12h}$$
- **2nd Order Derivative**:
  $$f''(x) \approx \frac{-f(x + 2h) + 16f(x + h) - 30f(x) + 16f(x - h) - f(x - 2h)}{12h^2}$$
- **3rd Order Derivative**:
  $$f'''(x) \approx \frac{f(x + 2h) - 2f(x + h) + 2f(x - h) - f(x - 2h)}{2h^3}$$

### Example
```python
import morphosml as mml
import numpy as np

f = lambda x: np.exp(x) * np.sin(x)

df = mml.derivative(f, x=0.5, order=1)
d2f = mml.derivative(f, x=0.5, order=2)
d3f = mml.derivative(f, x=0.5, order=3)

print(f"f'(0.5)  = {df:.8f}")
print(f"f''(0.5) = {d2f:.8f}")
print(f"f'''(0.5)= {d3f:.8f}")
```

---

## 2. Multivariate Optimization: Gradient & Hessian

### Gradient Vector $\nabla f(\mathbf{x})$
```python
# Sphere function: f(x, y, z) = x^2 + 2y^2 + 3z^2
sphere = lambda v: v[0]**2 + 2.0 * v[1]**2 + 3.0 * v[2]**2

grad = mml.gradient(sphere, [1.0, 2.0, 3.0])
print(f"Gradient: {grad}") # Expected: [2.0, 8.0, 18.0]
```

### Symmetric Hessian Matrix $H$
The Hessian captures the local curvature of the loss surface:
$$H_{i, j} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

```python
# Rosenbrock banana function
rosen = lambda v: (1.0 - v[0])**2 + 100.0 * (v[1] - v[0]**2)**2

H = mml.hessian(rosen, [1.0, 1.0])
print(f"Curvature Matrix H (2x2):\n{H.to_list()}")
# Eigenvalues of H indicate positive definiteness at local minima
```

---

## 3. Vector-Valued Functions: Jacobian Matrix $J$

For vector fields $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:
$$J_{i, j} = \frac{\partial f_i}{\partial x_j}$$

```python
# Polar to Cartesian coordinate transform: f(r, theta) = [r * cos(theta), r * sin(theta)]
def polar_to_cartesian(v):
    r, theta = v[0], v[1]
    return [r * np.cos(theta), r * np.sin(theta)]

J = mml.jacobian(polar_to_cartesian, [2.0, np.pi / 4])
print(f"Jacobian shape: ({J.rows()}, {J.cols()})")
print(f"J =\n{J.to_list()}")
```

---

## 4. Numerical Quadrature (Integration)

MorphosML supports multiple quadrature methods:
- **Composite Simpson's 1/3** (`"simpson"`): Standard high-accuracy numerical integration.
- **Composite Trapezoidal** (`"trapezoidal"`): Fast linear interpolation.
- **Gauss-Legendre Quadrature** (`"gauss_legendre"`): Optimal integration points with high algebraic degree.

```python
# Compute standard normal CDF integral from 0 to 1.96:
normal_pdf = lambda x: (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)

cdf = mml.integrate(normal_pdf, a=0.0, b=1.96, method="simpson")
print(f"P(0 <= Z <= 1.96) = {cdf:.5f}") # ~ 0.47500

# 2D Double Integral over rectangular region [0, 1] x [0, 2]
vol = mml.integrate_2d(lambda x, y: x * y**2, x_a=0.0, x_b=1.0, y_a=0.0, y_b=2.0)
print(f"Double integral: {vol:.4f}") # 4/3 ~ 1.3333
```

---

## 5. Numerical Limits & Singularity Analysis

Evaluate tricky limits, including removable singularities:
$$\lim_{x \to 0} \frac{\sin x}{x} = 1, \quad \lim_{x \to 0^+} \frac{1}{x} = +\infty$$

```python
# Removable singularity: function undefined at x=0
sinc = lambda x: np.sin(x) / x

lim_val = mml.limit(sinc, x_target=0.0)
print(f"lim sin(x)/x as x -> 0: {lim_val}") # 1.0

# Detailed diagnostics with evaluate_limit
res = mml.calculus.evaluate_limit(lambda x: 1.0 / x, x_target=0.0, direction="right")
print(f"Right-hand limit exists? {res.exists}, Is infinite? {res.is_infinite}")
```

---

## 6. The `@differentiable` Decorator

Equip any Python function with a fluent calculus API:

```python
@mml.differentiable
def loss(x):
    return x**4 - 3 * x**2 + 2

# Seamlessly call calculus methods directly on the function object!
print(f"Loss value at 2.0: {loss(2.0)}")
print(f"Derivative at 2.0: {loss.derivative(2.0)}")
print(f"Curvature at 2.0:  {loss.derivative(2.0, order=2)}")
print(f"Integral [0, 2]:   {loss.integrate(0.0, 2.0)}")
```
