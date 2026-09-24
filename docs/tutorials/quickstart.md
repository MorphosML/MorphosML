# MorphosML Quickstart Tutorial

This hands-on tutorial walks you through installing MorphosML, working with high-performance linear algebra primitives, training models, streaming large datasets with `mmap`, and using the numerical calculus engine.

---

## 1. Installation

From source with editable C++ bindings:
```bash
git clone https://github.com/MorphosML/MorphosML.git
cd MorphosML
pip install -e . --no-build-isolation
```

---

## 2. High-Performance Linear Algebra

MorphosML matrices are backed by contiguous 1D flat buffers, providing bare-metal cache efficiency:

```python
import morphosml as mml
import numpy as np

# Create an owning contiguous matrix
A = mml.Matrix([[1.0, 2.0], [3.0, 4.0]])
B = mml.Matrix.identity(2)

# Cache-optimized matrix multiplication (i-k-j loop tiling)
C = A @ B
print(f"Matrix C shape: ({C.rows()}, {C.cols()})")

# Element access and modification
print(f"C[0, 1] = {C[0, 1]}")
C[0, 1] = 99.0

# Zero-copy interoperability with NumPy
np_arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
view = mml.TensorView(np_arr) # 0 memory allocations, 0 byte copies!
print(f"TensorView element: {view[1, 0]}")
```

---

## 3. Training a Machine Learning Model

Train a multivariate linear regression model using gradient descent:

```python
from morphosml import LinearRegression, train_test_split, r2_score, set_seed
import numpy as np

# Set deterministic seed
set_seed(42)

# Generate synthetic dataset: y = 3.5 * x_1 - 1.2 * x_2 + 0.5
X = np.random.randn(1000, 2)
y = 3.5 * X[:, 0] - 1.2 * X[:, 1] + 0.5 + 0.05 * np.random.randn(1000)

# Split into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit LinearRegression
model = LinearRegression(learning_rate=0.05, epochs=800)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"R² Score: {score:.4f}")
print(f"Learned weights: {model.weights.to_list()}, bias: {model.bias:.4f}")
```

---

## 4. Out-of-Core Dataset Streaming with `mmap`

Handle multi-gigabyte datasets without exhausting physical RAM:

```python
import morphosml as mml
import numpy as np

# 1. Export large data to .mldat binary format
raw_data = np.random.randn(50000, 64)
mml.data.MMapDataset.dump("dataset_large.mldat", raw_data)

# 2. Open via POSIX mmap (0 ms parse time!)
dataset = mml.data.MMapDataset("dataset_large.mldat")
print(f"Dataset shape: {dataset.shape}, Checksum: {dataset.checksum}")

# 3. Create an asynchronous, double-buffered DataLoader
loader = mml.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    seed=42,
    prefetch_batches=2 # Asynchronous background worker
)

for step, batch in enumerate(loader):
    # Process batch directly from kernel pages
    if step % 100 == 0:
        print(f"Step {step}: batch shape = ({batch.rows()}, {batch.cols()})")
```

---

## 5. Differentiable Programming & Calculus

Compute high-precision derivatives, gradients, and integrals directly in Python:

```python
import morphosml as mml
import numpy as np

# High-precision scalar derivatives (O(h^4) 5-point stencil)
f = lambda x: np.sin(x) * np.exp(x)
df = mml.derivative(f, x=1.0, order=1)
d2f = mml.derivative(f, x=1.0, order=2)
print(f"f'(1.0) = {df:.6f}, f''(1.0) = {d2f:.6f}")

# Multivariate Rosenbrock function gradient and Hessian
rosenbrock = lambda v: (1.0 - v[0])**2 + 100.0 * (v[1] - v[0]**2)**2
grad = mml.gradient(rosenbrock, [1.0, 1.0])
hess = mml.hessian(rosenbrock, [1.0, 1.0])
print(f"Gradient at minimum: {grad}")
print(f"Hessian matrix:\n{hess.to_list()}")

# Numerical Quadrature (Simpson's 1/3)
area = mml.integrate(lambda x: x**2, a=0.0, b=3.0)
print(f"Integral of x^2 on [0, 3]: {area:.4f}") # 9.0000

# Limits with removable singularities
lim = mml.limit(lambda x: np.sin(x) / x, x_target=0.0)
print(f"lim_{x->0} sin(x)/x = {lim}") # 1.0
```
