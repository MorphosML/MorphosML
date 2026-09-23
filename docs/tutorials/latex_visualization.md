# LaTeX Formatting & Jupyter Visualization Tutorial

MorphosML provides built-in LaTeX generation and interactive Jupyter Notebook visualization for linear algebra primitives and machine learning models.

---

## 1. Automatic Notebook Rendering

By default, importing `morphosml` registers the `_repr_latex_()` display hook for all core classes. In a Jupyter Notebook or Google Colab, simply evaluating a matrix, vector, or model renders beautiful mathematical typesetting:

```python
import morphosml as mml

mat = mml.Matrix([
    [1.5, 2.0, 3.2],
    [4.0, 5.1, 6.0],
    [7.8, 8.0, 9.4]
])

# In a Jupyter cell:
mat # Automatically renders as a LaTeX matrix!
```

---

## 2. Converting Matrices and Vectors to LaTeX

Generate raw LaTeX code for export to LaTeX documents, Markdown reports, or papers:

```python
import morphosml as mml

A = mml.Matrix.identity(3)
latex_code = mml.latex.matrix_to_latex(A)
print(latex_code)
# Output: \begin{pmatrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1\end{pmatrix}
```

### Automatic Ellipsis Truncation for Large Matrices

For large matrices, MorphosML automatically formats the display with `\dots`, `\vdots`, and `\ddots`:

```python
large_mat = mml.Matrix.random(100, 100)
code = mml.latex.matrix_to_latex(large_mat, max_rows=4, max_cols=4)
print(code)
# Output includes formatted \ddots and boundary elements cleanly truncated
```

---

## 3. Explaining Machine Learning Models

Render the exact analytical equation of a trained model:

### Linear Regression
```python
from morphosml import LinearRegression
import numpy as np

X = np.random.randn(100, 3)
y = 2.5 * X[:, 0] - 1.2 * X[:, 1] + 0.8 * X[:, 2] + 4.0

model = LinearRegression(learning_rate=0.01, epochs=1000).fit(X, y)

# Render model equation with custom variable names
latex_eq = mml.latex.linear_regression_to_latex(
    model, 
    feature_names=["\\text{age}", "\\text{income}", "\\text{debt}"]
)
print(latex_eq)
# Output: \hat{y} = 2.5000\,\text{age} - 1.2000\,\text{income} + 0.8000\,\text{debt} + 4.0000
```

### Logistic Regression
```python
from morphosml import LogisticRegression

X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int).tolist()

clf = LogisticRegression().fit(X, y)
print(clf._repr_latex_())
# Output: $$P(y=1|\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-(w_1 x_1 + w_2 x_2 + b)}}$$
```

---

## 4. The `@latexify` Decorator

Use the `@latexify` converter to wrap any function or object into an interactive displayable LaTeX component:

```python
from morphosml.latex import latexify

# Directly wrap and display in notebooks
display_obj = latexify(model)
```
