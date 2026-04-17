# MorphosML

MorphosML is a lightweight Machine Learning library built with Python and NumPy.

The goal of this project is to provide clear, simple, and educational implementations of core machine learning algorithms while maintaining practical usability.

---

## 🚀 Features

* Linear Regression (from scratch)
* Logistic Regression (from scratch)
* Simple and clean API inspired by scikit-learn
* Lightweight and easy to understand
* Ready for real usage and experimentation

---

## 📦 Installation

```bash
pip install morphosml
```

---

## 🧠 Example

### Logistic Regression

```python
from morphosml import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(learning_rate=0.0001, n_iters=2000)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
```

---

## 🎯 Project Goals

MorphosML is designed to be:

* Educational → understand how ML works internally
* Practical → usable in real experiments
* Lightweight → minimal dependencies
* Extensible → easy to expand and improve

---

## ⚡ Roadmap

### v0.2.0 (Next Release)

* Vectorization improvements
* Numba integration for performance
* KNN implementation
* Metrics module (MSE, Accuracy, etc.)
* BaseModel abstraction
* Improved testing and benchmarks

---

## 🤝 Contributing

Contributions are welcome.

If you want to improve performance, add new models, or enhance documentation, feel free to open an issue or submit a pull request.

---

## 📄 License

MIT License

---

## 🌍 Repository

https://github.com/MorphosML/MorphosML

---

## 💡 Final Note

This project started as a learning initiative and is evolving into a structured ML library.

Small steps lead to meaningful outcomes.
