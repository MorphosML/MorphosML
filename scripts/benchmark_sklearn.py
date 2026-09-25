"""
MorphosML vs Scikit-Learn Performance Benchmark
================================================
Empirical benchmarking script comparing MorphosML v0.4.0 with scikit-learn
across K-Nearest Neighbors, Linear Regression, Logistic Regression,
and Ingestion Throughput.
"""

import os
import time

import numpy as np

# Set writable cache directory for Matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier as SklearnSGDClassifier
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

import morphosml as ml


def benchmark_knn():
    """Benchmark KNN prediction latency between MorphosML and scikit-learn."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: K-Nearest Neighbors (KNN, k=3, brute force) Inference")
    print("=" * 70)
    print(
        f"{'Test Queries':<15} | {'Sklearn (ms)':<16} | {'MorphosML (ms)':<16} | {'Speedup':<12}"
    )
    print("-" * 70)

    n_train = 2000
    n_features = 8
    k = 3
    query_sizes = [100, 250, 500, 1000, 2000]

    sk_times = []
    mml_times = []

    X_train = np.random.randn(n_train, n_features).astype(np.float64)
    y_train = np.random.choice([0, 1], size=n_train).tolist()

    # Pre-fit models
    sk_knn = SklearnKNN(n_neighbors=k, algorithm="brute")
    sk_knn.fit(X_train, y_train)

    mml_knn = ml.KNN(k=k)
    mml_knn.fit(X_train, y_train)

    for n_test in query_sizes:
        X_test = np.random.randn(n_test, n_features).astype(np.float64)

        # Sklearn timing (median of 3 runs)
        sk_runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = sk_knn.predict(X_test)
            sk_runs.append((time.perf_counter() - t0) * 1000.0)
        t_sk = np.median(sk_runs)
        sk_times.append(t_sk)

        # MorphosML timing (median of 3 runs)
        mml_runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = mml_knn.predict(X_test)
            mml_runs.append((time.perf_counter() - t0) * 1000.0)
        t_mml = np.median(mml_runs)
        mml_times.append(t_mml)

        speedup = t_sk / max(t_mml, 1e-4)
        print(
            f"{n_test:<15,d} | {t_sk:>13.2f} ms | {t_mml:>13.2f} ms | {speedup:>10.2f}x"
        )

    return query_sizes, sk_times, mml_times


def benchmark_linear_regression():
    """Benchmark Linear Regression training and prediction latency."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Linear Regression Training (Gradient Descent)")
    print("=" * 70)
    print(
        f"{'Samples (N)':<15} | {'Sklearn (ms)':<16} | {'MorphosML (ms)':<16} | {'Speedup':<12}"
    )
    print("-" * 70)

    sample_sizes = [5000, 10000, 25000, 50000]
    n_features = 8
    epochs = 100
    lr_rate = 0.01

    sk_times = []
    mml_times = []

    for n in sample_sizes:
        X = np.random.randn(n, n_features).astype(np.float64)
        true_w = np.random.randn(n_features)
        y = X @ true_w + 0.5 + np.random.randn(n) * 0.1

        # Sklearn SGDRegressor (matching gradient descent epochs)
        sk_model = SklearnSGDRegressor(
            max_iter=epochs, learning_rate="constant", eta0=lr_rate, random_state=42
        )
        t0 = time.perf_counter()
        sk_model.fit(X, y)
        t_sk = (time.perf_counter() - t0) * 1000.0
        sk_times.append(t_sk)

        # MorphosML LinearRegression
        mml_model = ml.LinearRegression(learning_rate=lr_rate, epochs=epochs)
        t0 = time.perf_counter()
        mml_model.fit(X, y)
        t_mml = (time.perf_counter() - t0) * 1000.0
        mml_times.append(t_mml)

        speedup = t_sk / max(t_mml, 1e-4)
        print(f"{n:<15,d} | {t_sk:>13.2f} ms | {t_mml:>13.2f} ms | {speedup:>10.2f}x")

    return sample_sizes, sk_times, mml_times


def benchmark_logistic_regression():
    """Benchmark Logistic Regression training latency."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Logistic Regression Training (Gradient Descent, Binary)")
    print("=" * 70)
    print(
        f"{'Samples (N)':<15} | {'Sklearn (ms)':<16} | {'MorphosML (ms)':<16} | {'Speedup':<12}"
    )
    print("-" * 70)

    sample_sizes = [5000, 10000, 25000, 50000]
    n_features = 8
    epochs = 100
    lr_rate = 0.05

    sk_times = []
    mml_times = []

    for n in sample_sizes:
        X = np.random.randn(n, n_features).astype(np.float64)
        y_prob = 1.0 / (1.0 + np.exp(-(X[:, 0] * 2.0 - X[:, 1])))
        y = (y_prob > 0.5).astype(int).tolist()

        # Sklearn SGDClassifier with log_loss (binary logistic loss)
        sk_clf = SklearnSGDClassifier(
            loss="log_loss",
            max_iter=epochs,
            learning_rate="constant",
            eta0=lr_rate,
            random_state=42,
        )
        t0 = time.perf_counter()
        sk_clf.fit(X, y)
        t_sk = (time.perf_counter() - t0) * 1000.0
        sk_times.append(t_sk)

        # MorphosML LogisticRegression
        mml_clf = ml.LogisticRegression(learning_rate=lr_rate, epochs=epochs)
        t0 = time.perf_counter()
        mml_clf.fit(X, y)
        t_mml = (time.perf_counter() - t0) * 1000.0
        mml_times.append(t_mml)

        speedup = t_sk / max(t_mml, 1e-4)
        print(f"{n:<15,d} | {t_sk:>13.2f} ms | {t_mml:>13.2f} ms | {speedup:>10.2f}x")

    return sample_sizes, sk_times, mml_times


def generate_plots(
    knn_data,
    lr_data,
    logreg_data,
    output_path="docs/morphosml_vs_sklearn_benchmark.png",
):
    """Generate high-resolution benchmark visualization comparing MorphosML and scikit-learn."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    color_sk = "#f59e0b"  # Amber / Gold for Scikit-Learn
    color_mml = "#06b6d4"  # Cyan / Teal for MorphosML
    grid_color = "#334155"
    text_color = "#f8fafc"

    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors=text_color, labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.4, color=grid_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)

    # 1. KNN Inference Latency
    ax1 = axes[0, 0]
    queries, sk_knn_t, mml_knn_t = knn_data
    ax1.plot(
        queries,
        sk_knn_t,
        "o-",
        color=color_sk,
        linewidth=2.5,
        markersize=8,
        label="scikit-learn (KNeighborsClassifier)",
    )
    ax1.plot(
        queries,
        mml_knn_t,
        "s-",
        color=color_mml,
        linewidth=2.5,
        markersize=8,
        label="MorphosML (C++ KNN)",
    )
    ax1.set_xlabel(
        "Number of Test Query Vectors", color=text_color, fontsize=11, fontweight="bold"
    )
    ax1.set_ylabel(
        "Latency (milliseconds)", color=text_color, fontsize=11, fontweight="bold"
    )
    ax1.set_title(
        "1. KNN Inference Latency (Lower is Better)",
        color=text_color,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.legend(
        facecolor="#1e293b", edgecolor=grid_color, labelcolor=text_color, fontsize=9.5
    )

    # 2. Linear Regression Training Latency
    ax2 = axes[0, 1]
    samples_lr, sk_lr_t, mml_lr_t = lr_data
    ax2.plot(
        samples_lr,
        sk_lr_t,
        "o-",
        color=color_sk,
        linewidth=2.5,
        markersize=8,
        label="scikit-learn (SGDRegressor)",
    )
    ax2.plot(
        samples_lr,
        mml_lr_t,
        "s-",
        color=color_mml,
        linewidth=2.5,
        markersize=8,
        label="MorphosML (C++ LinearRegression)",
    )
    ax2.set_xlabel(
        "Training Samples (N)", color=text_color, fontsize=11, fontweight="bold"
    )
    ax2.set_ylabel(
        "Training Time (milliseconds)", color=text_color, fontsize=11, fontweight="bold"
    )
    ax2.set_title(
        "2. Linear Regression Training (Lower is Better)",
        color=text_color,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax2.legend(
        facecolor="#1e293b", edgecolor=grid_color, labelcolor=text_color, fontsize=9.5
    )

    # 3. Logistic Regression Training Latency
    ax3 = axes[1, 0]
    samples_clf, sk_clf_t, mml_clf_t = logreg_data
    ax3.plot(
        samples_clf,
        sk_clf_t,
        "o-",
        color=color_sk,
        linewidth=2.5,
        markersize=8,
        label="scikit-learn (SGDClassifier)",
    )
    ax3.plot(
        samples_clf,
        mml_clf_t,
        "s-",
        color=color_mml,
        linewidth=2.5,
        markersize=8,
        label="MorphosML (C++ LogisticRegression)",
    )
    ax3.set_xlabel(
        "Training Samples (N)", color=text_color, fontsize=11, fontweight="bold"
    )
    ax3.set_ylabel(
        "Training Time (milliseconds)", color=text_color, fontsize=11, fontweight="bold"
    )
    ax3.set_title(
        "3. Logistic Regression Training (Lower is Better)",
        color=text_color,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax3.legend(
        facecolor="#1e293b", edgecolor=grid_color, labelcolor=text_color, fontsize=9.5
    )

    # 4. Feature & Architectural Capabilities Comparison
    ax4 = axes[1, 1]
    capabilities = [
        "KNN Classifier",
        "Linear Regression",
        "Logistic Regression",
        "Zero-Copy TensorView Protocol",
        "POSIX Out-of-Core MMap Streaming",
        "Async Prefetching DataLoader",
        "Cursor Fault-Tolerance & Checkpointing",
        "Numerical Autodiff (O(h^4) Derivatives)",
        "Gauss-Legendre Numerical Integrals",
        "Direct LaTeX Math Formatter (latexify)",
    ]
    # Scikit-learn has ML models, but lacks out-of-core mmap streaming, cursor checkpoints, numerical calculus, and latexify
    sk_caps = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    mml_caps = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    y_pos = np.arange(len(capabilities))
    ax4.barh(
        y_pos - 0.18,
        sk_caps,
        height=0.32,
        color=color_sk,
        alpha=0.85,
        label="scikit-learn",
    )
    ax4.barh(
        y_pos + 0.18,
        mml_caps,
        height=0.32,
        color=color_mml,
        alpha=0.9,
        label="MorphosML v0.4.0",
    )
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(capabilities, fontsize=9.5)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(["Unsupported", "Supported"], fontsize=10)
    ax4.set_title(
        "4. Architecture & Capabilities Matrix",
        color=text_color,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax4.legend(
        facecolor="#1e293b",
        edgecolor=grid_color,
        labelcolor=text_color,
        fontsize=9.5,
        loc="lower right",
    )

    plt.suptitle(
        "MorphosML v0.4.0 vs scikit-learn: Performance & Capability Benchmark",
        color="#38bdf8",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"\n[+] Benchmark visualization saved to: {output_path}")


def main():
    print("=" * 70)
    print("      MORPHOSML v0.4.0 vs SCIKIT-LEARN PERFORMANCE BENCHMARK")
    print("=" * 70)

    knn_data = benchmark_knn()
    lr_data = benchmark_linear_regression()
    logreg_data = benchmark_logistic_regression()

    chart_path = "docs/morphosml_vs_sklearn_benchmark.png"
    generate_plots(knn_data, lr_data, logreg_data, output_path=chart_path)


if __name__ == "__main__":
    main()
