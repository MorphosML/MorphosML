# Contributing to MorphosML

Thank you for your interest in contributing to **MorphosML**! We welcome contributions to our High-Performance Computing C++ core, Python bindings, calculus engine, data ingestion pipeline, and documentation.

---

## 1. Development Prerequisites

Ensure your system has the following dependencies installed:
- **C++ Compiler**: GCC (g++ 9+) or Clang (clang++ 10+) with full **C++17** support.
- **CMake**: Version 3.15 or higher (`cmake --version`).
- **Python**: Version 3.12 or higher.
- **Python Tooling**: `pytest`, `ruff`, `black`, `pybind11`, `setuptools`, `wheel`.

---

## 2. Environment Setup

Clone the repository and set up a local Python virtual environment:

```bash
git clone https://github.com/MorphosML/MorphosML.git
cd MorphosML

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest ruff black
```

---

## 3. Building the C++ Core

MorphosML compiles its native C++ engine via pybind11. To compile the extension in-place for local development:

```bash
# Build C++ extension in-tree
python setup.py build_ext --inplace

# Install in editable mode without build isolation
pip install -e . --no-build-isolation --no-deps
```

---

## 4. Code Quality & Standards

We enforce strict formatting and linting rules across all Python and C++ source code.

### Python Formatting
```bash
# Format Python code with Black (py312 target)
black --target-version py312 src/ tests/

# Run Ruff linter
ruff check src/ tests/
```

### C++ Standards
- Standard: **C++17** (`-std=c++17`).
- Memory Safety: Follow the **Rule of 5** for classes managing resources; prefer RAII and smart pointers over raw memory allocation where applicable.
- In-place Buffers: Use contiguous 1D arrays for matrices; avoid `vector<vector<double>>`.
- Documentation: Add Doxygen comments (`/** ... */`) to all public classes, structs, methods, and functions in `cpp/include/morphosml/`.

---

## 5. Running Tests & Automated CI Validation

Before committing, run the full test suite and CI validation check:

```bash
# Run unit tests directly with pytest
pytest tests/ -v

# Run the automated all-in-one CI validation script
./scripts/ci_deploy.sh --check-only
```

All 61+ unit tests must pass cleanly (100% green), and both Ruff and Black must report zero errors or warnings.

---

## 6. Git Commit & Pull Request Guidelines

1. **Focused Commits**: Keep commits atomic and focused on a single feature, bugfix, or documentation improvement.
2. **Clear Commit Messages**: Use conventional commit prefixes:
   - `feat:` for new capabilities or algorithms.
   - `fix:` for bug fixes.
   - `docs:` for documentation updates.
   - `perf:` for performance optimizations.
   - `refactor:` for code restructuring.
3. **Open an Issue / PR**: Reference any relevant GitHub issue numbers in your PR title or description (e.g., `fixes #12`).
