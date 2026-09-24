# MorphosML CLI Commands & Developer Cheatsheet

This reference provides a complete catalog of all command-line operations, scripts, build steps, and quality assurance tools available in MorphosML.

---

## Table of Contents
- [Installation Commands](#installation-commands)
- [Building & Compilation](#building--compilation)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Benchmarking & Profiling](#benchmarking--profiling)
- [CI/CD & Pre-Flight Automation](#cicd--pre-flight-automation)
- [Packaging & PyPI Deployment](#packaging--pypi-deployment)
- [Git & Branch Operations](#git--branch-operations)

---

## Installation Commands

### 1. Production Install from PyPI
Install the latest stable release of MorphosML via pip:
```bash
pip install morphosml
```

### 2. Local Editable Development Install
Compiles C++17 extensions in-place and links to your current working tree:
```bash
# Ensure your virtual environment is active
source .venv/bin/activate

# Install in editable mode without isolated build overhead
pip install -e . --no-build-isolation --no-deps
```

### 3. Install Development Dependencies
```bash
pip install -e ".[dev]"
# Or install individually:
pip install pytest ruff black build twine scikit-build-core pybind11 matplotlib
```

---

## Building & Compilation

### 1. Build via Python Build Frontend (`python -m build`)
Generates standardized source distributions (`.tar.gz`) and binary wheels (`.whl`):
```bash
# Clean previous build artifacts
rm -rf dist/ build/ *.egg-info

# Build both sdist and wheel
python -m build

# Build only source distribution (sdist)
python -m build --sdist

# Build only binary wheel
python -m build --wheel
```

### 2. Direct CMake & Ninja Compilation (C++ Core)
If working directly on the C++ headers and source files in `cpp/`:
```bash
# Create build directory
mkdir -p build && cd build

# Configure with Clang++ and Ninja
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..

# Compile all C++ targets
ninja

# Run tests or inspect generated shared libraries
ls -lh morphosml*.so
```

---

## Testing & Quality Assurance

### 1. Pytest Test Suite
MorphosML has a comprehensive unit test suite covering Linear Algebra, Ingestion, Calculus, and ML Models:
```bash
# Run full test suite with verbose output
pytest tests/ -v

# Run with concise single-line progress
pytest tests/ -q

# Run a specific test module
pytest tests/test_calculus_derivatives.py -v
pytest tests/test_tensor_view.py -v
pytest tests/test_mmap_dataset.py -v
pytest tests/test_linear_regression.py -v
pytest tests/test_latexify.py -v

# Run tests matching a keyword pattern
pytest -k "tensor_view or calculus" -v

# Run tests and stop on first failure
pytest tests/ -x
```

### 2. Static Analysis & Linting (`ruff`)
```bash
# Check code style, imports, and syntax across src and tests
ruff check src/ tests/

# Automatically apply safe fixes for detected lints
ruff check src/ tests/ --fix
```

### 3. Code Formatting (`black`)
```bash
# Check formatting without modifying files
black --check src/ tests/ scripts/

# Format all Python files to standard 88-character lines
black src/ tests/ scripts/
```

---

## Benchmarking & Profiling

### Run Version Comparison Benchmark (v0.2.0 vs v0.3.0)
MorphosML provides an automated benchmarking script that measures memory conversion latency, memory footprint overhead, and batch ingestion throughput, automatically rendering a 4-panel dark-mode dashboard figure:
```bash
python scripts/compare_versions.py
```
- **Output:** Outputs live terminal metrics table and writes **`docs/morphosml_v0.2_vs_v0.3_benchmark.png`**.

---

## CI/CD & Pre-Flight Automation

MorphosML includes a master pre-flight script **[`scripts/ci_deploy.sh`](../scripts/ci_deploy.sh)** that automates linting, formatting, compiling, and testing before code is committed or released:

### 1. Validation Only Mode (Safe Check)
Validates the entire repository without writing commits:
```bash
./scripts/ci_deploy.sh --check-only
```
- Verifies virtual environment.
- Re-compiles C++ bindings.
- Runs `black --check`.
- Runs `ruff check`.
- Runs all `pytest` unit tests (must pass 100%).

### 2. Automated Commit & Push Mode
Validates everything and automatically commits and pushes if all checks pass:
```bash
./scripts/ci_deploy.sh --auto-commit
```

---

## Packaging & PyPI Deployment

### 1. Validate Artifact Integrity (`twine check`)
Before uploading to PyPI, always inspect the generated distribution archives:
```bash
twine check dist/*
```
Expected output:
```text
Checking dist/morphosml-0.3.1-cp312-cp312-linux_x86_64.whl: PASSED
Checking dist/morphosml-0.3.1.tar.gz: PASSED
```

### 2. Upload Source Distribution to PyPI
```bash
twine upload dist/morphosml-0.3.1.tar.gz
```
- **Username:** `__token__`
- **Password:** Your PyPI API Token (`pypi-AgEI...`)

### 3. Upload with Duplicate Protection
If uploading multiple artifacts without erroring on already existing packages:
```bash
twine upload --skip-existing dist/*
```

---

## Git & Branch Operations

### 1. Release Tagging
To create a standard semantic version release tag:
```bash
# Create annotated tag
git tag -a v0.3.1 -m "MorphosML v0.3.1: HPC Ingestion, Contiguous Tensors & Calculus Core"

# Push tag to GitHub
git push origin v0.3.1
```

### 2. Merging Development Branch into Main
```bash
# Switch to main branch
git checkout main
git pull origin main

# Merge development branch
git merge 0.2.0_version -m "Merge branch '0.2.0_version': Release v0.3.1"

# Push to GitHub
git push origin main
```
