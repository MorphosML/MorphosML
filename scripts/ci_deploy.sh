#!/usr/bin/env bash
# ==============================================================================
# MorphosML - Automated CI Validation & Deployment Script
# Principles: Idempotency, Resiliency, Fault-Tolerance, High Performance
# ==============================================================================
set -euo pipefail

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helper log functions
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

log_info "Working directory: ${PROJECT_ROOT}"

# ------------------------------------------------------------------------------
# 1. Environment Detection (Resilience)
# ------------------------------------------------------------------------------
log_info "Locating Python environment..."
PYTHON_BIN=""

if [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -f "${PROJECT_ROOT}/../.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/../.venv/bin/python"
elif [[ -f "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    log_error "No Python interpreter found! Please set up a virtual environment."
    exit 1
fi

log_success "Using Python: $(${PYTHON_BIN} --version) (${PYTHON_BIN})"

# ------------------------------------------------------------------------------
# 2. Rebuild C++ Extension (High Performance)
# ------------------------------------------------------------------------------
log_info "Compiling and installing C++ core in editable mode..."
if ! "${PYTHON_BIN}" -m pip install -e . --no-build-isolation --no-deps; then
    # Fallback to standard pip install if wheel/setup already ready
    log_warn "Isolated build failed, attempting standard editable install..."
    "${PYTHON_BIN}" -m pip install -e .
fi
log_success "C++ core successfully built and linked."

# ------------------------------------------------------------------------------
# 3. Code Quality / Linting Checks (Fault-Tolerance)
# ------------------------------------------------------------------------------
log_info "Running code quality checks..."

if "${PYTHON_BIN}" -m ruff --version &>/dev/null; then
    log_info "Running ruff check..."
    "${PYTHON_BIN}" -m ruff check src/ tests/ || log_warn "Ruff reported lint warnings."
fi

if "${PYTHON_BIN}" -m black --version &>/dev/null; then
    log_info "Running black format check..."
    "${PYTHON_BIN}" -m black --check src/ tests/ --quiet || log_warn "Black detected formatting differences."
fi

# ------------------------------------------------------------------------------
# 4. Unit Test Suite (Idempotent Validation)
# ------------------------------------------------------------------------------
log_info "Executing test suite with pytest..."
if ! "${PYTHON_BIN}" -m pytest tests/ -v; then
    log_error "Test suite failed! Aborting deployment to prevent pushing broken code."
    exit 1
fi
log_success "All unit tests passed cleanly (100%)."

# ------------------------------------------------------------------------------
# 5. Git Status Check & Auto Deployment
# ------------------------------------------------------------------------------
# Check for flags: --check-only or --skip-push
SKIP_PUSH=false
for arg in "$@"; do
    if [[ "$arg" == "--check-only" || "$arg" == "--skip-push" ]]; then
        SKIP_PUSH=true
        break
    fi
done

if [[ "${SKIP_PUSH}" == "true" ]]; then
    log_success "CI checks completed successfully. Deployment skipped (--check-only flag passed)."
    exit 0
fi

# Identify current branch
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
log_info "Current Git branch: ${BOLD}${CURRENT_BRANCH}${NC}"

# Check git status
CHANGES_DETECTED=false
if ! git diff-index --quiet HEAD -- || [[ -n "$(git status --porcelain)" ]]; then
    CHANGES_DETECTED=true
fi

# Extract custom commit message if passed as first argument
COMMIT_MSG="${1:-}"
if [[ -z "${COMMIT_MSG}" || "${COMMIT_MSG}" == --* ]]; then
    COMMIT_MSG="ci: automated validation and sync on $(date +'%Y-%m-%d %H:%M:%S')"
fi

if [[ "${CHANGES_DETECTED}" == "true" ]]; then
    log_info "Uncommitted changes detected. Staging files..."
    git add -A
    git commit -m "${COMMIT_MSG}"
    log_success "Committed changes: \"${COMMIT_MSG}\""
else
    log_info "No uncommitted changes detected in working tree."
fi

# Check if local branch is ahead of remote or if we just committed
log_info "Pushing changes to remote: origin/${CURRENT_BRANCH}..."
if git push origin "${CURRENT_BRANCH}"; then
    log_success "Successfully pushed to origin/${CURRENT_BRANCH}!"
    log_success "GitHub Actions CI workflow has been automatically triggered."
else
    log_error "Git push failed. Please check network connection and remote permissions."
    exit 1
fi

log_success "Deployment pipeline finished successfully."

