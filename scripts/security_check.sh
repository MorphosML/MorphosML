#!/usr/bin/env bash
# ==============================================================================
# MorphosML Master Security Check Script
# ==============================================================================
# Automates multi-layer security auditing:
# - Secret / credential leak scanning
# - C++ memory safety inspection
# - Python dependency vulnerability scan (pip-audit)
# - AST Python security linter (bandit)
# - File permissions and .gitignore compliance
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$ROOT_DIR")"

# Locate active or workspace Python virtual environment
if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    VENV_PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
    VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
elif [ -f "${PARENT_DIR}/.venv/bin/python" ]; then
    VENV_PYTHON="${PARENT_DIR}/.venv/bin/python"
else
    VENV_PYTHON="$(which python3)"
fi

echo "======================================================"
echo "    Running MorphosML Security Audit & Vulnerability Scan"
echo "    Python Interpreter: $VENV_PYTHON"
echo "======================================================"

"$VENV_PYTHON" "${SCRIPT_DIR}/security_audit.py" "$@"
