"""
MorphosML Automated Security & Vulnerability Scanner
=====================================================
Performs automated multi-layer security auditing:
1. Secret Leak Detection (PyPI tokens, GitHub PATs, private keys, passwords)
2. Python Dependency Vulnerability Audit (pip-audit / PyPA CVE database)
3. AST Python Static Security Analysis (bandit)
4. C++ Memory Safety & Deprecated Functions Inspection
5. Sensitive File & .gitignore Compliance Checks
"""

import re
import subprocess
import sys
from pathlib import Path

# ANSI Color Codes
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"


def print_header(title: str):
    print(f"\n{BOLD}{CYAN}=== [ {title} ] ==={NC}")


def check_secrets_and_credentials(root_dir: Path) -> list[str]:
    """Scan codebase for leaked secrets, API tokens, and private keys."""
    print_header("1. Secret & Credential Leak Detection")
    findings = []

    secret_patterns = [
        (r"pypi-AgEI[A-Za-z0-9-_]{50,}", "PyPI API Token"),
        (r"ghp_[A-Za-z0-9]{36,}", "GitHub Personal Access Token"),
        (r"github_pat_[A-Za-z0-9_]{50,}", "GitHub Fine-Grained PAT"),
        (
            r"-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----",
            "Private Cryptographic Key",
        ),
        (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
        (
            r"(?i)(password|secret|api_key)\s*=\s*['\"][A-Za-z0-9@#$%^&+=_-]{8,}['\"]",
            "Hardcoded Password/Secret",
        ),
    ]

    excluded_dirs = {".git", ".venv", "build", "dist", ".pytest_cache", "__pycache__"}
    excluded_files = {"security_audit.py", "security_check.sh"}

    scanned_files = 0
    for file_path in root_dir.rglob("*"):
        if file_path.is_file() and not any(
            part in excluded_dirs for part in file_path.parts
        ):
            if file_path.name in excluded_files or file_path.suffix in {
                ".png",
                ".so",
                ".whl",
                ".tar.gz",
                ".mldat",
            }:
                continue
            scanned_files += 1
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                for pattern, desc in secret_patterns:
                    matches = re.finditer(pattern, content)
                    for m in matches:
                        line_num = content[: m.start()].count("\n") + 1
                        findings.append(
                            f"{desc} in {file_path.relative_to(root_dir)}:L{line_num}"
                        )
            except Exception as e:
                findings.append(f"Could not read {file_path}: {e}")

    if not findings:
        print(
            f"{GREEN}[PASS] Scanned {scanned_files} files. No secrets or tokens detected.{NC}"
        )
    else:
        print(f"{RED}[FAIL] Potential secrets detected:{NC}")
        for f in findings:
            print(f"  {RED}! {f}{NC}")
    return findings


def check_cpp_memory_safety(root_dir: Path) -> list[str]:
    """Inspect C++ source and header files for unsafe legacy functions."""
    print_header("2. C++ Memory Safety & Deprecated Functions Inspection")
    findings = []
    cpp_dir = root_dir / "cpp"
    if not cpp_dir.exists():
        print(f"{YELLOW}[SKIP] cpp/ directory not found.{NC}")
        return findings

    unsafe_functions = [
        (
            r"\bstrcpy\s*\(",
            "Unbounded buffer copy 'strcpy' (use strncpy or std::string)",
        ),
        (
            r"\bstrcat\s*\(",
            "Unbounded string concatenation 'strcat' (use strncat or std::string)",
        ),
        (r"\bsprintf\s*\(", "Unbounded format 'sprintf' (use snprintf)"),
        (r"\bgets\s*\(", "Dangerous function 'gets' (use fgets)"),
    ]

    cpp_files = list(cpp_dir.rglob("*.cpp")) + list(cpp_dir.rglob("*.hpp"))
    for file_path in cpp_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for pattern, desc in unsafe_functions:
                matches = re.finditer(pattern, content)
                for m in matches:
                    line_num = content[: m.start()].count("\n") + 1
                    findings.append(
                        f"{desc} in {file_path.relative_to(root_dir)}:L{line_num}"
                    )
        except Exception as e:
            findings.append(f"Error reading {file_path}: {e}")

    if not findings:
        print(
            f"{GREEN}[PASS] Scanned {len(cpp_files)} C++ source/header files. No unsafe C string functions.{NC}"
        )
    else:
        print(f"{RED}[FAIL] Insecure C++ functions found:{NC}")
        for f in findings:
            print(f"  {RED}! {f}{NC}")
    return findings


def check_dependency_vulnerabilities() -> list[str]:
    """Run pip-audit against installed packages and CVE databases."""
    print_header("3. Python Dependency Vulnerability Audit (pip-audit)")
    findings = []
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip_audit"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0:
            print(
                f"{GREEN}[PASS] pip-audit passed: No known vulnerabilities found.{NC}"
            )
        else:
            findings.append("Known dependency vulnerabilities identified:")
            findings.extend(proc.stdout.strip().splitlines())
            print(f"{RED}[FAIL] {proc.stdout.strip()}{NC}")
    except FileNotFoundError:
        print(
            f"{YELLOW}[WARN] pip-audit is not installed. Install with `pip install pip-audit`.{NC}"
        )
    except subprocess.TimeoutExpired:
        print(
            f"{YELLOW}[WARN] pip-audit timed out contacting vulnerability advisory servers.{NC}"
        )
    except Exception as e:
        print(f"{YELLOW}[WARN] pip-audit check failed to run: {e}{NC}")
    return findings


def check_bandit_static_analysis(root_dir: Path) -> list[str]:
    """Run bandit AST security linter on Python source code."""
    print_header("4. Python AST Security Analysis (bandit)")
    findings = []
    src_dir = root_dir / "src"
    if not src_dir.exists():
        return findings

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "bandit", "-r", str(src_dir), "-q"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            print(
                f"{GREEN}[PASS] bandit passed: 0 security issues identified in src/.{NC}"
            )
        else:
            findings.append("Bandit identified potential security risks:")
            findings.extend(proc.stdout.strip().splitlines())
            print(f"{RED}[FAIL] Bandit issues found:{NC}\n{proc.stdout.strip()}")
    except FileNotFoundError:
        print(
            f"{YELLOW}[WARN] bandit is not installed. Install with `pip install bandit`.{NC}"
        )
    except Exception as e:
        print(f"{YELLOW}[WARN] bandit execution error: {e}{NC}")
    return findings


def check_gitignore_hygiene(root_dir: Path) -> list[str]:
    """Ensure sensitive patterns and temporary build artifacts are ignored."""
    print_header("5. Sensitive File & .gitignore Compliance")
    findings = []
    gitignore_path = root_dir / ".gitignore"
    if not gitignore_path.exists():
        findings.append("Missing .gitignore file!")
        print(f"{RED}[FAIL] .gitignore is missing!{NC}")
        return findings

    content = gitignore_path.read_text(encoding="utf-8", errors="ignore")
    required_patterns = [
        "*.env",
        "*.pem",
        "*.key",
        "compile_commands.json",
        "dist/",
        "build/",
        "*.swp",
    ]

    missing = [pat for pat in required_patterns if pat not in content]
    if not missing:
        print(
            f"{GREEN}[PASS] .gitignore includes all essential sensitive and build artifact patterns.{NC}"
        )
    else:
        for m in missing:
            findings.append(f"Missing recommended ignore pattern: {m}")
            print(f"{YELLOW}[WARN] Missing in .gitignore: {m}{NC}")
    return findings


def main():
    print(f"{BOLD}======================================================{NC}")
    print(f"{BOLD}        MORPHOSML SECURITY & VULNERABILITY AUDIT       {NC}")
    print(f"{BOLD}======================================================{NC}")

    root_dir = Path(__file__).resolve().parent.parent

    secrets = check_secrets_and_credentials(root_dir)
    cpp_safety = check_cpp_memory_safety(root_dir)
    dep_vulns = check_dependency_vulnerabilities()
    bandit_issues = check_bandit_static_analysis(root_dir)
    gitignore_issues = check_gitignore_hygiene(root_dir)

    total_issues = len(secrets) + len(cpp_safety) + len(dep_vulns) + len(bandit_issues)

    print("\n" + "=" * 54)
    if total_issues == 0:
        print(
            f"{GREEN}{BOLD}SECURITY AUDIT PASSED: 0 vulnerabilities or leaks found!{NC}"
        )
        sys.exit(0)
    else:
        print(f"{RED}{BOLD}SECURITY AUDIT FAILED: {total_issues} issues detected!{NC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
