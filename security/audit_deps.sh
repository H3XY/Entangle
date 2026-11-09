#!/usr/bin/env bash
# Scan this project's Python dependencies for known CVEs with pip-audit.
#
# Checks every package installed in .venv (including indirect dependencies)
# against the PyPI / OSV vulnerability databases.
# Writes results/dependency_audit.md and exits non-zero if anything is found.
#
# Usage:  ./security/audit_deps.sh            # audit .venv
#         ./security/audit_deps.sh --fix      # also upgrade vulnerable packages
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[ -x "$PY" ] || { echo "No .venv found. Create it first: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 2; }
"$PY" -m pip_audit --version >/dev/null 2>&1 || "$PY" -m pip install -q pip-audit

mkdir -p results
status=0
"$PY" -m pip_audit --local --progress-spinner off --desc on --format json "$@" \
    > results/.dependency_audit.json || status=$?
"$PY" security/audit_report.py results/.dependency_audit.json results/dependency_audit.md
rm -f results/.dependency_audit.json
exit "$status"
