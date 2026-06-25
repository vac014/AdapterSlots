#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"

# Install adapter_slots in editable mode from the bind-mounted project directory.
# This must happen at runtime (not build time) so code changes on the host
# are immediately reflected inside the container.
if [ -f "${WORKSPACE}/setup.py" ] || [ -f "${WORKSPACE}/pyproject.toml" ]; then
    echo "[entrypoint] pip install -e ${WORKSPACE}"
    pip install --no-deps -q -e "${WORKSPACE}"
else
    echo "[entrypoint] WARNING: no setup.py/pyproject.toml at ${WORKSPACE}, skipping adapter_slots install"
fi

exec "$@"
