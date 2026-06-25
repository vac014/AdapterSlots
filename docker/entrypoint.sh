#!/usr/bin/env bash
set -euo pipefail

# Install the bind-mounted repository editable, so the container runs the
# checked-out code.
WORKSPACE="${WORKSPACE:-/workspace}"
if [ -f "${WORKSPACE}/pyproject.toml" ]; then
    pip install --no-deps -q -e "${WORKSPACE}"
else
    echo "entrypoint: no pyproject.toml at ${WORKSPACE}; adapter_slots not installed" >&2
fi

exec "$@"
