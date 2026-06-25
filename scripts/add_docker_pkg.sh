#!/usr/bin/env bash
# Usage: ./scripts/add_docker_pkg.sh <package-name>
# Installs the package into adapter_env, finds its version, and inserts
# the pinned line into docker/requirements_docker.txt in alphabetical order.
set -euo pipefail

PKG="${1:?Usage: $0 <package-name>}"
REQS="$(dirname "$0")/../docker/requirements_docker.txt"

# Resolve to absolute path so we can cd freely
REQS="$(realpath "$REQS")"

echo "==> Installing ${PKG} into adapter_env …"
conda run -n adapter_env pip install --quiet "$PKG"

VERSION="$(conda run -n adapter_env pip show "$PKG" 2>/dev/null | awk '/^Version:/{print $2}')"
if [[ -z "$VERSION" ]]; then
    echo "ERROR: could not determine version for '${PKG}'" >&2
    exit 1
fi

PINNED="${PKG}==${VERSION}"
echo "==> Resolved: ${PINNED}"

# Check if the exact line already exists
if grep -qxF "$PINNED" "$REQS"; then
    echo "Already present in requirements_docker.txt -- nothing to do."
    exit 0
fi

# Check if a different version is already pinned
if grep -qi "^${PKG}==" "$REQS"; then
    EXISTING="$(grep -i "^${PKG}==" "$REQS")"
    echo "WARNING: replacing existing line '${EXISTING}' with '${PINNED}'"
    # Remove the old line first (case-insensitive match on the package name)
    grep -iv "^${PKG}==" "$REQS" > "${REQS}.tmp" && mv "${REQS}.tmp" "$REQS"
fi

# Insert in alphabetical order (skip comment/blank lines when sorting)
# Strategy: append, then re-sort comment header + sorted body
HEADER="$(grep -E '^(#|[[:space:]]*$)' "$REQS")"
BODY="$(grep -vE '^(#|[[:space:]]*$)' "$REQS")"

{
    echo "$HEADER"
    printf '%s\n%s\n' "$BODY" "$PINNED" | sort --ignore-case
} > "${REQS}.tmp"

mv "${REQS}.tmp" "$REQS"

echo "==> Added to docker/requirements_docker.txt:"
grep -n "^${PKG}==" "$REQS"
