#!/usr/bin/env bash
# Build dist/libmojoapp-mandelzoom.so from the single Mojo compilation unit.
#
# src/capi.mojo is the entry point and src/mandel.mojo holds the kernels;
# -I src resolves the `mandel` module. Both live in one unit on purpose:
# build cost here is a flat ~2-5s regardless of how many functions it holds.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

# MODULAR_HOME must point at <prefix>/share/max or every `std` import fails
# with "unable to locate module 'std'". `pixi run` sets it; a bare `mojo` does
# not, so derive it from the resolved binary when it is missing.
if [ -z "${MODULAR_HOME:-}" ]; then
    mojo_bin="$(command -v mojo || true)"
    if [ -n "$mojo_bin" ]; then
        export MODULAR_HOME="$(dirname "$(dirname "$(readlink -f "$mojo_bin")")")/share/max"
    fi
fi

mojo build --emit shared-lib -I "$repo_dir/src" \
    "$repo_dir/src/capi.mojo" \
    -o "$repo_dir/dist/libmojoapp-mandelzoom.so"

echo "built $repo_dir/dist/libmojoapp-mandelzoom.so"
