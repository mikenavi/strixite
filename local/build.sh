#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
python3 "$repo/local/verify-source.py"
[[ "${1:-}" == --verify-only ]] && exit 0
: "${STRIX_ROCM_ROOT:?Set the external pinned ROCm SDK root}"
: "${STRIX_HIP_LIB:?Set the exact HIP runtime shared-library path}"
: "${STRIX_BUILD_DIR:?Set a separate build directory}"
cmake -S "$repo" -B "$STRIX_BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-include stddef.h" -DSTRIX_ROCM_ROOT="$STRIX_ROCM_ROOT" -DSTRIX_HIP_LIB="$STRIX_HIP_LIB" -DSTRIX_GPU_ARCH="${GPU_ARCH:-gfx1151}"
cmake --build "$STRIX_BUILD_DIR" --target strix_server strix_api_tests -j "${JOBS:-1}"
if [[ -n "${STRIX_TOKENIZER:-}" ]]; then
 LD_LIBRARY_PATH="$(dirname "$STRIX_HIP_LIB"):${LD_LIBRARY_PATH:-}" "$STRIX_BUILD_DIR/strix_api_tests" "$STRIX_TOKENIZER"
fi
