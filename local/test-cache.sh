#!/usr/bin/env bash
# CPU-only admission for b175d27 + native-api-v2.patch + turn-cache.patch.
set -euo pipefail
if [[ $# != 1 ]]; then echo "Usage: $0 NEW_BUILD_OUTPUT_DIR" >&2; exit 2; fi
source_dir=$(cd "$(dirname "$0")/.." && pwd)
output=$(realpath -m "$1")
tests="$source_dir/tests/fleet"
mkdir -p "$output/build" "$output/tmp" "$output/logs"
export TMPDIR="$output/tmp"
compiler=${CXX:-g++}
flags=(-std=c++20 -O0 -ffunction-sections -fdata-sections -pthread -include cstddef -I"$source_dir")
for test in turn_cache_regression turn_cache_tests cache_memory_pressure; do
    nice -n 10 "$compiler" "${flags[@]}" "$tests/$test.cpp" \
        "$source_dir/serve/prompt_cache.cpp" "$source_dir/serve/host_buffer.cpp" \
        "$source_dir/formats/strixw.cpp" -Wl,--gc-sections -o "$output/build/$test"
    "$output/build/$test" "$output/tmp/$test" > "$output/logs/$test.log" 2>&1
    cat "$output/logs/$test.log"
done
"$output/build/cache_memory_pressure" "$output/tmp/memory-budget" blocked > "$output/logs/memory-budget.log" 2>&1
cat "$output/logs/memory-budget.log"
nice -n 10 "$compiler" "${flags[@]}" -I"$source_dir/third_party/cpp-httplib" \
    "$tests/turn_cache_config_tests.cpp" "$source_dir/serve/server_config.cpp" \
    -Wl,--gc-sections -o "$output/build/turn_cache_config_tests"
"$output/build/turn_cache_config_tests" "$output/tmp/config.conf" > "$output/logs/config.log" 2>&1
cat "$output/logs/config.log"
