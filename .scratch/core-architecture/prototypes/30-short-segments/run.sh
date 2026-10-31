#!/usr/bin/env bash
# PROTOTYPE - builds and runs the short-segment lowering prototype.
# Needs llvm-21-dev, libkokkos-dev (OpenMP) and libbenchmark-dev (Ubuntu 26.04 packages).
# Usage: BUILD=<build dir> ./run.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="${BUILD:?set BUILD to a build directory}"
cmake -S "$here" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm > /dev/null
cmake --build "$build" > /dev/null
export OMP_PROC_BIND=spread OMP_PLACES=threads
# Leave two vCPUs free: with every vCPU busy, any preemption stalls the OpenMP barrier.
top=$(( $(nproc) - 2 ))
OMP_NUM_THREADS=4 "$build/check" "$here"/data/*.txt 2>/dev/null | tee "$here/check.txt"
for t in 1 "$top"; do
  OMP_NUM_THREADS=$t "$build/bench" "$here"/data/*.txt \
    --benchmark_repetitions=10 --benchmark_enable_random_interleaving=true \
    --benchmark_report_aggregates_only=true --benchmark_min_time=$([ "$t" = 1 ] && echo 0.1s || echo 0.4s) \
    --benchmark_out="$here/results-${t}threads.json" --benchmark_out_format=json "$@" 2>/dev/null
done
