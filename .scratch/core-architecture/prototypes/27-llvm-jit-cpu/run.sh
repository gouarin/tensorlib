#!/usr/bin/env bash
# PROTOTYPE - builds and runs the LLVM JIT prototype.
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
for t in 1 4 "$top"; do OMP_NUM_THREADS=$t "$build/check"; done 2>/dev/null | tee "$here/check.txt"
OMP_NUM_THREADS=$top "$build/compile_times" 2>/dev/null | tee "$here/compile-times.txt"
for t in 1 "$top"; do
  OMP_NUM_THREADS=$t "$build/bench" \
    --benchmark_repetitions=10 --benchmark_enable_random_interleaving=true \
    --benchmark_report_aggregates_only=true --benchmark_min_time=0.2s \
    --benchmark_out="$here/results-${t}threads.json" --benchmark_out_format=json "$@" 2>/dev/null
  OMP_NUM_THREADS=$t "$build/ceilings" \
    --benchmark_repetitions=5 --benchmark_enable_random_interleaving=true \
    --benchmark_report_aggregates_only=true --benchmark_min_time=0.2s \
    --benchmark_out="$here/ceilings-${t}threads.json" --benchmark_out_format=json 2>/dev/null
done
