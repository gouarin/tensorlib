#!/usr/bin/env bash
# PROTOTYPE - builds and runs the multi-output fusion prototype.
# Usage: DEPS=<dir with install/{kokkos,benchmark} and src/{xtensor,xtl,xsimd}> ./run.sh [extra benchmark flags]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
: "${DEPS:?set DEPS to the directory holding install/kokkos, install/benchmark and src/xtensor, src/xtl, src/xsimd}"
build="$DEPS/build/proto09"
cmake -S "$here" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$DEPS/install/kokkos;$DEPS/install/benchmark" \
  -DXTENSOR_INCLUDE="$DEPS/src/xtensor/include" -DXTL_INCLUDE="$DEPS/src/xtl/include" \
  -DXSIMD_INCLUDE="$DEPS/src/xsimd/include" > /dev/null
cmake --build "$build" > /dev/null
export OMP_PROC_BIND=spread OMP_PLACES=threads
OMP_NUM_THREADS=$(( $(nproc) - 2 )) "$build/check"
# Leave two vCPUs free: with every vCPU busy, any preemption stalls the OpenMP barrier.
for t in 1 "$(( $(nproc) - 2 ))"; do
  OMP_NUM_THREADS=$t "$build/bench" \
    --benchmark_repetitions=10 --benchmark_enable_random_interleaving=true \
    --benchmark_report_aggregates_only=true --benchmark_min_time=0.2s \
    --benchmark_out="$here/results-${t}threads.json" --benchmark_out_format=json "$@"
done
