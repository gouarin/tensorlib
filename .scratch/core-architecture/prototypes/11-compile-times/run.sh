#!/usr/bin/env bash
# PROTOTYPE - generates the expressions, measures compile times, then run times.
# Usage: DEPS=<dir with install/benchmark and src/{xtensor,xtl,xsimd}> ./run.sh [K ...]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
: "${DEPS:?set DEPS}"
gen="$DEPS/p11/gen"
ks=("${@:-1 4 16 32 64}")
python3 "$here/gen.py" "$gen" ${ks[@]} 16 > /dev/null
python3 "$here/measure.py" "$gen" "$DEPS" "$here/compile-times.json" ${ks[@]}

inc=(-I"$here" -I"$DEPS/src/xtensor/include" -I"$DEPS/src/xtl/include" -I"$DEPS/src/xsimd/include"
     -DXTENSOR_USE_XSIMD -DXTENSOR_USE_OPENMP -I"$DEPS/install/benchmark/include")
g++ -O3 -std=c++20 -fopenmp "${inc[@]}" "$here/bench.cpp" "$here/tl_lib.cpp" \
  "$gen"/{tl,et,xt_eval,xt_lazy}_16.cpp -o "$gen/bench" \
  -L"$DEPS/install/benchmark/lib" -L"$DEPS/install/benchmark/lib64" -lbenchmark -lpthread
# Leave two vCPUs free (see the multi-output fusion prototype).
OMP_NUM_THREADS=$(( $(nproc) - 2 )) OMP_PROC_BIND=spread OMP_PLACES=threads "$gen/bench" \
  --benchmark_repetitions=5 --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true --benchmark_min_time=0.5s \
  --benchmark_out="$here/run-times.json" --benchmark_out_format=json
