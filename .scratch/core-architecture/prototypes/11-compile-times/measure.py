"""PROTOTYPE - throwaway. Compile-time measurement of the generated TUs.

Each TU is compiled REPS times, one compile at a time, and the median wall time,
the peak memory of the compiler and the object size are reported. The variants
are interleaved inside each repetition so they see the same machine state.
Usage: python3 measure.py GENDIR DEPS OUT.json K [K ...]
"""
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPS = 3
TIMEOUT = 900
VARIANTS = ["tl", "et", "xt_eval", "xt_lazy"]


def compile_once(cmd):
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        _, status, ru = os.wait4(p.pid, 0)
    except KeyboardInterrupt:
        p.kill()
        raise
    wall = time.perf_counter() - t0
    err = p.stderr.read().decode()
    if status != 0:
        raise RuntimeError(err[-2000:])
    return wall, ru.ru_maxrss / 1024  # MB


def main():
    gen, deps, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    ks = list(map(int, sys.argv[4:]))
    here = Path(__file__).resolve().parent
    flags = ["g++", "-O3", "-std=c++20", "-fopenmp", f"-I{here}",
             f"-I{deps}/src/xtensor/include", f"-I{deps}/src/xtl/include", f"-I{deps}/src/xsimd/include",
             "-DXTENSOR_USE_XSIMD", "-DXTENSOR_USE_OPENMP", "-c"]
    results = []
    for k in ks:
        samples = {v: [] for v in VARIANTS}
        for rep in range(REPS):
            for i in range(len(VARIANTS)):
                v = VARIANTS[(i + rep) % len(VARIANTS)]
                src, obj = gen / f"{v}_{k}.cpp", gen / f"{v}_{k}.o"
                samples[v].append(compile_once(flags + [str(src), "-o", str(obj)]))
        for v in VARIANTS:
            walls = [s[0] for s in samples[v]]
            row = dict(variant=v, K=k, wall_s=statistics.median(walls), wall_min=min(walls), wall_max=max(walls),
                       rss_mb=max(s[1] for s in samples[v]), obj_kb=(gen / f"{v}_{k}.o").stat().st_size / 1024)
            results.append(row)
            print(f"K={k:3d} {v:8s} {row['wall_s']:8.2f} s  ({row['wall_min']:.2f}-{row['wall_max']:.2f})"
                  f"  {row['rss_mb']:7.0f} MB  obj {row['obj_kb']:8.0f} KB", flush=True)
        out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
