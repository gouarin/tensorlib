#!/usr/bin/env python3
"""Checks that code under test keeps up with the hand-written code it replaces.

A benchmark executable in benchmarks/ runs the same kernels in several variants, named
`<kernel>_<variant><<case>>`, such as `expression_fixed<vec3>` and `expression_loop<vec3>`.
benchmarks/limits.toml names, for each executable, the `candidate` variant (the library
code) and the `reference` variant (the hand-written code), and the limits between them.
Two checks compare each candidate with its reference:

- `ratios` runs the benchmarks and checks that the median time of the candidate over that
  of the reference stays at or below `max_ratio`. Both run in the same process, interleaved,
  so the speed of the machine cancels out and the check holds on shared CI runners.
  Failing cases run again, up to twice, before they count as failures;
- `codegen` disassembles the executable and checks that no candidate kernel calls a
  function the reference does not call: a call left in a kernel means that the compiler
  did not inline the library code. It does not depend on timing at all.

A case can have its own limit for a compiler, and optionally an architecture
(`[[<executable>.known]]`): it records a known gap and its reason, and the gap cannot grow.
Both checks need an optimised build.
"""

import argparse
import json
import pathlib
import platform
import re
import statistics
import subprocess
import sys
import tomllib

NAME = re.compile(r"^(?P<kernel>\w+?)_(?P<variant>[a-z]+)<(?P<case>.+)>$")
FUNCTION = re.compile(r"^[0-9a-f]+ <(?P<name>.*)>:$")
KERNEL = re.compile(r"kernel::(?P<kernel>\w+?)_(?P<variant>[a-z]+)(?P<rest><.*?>)\(")
CALLS = {"bl", "blr", "call", "callq"}
RUN = ["--benchmark_repetitions=10", "--benchmark_enable_random_interleaving=true",
       "--benchmark_min_time=0.05s", "--benchmark_format=json"]


def limits_for(path, executable, compiler, arch):
    """The table of `executable` and its known gaps for this compiler and architecture."""
    tables = tomllib.loads(pathlib.Path(path).read_text())
    if executable not in tables:
        sys.exit(f"error: {path} has no table [{executable}] for the executable {executable}")
    table = tables[executable]
    known = [k for k in table.get("known", []) if k["compiler"] == compiler and k.get("arch", arch) == arch]
    return table, known


def run(executable, names=None):
    """Median real time of each benchmark, by name."""
    args = [executable, *RUN]
    if names:
        args.append("--benchmark_filter=^(" + "|".join(re.escape(n) for n in names) + ")$")
    report = json.loads(subprocess.run(args, check=True, capture_output=True, text=True).stdout)
    return median_times(report)


def median_times(report):
    times = {}
    for result in report["benchmarks"]:
        if result.get("run_type", "iteration") == "iteration":
            times.setdefault(result["run_name"], []).append(result["real_time"])
    return {name: statistics.median(values) for name, values in times.items()}


def ratios(times, table, known):
    """(case, ratio, limit) for each candidate benchmark with a reference."""
    limit_of = {k["case"]: k["max_ratio"] for k in known if "max_ratio" in k}
    for name, time in sorted(times.items()):
        match = NAME.match(name)
        if not match or match["variant"] != table["candidate"]:
            continue
        reference = f"{match['kernel']}_{table['reference']}<{match['case']}>"
        if reference in times:
            case = f"{match['kernel']}<{match['case']}>"
            yield case, name, reference, time / times[reference], limit_of.get(case, table["max_ratio"])


def check_ratios(args, table, known):
    times = json_times(args.results) if args.results else run(args.executable)
    failing = []
    for case, candidate, reference, ratio, limit in ratios(times, table, known):
        status = "ok" if ratio <= limit else "too slow"
        print(f"{case:32} {table['candidate']}/{table['reference']} = {ratio:5.2f}  (limit {limit:.2f})  {status}")
        if ratio > limit:
            failing.append((case, candidate, reference, limit))
    for attempt in range(2 if not args.results else 0):
        if not failing:
            break
        print(f"running again: {', '.join(case for case, *_ in failing)}")
        times = run(args.executable, [n for _, c, r, _ in failing for n in (c, r)])
        failing = [(case, c, r, limit) for case, c, r, limit in failing if times[c] / times[r] > limit]
    for case, c, r, limit in failing:
        print(f"error: {case}: {table['candidate']} is more than {limit:.2f} times slower than {table['reference']}", file=sys.stderr)
    return 1 if failing else 0


def json_times(path):
    return median_times(json.loads(pathlib.Path(path).read_text()))


def calls(disassembly):
    """Called symbols of each kernel, by (kernel, variant, template arguments)."""
    result, current = {}, None
    for line in disassembly.splitlines():
        if match := FUNCTION.match(line):
            kernel = KERNEL.search(match["name"])
            current = (kernel["kernel"], kernel["variant"], kernel["rest"]) if kernel else None
            if current:
                result[current] = set()
        elif current and "\t" in line:
            # "addr:\tmnemonic operands": a tab (aarch64) or spaces (x86-64) end the mnemonic.
            instruction = line.split("\t", 1)[1].split(None, 1)
            if instruction and instruction[0] in CALLS:
                operands = instruction[1] if len(instruction) > 1 else ""
                target = re.search(r"<(.*)>", operands)
                result[current].add((target[1] if target else operands.strip()).split("(")[0])
    return result


def check_codegen(args, table, known):
    if args.disassembly:
        disassembly = pathlib.Path(args.disassembly).read_text()
    else:
        disassembly = subprocess.run([args.objdump, "-d", "-C", "--no-show-raw-insn", args.executable],
                                     check=True, capture_output=True, text=True).stdout
    allowed = [k["kernel"] for k in known if "kernel" in k]
    kernels = calls(disassembly)
    errors, checked = [], 0
    for (kernel, variant, rest), called in sorted(kernels.items()):
        reference = kernels.get((kernel, table["reference"], rest))
        if variant != table["candidate"] or reference is None:
            continue
        checked += 1
        extra = called - reference
        if extra and not any(a in rest for a in allowed):
            errors.append(f"kernel::{kernel}_{variant}{rest} calls {', '.join(sorted(extra))}, "
                          f"which kernel::{kernel}_{table['reference']} does not call")
    print(f"{checked} {table['candidate']} kernels compared with their {table['reference']} version")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors or not checked else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=["ratios", "codegen"])
    parser.add_argument("executable", help="benchmark executable; its name selects the table of the limits")
    parser.add_argument("--limits", required=True, help="benchmarks/limits.toml")
    parser.add_argument("--compiler", required=True, help="CMAKE_CXX_COMPILER_ID: GNU, Clang...")
    parser.add_argument("--arch", default=platform.machine().replace("arm64", "aarch64"),
                        help="architecture the known gaps are selected for (default: this machine's)")
    parser.add_argument("--objdump", default="objdump")
    parser.add_argument("--results", help="read Google Benchmark JSON instead of running (self-tests)")
    parser.add_argument("--disassembly", help="read objdump output instead of disassembling (self-tests)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep the report in order with the errors
    table, known = limits_for(args.limits, pathlib.Path(args.executable).stem, args.compiler, args.arch)
    check = check_ratios if args.check == "ratios" else check_codegen
    return check(args, table, known)


if __name__ == "__main__":
    sys.exit(main())
