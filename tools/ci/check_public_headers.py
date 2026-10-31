#!/usr/bin/env python3
"""Checks what tensorlib's public headers include.

Two rules from the style rules of the spec (sections 14 and 15):

- a public header includes only tensorlib headers and the standard headers of the
  whitelist (component headers may also include their own dependency);
- public headers have no conditional path on the language version (`__cplusplus`)
  or on library feature-test macros (`__cpp_*`, `__has_include`).

The include tree comes from the compiler (`-H`), so the report names the whole
chain from the header a user includes down to the offending one.
"""

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

WHITELIST = {
    "cstddef", "cstdint", "type_traits", "concepts", "utility", "initializer_list",
    "span", "array", "compare", "limits", "source_location", "cmath",
}

# Header a user includes -> prefixes of the third-party headers it may include.
ENTRY_POINTS = {
    "tl/tl.hpp": (),
}

CONDITIONAL = re.compile(r"^\s*#\s*(if|elif|ifdef|ifndef)\b.*(__cplusplus|__cpp_|__has_include)")


def is_tensorlib(path, roots):
    return any(path.is_relative_to(root / "tl") for root in roots)


def include_tree(compiler, entry, roots):
    """Yields (chain, path) for each header included while compiling `entry`."""
    flags = [f"-I{root}" for root in roots]
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as tu:
        tu.write(f"#include <{entry}>\n")
        tu.flush()
        run = subprocess.run([compiler, "-std=c++20", "-fsyntax-only", "-H", *flags, tu.name],
                             capture_output=True, text=True, check=False)
    if run.returncode != 0:
        sys.exit(f"{compiler} failed on <{entry}>:\n{run.stderr}")
    stack = []
    for line in run.stderr.splitlines():
        match = re.match(r"^(\.+) (.+)$", line)
        if not match:
            continue
        depth = len(match.group(1))
        path = pathlib.Path(match.group(2)).resolve()
        del stack[depth - 1:]
        stack.append(path)
        yield list(stack[:-1]), path


def header_name(path, roots):
    for root in roots:
        if path.is_relative_to(root):
            return path.relative_to(root).as_posix()
    return path.name


def check_includes(compiler, roots):
    errors = []
    for entry, allowed in ENTRY_POINTS.items():
        for chain, path in include_tree(compiler, entry, roots):
            parent = chain[-1] if chain else None
            if parent is not None and not is_tensorlib(parent, roots):
                continue  # inside a standard or third-party header: not ours to judge
            if is_tensorlib(path, roots) or path.name in WHITELIST:
                continue
            if any(str(path).find(prefix) >= 0 for prefix in allowed):
                continue
            names = [f"<{entry}>"] + [f"<{header_name(p, roots)}>" for p in chain[1:]] + [f"<{header_name(path, roots)}>"]
            errors.append(f"<{header_name(path, roots)}> is not on the public header whitelist, "
                          f"included through {' -> '.join(names)}")
    return errors


def check_conditionals(roots):
    errors = []
    for root in roots:
        for header in sorted((root / "tl").rglob("*.hpp")):
            for number, line in enumerate(header.read_text().splitlines(), start=1):
                if CONDITIONAL.match(line):
                    errors.append(f"{header}:{number}: conditional path on the language or library version: {line.strip()}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compiler", default="c++", help="GCC or Clang driver used to list includes")
    parser.add_argument("include_dirs", nargs="+", type=pathlib.Path,
                        help="directories that hold tl/ (the source tree and the build tree)")
    args = parser.parse_args()
    roots = [d.resolve() for d in args.include_dirs]
    errors = check_includes(args.compiler, roots) + check_conditionals(roots)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
