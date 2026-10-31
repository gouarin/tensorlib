#!/usr/bin/env python3
"""Checks the one-line facade rule of ADR 0011.

A public operation is a function template constrained by `tl::operand`, declared in a
public header (under tl/ but not tl/detail/). Its body must be a single statement: it
turns its operands into expressions with `tl::as_expr` and calls a function compiled
once in a .cpp. The check reads Clang's JSON AST of the umbrella header.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile


MINIMUM_CLANG = 16  # the JSON AST names constraints through ImplicitConceptSpecializationDecl


def clang_major(compiler):
    run = subprocess.run([compiler, "-dumpversion"], capture_output=True, text=True, check=True)
    return int(run.stdout.split(".")[0])


def ast(compiler, entry, roots):
    flags = [f"-I{root}" for root in roots]
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as tu:
        tu.write(f"#include <{entry}>\n")
        tu.flush()
        run = subprocess.run([compiler, "-std=c++20", "-fsyntax-only", "-Xclang", "-ast-dump=json", *flags, tu.name],
                             capture_output=True, text=True, check=False)
    if run.returncode != 0:
        sys.exit(f"{compiler} failed on <{entry}>:\n{run.stderr}")
    return json.loads(run.stdout)


def walk(node, state, found):
    """Visits nodes in source order. Clang writes a location's file only when it changes,
    so the current file is carried along in `state`."""
    for key in ("loc", "range"):
        loc = node.get(key, {})
        for part in (loc, loc.get("begin", {}), loc.get("expansionLoc", {}), loc.get("begin", {}).get("expansionLoc", {})):
            if "file" in part:
                state["file"] = part["file"]
    if node.get("kind") == "FunctionTemplateDecl":
        found.append((node, state["file"], node.get("loc", {}).get("line")))
    if node.get("kind") == "ConceptDecl" and node.get("name") == "operand":
        state["operand"] = (state["file"], node.get("loc", {}).get("offset"))
    for child in node.get("inner", []):
        walk(child, state, found)


def constrained_by(node, concept_offset):
    """True when a constraint of `node` refers to the concept declared at `concept_offset`.
    The JSON AST does not name the concept in a constraint, only where it is declared."""
    if node.get("kind") == "ImplicitConceptSpecializationDecl" and node.get("loc", {}).get("offset") == concept_offset:
        return True
    return any(constrained_by(child, concept_offset) for child in node.get("inner", []))


def body_statements(template):
    for decl in template.get("inner", []):
        if decl.get("kind") == "FunctionDecl":
            for part in decl.get("inner", []):
                if part.get("kind") == "CompoundStmt":
                    return len(part.get("inner", []))
    return None  # declaration only


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compiler", default="clang++", help="Clang driver")
    parser.add_argument("--entry", default="tl/tl.hpp", help="header to analyse")
    parser.add_argument("include_dirs", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    roots = [d.resolve() for d in args.include_dirs]
    if clang_major(args.compiler) < MINIMUM_CLANG:
        sys.exit(f"check_facades.py needs Clang {MINIMUM_CLANG} or newer, got {args.compiler}")
    found = []
    state = {"file": "", "operand": None}
    walk(ast(args.compiler, args.entry, roots), state, found)
    if state["operand"] is None:
        return 0  # no tl::operand yet, so no public operation to check
    errors = []
    for template, file, line in found:
        path = pathlib.Path(file).resolve() if file else None
        if path is None or not any(path.is_relative_to(r / "tl") for r in roots):
            continue
        if any(path.is_relative_to(r / "tl" / "detail") for r in roots):
            continue
        if not constrained_by(template, state["operand"][1]):
            continue
        count = body_statements(template)
        if count is not None and count != 1:
            errors.append(f"{path}:{line}: public operation '{template.get('name')}' has {count} statements; "
                          "a facade is one line that calls a compiled body (ADR 0011)")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
