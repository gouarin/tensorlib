"""Sphinx configuration of the tensorlib documentation."""

import pathlib
import re
import subprocess
import warnings

docs = pathlib.Path(__file__).resolve().parent.parent

# The API pages come from the Doxygen XML of the public headers, regenerated at each build.
# Doxygen creates only the last level of its output directory.
output = docs.parent / "build" / "doxygen"
output.mkdir(parents=True, exist_ok=True)
subprocess.run(["doxygen", "Doxyfile"], cwd=docs, check=True)
xml = output / "xml"

# Doxygen writes `constexpr` as the return type of a constexpr constructor or destructor,
# and also marks it constexpr="yes"; Breathe then prints `constexpr constexpr`, which the
# Sphinx C++ domain rejects. The attribute is enough.
for page in xml.glob("*.xml"):
    text = page.read_text()
    if "<type>constexpr</type>" in text:
        page.write_text(text.replace("<type>constexpr</type>", "<type></type>"))

# Breathe lists private base classes, which are implementation details living in
# tl::detail (the storage of `fixed`); the C++ domain cannot resolve them either.
private_base = re.compile(r'\s*<basecompoundref[^>]*prot="private"[^>]*>.*?</basecompoundref>')
for page in xml.glob("*.xml"):
    text = page.read_text()
    if 'prot="private"' in text:
        page.write_text(private_base.sub("", text))

project = "tensorlib"
copyright = "tensorlib developers"
language = "en"

extensions = [
    "breathe",
    "myst_parser",
    "sphinx.ext.mathjax",
]

breathe_projects = {"tensorlib": str(xml)}
# Breathe 5.0 does not know the <conceptparts> element of Doxygen 1.18 and warns once per
# concept; the concepts render correctly without it.
warnings.filterwarnings("ignore", message=r".*unexpected element \"conceptparts\"")
breathe_default_project = "tensorlib"
breathe_default_members = ("members",)
# Users include <tl/tl.hpp> only, as the API index says; per-header include lines would
# point to the internal layout.
breathe_show_include = False

# The C++ domain cannot resolve the namespace `tl` itself in signatures.
nitpick_ignore = [("cpp:identifier", "tl")]

# Every page is written in MyST Markdown.
source_suffix = {".md": "markdown"}
myst_enable_extensions = ["colon_fence"]

# Code blocks and signatures are C++ unless they say otherwise.
primary_domain = "cpp"
highlight_language = "cpp"

html_theme = "furo"
html_title = project
