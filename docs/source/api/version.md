# Version

tensorlib exposes its version twice: `tl::header_version`, fixed in the headers a program includes, and `tl::library_version()`, the version of the compiled library it links against.
They differ when the headers and the library come from different installations.

```{literalinclude} ../../../examples/version.cpp
:language: cpp
```

```{doxygenstruct} tl::version
:members:
```

```{doxygenvariable} tl::header_version
```

```{doxygenfunction} tl::library_version
```
