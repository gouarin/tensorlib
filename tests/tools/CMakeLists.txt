# Self-tests of the CI checks in tools/ci, against the real headers and against fixtures
# that break each rule.
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_FOUND OR MSVC)
    return()
endif()

set(tools "${PROJECT_SOURCE_DIR}/tools/ci")
set(fixtures "${tools}/tests")
set(include_dirs "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include")

add_test(NAME check_headers.public_headers
    COMMAND Python3::Interpreter "${tools}/check_public_headers.py" --compiler "${CMAKE_CXX_COMPILER}" ${include_dirs})
add_test(NAME check_headers.rejects_banned_header
    COMMAND Python3::Interpreter "${tools}/check_public_headers.py" --compiler "${CMAKE_CXX_COMPILER}" "${fixtures}/banned_header")
set_tests_properties(check_headers.rejects_banned_header PROPERTIES
    PASS_REGULAR_EXPRESSION "<algorithm> is not on the public header whitelist, included through <tl/tl.hpp> -> <tl/detail/sort.hpp> -> <algorithm>")
add_test(NAME check_headers.rejects_version_conditional
    COMMAND Python3::Interpreter "${tools}/check_public_headers.py" --compiler "${CMAKE_CXX_COMPILER}" "${fixtures}/version_conditional")
set_tests_properties(check_headers.rejects_version_conditional PROPERTIES
    PASS_REGULAR_EXPRESSION "conditional path on the language or library version: #if __cplusplus")
set_tests_properties(check_headers.public_headers check_headers.rejects_banned_header check_headers.rejects_version_conditional
    PROPERTIES LABELS unit)

add_test(NAME check_benchmarks.rejects_slow_candidate
    COMMAND Python3::Interpreter "${tools}/check_benchmarks.py" ratios bench_demo --compiler GNU
            --limits "${fixtures}/benchmarks/limits.toml" --results "${fixtures}/benchmarks/bench_demo.json")
set_tests_properties(check_benchmarks.rejects_slow_candidate PROPERTIES
    PASS_REGULAR_EXPRESSION "error: expression<small>: fixed is more than 1.15 times slower than loop"
    FAIL_REGULAR_EXPRESSION "error: compound<big>")
add_test(NAME check_benchmarks.rejects_extra_call
    COMMAND Python3::Interpreter "${tools}/check_benchmarks.py" codegen bench_demo --compiler GNU
            --limits "${fixtures}/benchmarks/limits.toml" --disassembly "${fixtures}/benchmarks/bench_demo.s")
set_tests_properties(check_benchmarks.rejects_extra_call PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel::expression_fixed<tl::fixed<double, 32ul> > calls tl::fixed<double, 32ul> tl::detail::elementwise")
set_tests_properties(check_benchmarks.rejects_slow_candidate check_benchmarks.rejects_extra_call PROPERTIES LABELS unit)

# The facade check reads Clang's AST and needs Clang 16 or newer.
find_program(TL_CLANGXX NAMES clang++-21 clang++-20 clang++-19 clang++-18 clang++)
if(TL_CLANGXX)
    execute_process(COMMAND "${TL_CLANGXX}" -dumpversion OUTPUT_VARIABLE tl_clang_version OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
if(TL_CLANGXX AND tl_clang_version VERSION_GREATER_EQUAL 16)
    add_test(NAME check_facades.public_headers
        COMMAND Python3::Interpreter "${tools}/check_facades.py" --compiler "${TL_CLANGXX}" ${include_dirs})
    add_test(NAME check_facades.rejects_two_statements
        COMMAND Python3::Interpreter "${tools}/check_facades.py" --compiler "${TL_CLANGXX}" "${fixtures}/facades")
    set_tests_properties(check_facades.rejects_two_statements PROPERTIES
        PASS_REGULAR_EXPRESSION "public operation 'twice' has 2 statements")
    set_tests_properties(check_facades.public_headers check_facades.rejects_two_statements PROPERTIES LABELS unit)
endif()
