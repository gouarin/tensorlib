# Documentation examples: every public function has one. Each example is built and run
# as a test (label `example`) on the serial backend.
file(GLOB examples CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(source IN LISTS examples)
    get_filename_component(name "${source}" NAME_WE)
    add_executable(example_${name} "${source}")
    target_link_libraries(example_${name} PRIVATE tensorlib::core)
    tl_set_warnings(example_${name})
    add_test(NAME example.${name} COMMAND example_${name})
    set_tests_properties(example.${name} PROPERTIES LABELS example)
endforeach()
