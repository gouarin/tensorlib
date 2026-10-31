# A project that uses an installed tensorlib, as a user would.
cmake_minimum_required(VERSION 3.25.2)
project(tensorlib_consumer LANGUAGES CXX)

find_package(tensorlib REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE tensorlib::core)
