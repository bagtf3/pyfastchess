cmake_minimum_required(VERSION 3.18)
project(pyfastchess LANGUAGES CXX)

# Tell pybind11 to use modern FindPython
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED CONFIG)

# Build the python extension module (explicit, no variables hiding sources)
pybind11_add_module(pyfastchess._core
    src/backend.cpp
    src/binding.cpp
    src/mcts.cpp
    src/evaluator.cpp
    src/cache.cpp
)

# Explicit include dirs (same style as your working file)
target_include_directories(pyfastchess._core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/chess-library/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(pyfastchess._core PRIVATE cxx_std_17)
if (MSVC)
  target_compile_options(pyfastchess._core PRIVATE /permissive- /Zc:__cplusplus)
endif()

# Keep the module name stable so "import pyfastchess._core" works
set_target_properties(pyfastchess._core PROPERTIES OUTPUT_NAME "_core")

# Install the extension into the pyfastchess package directory (same as your original)
if (WIN32)
  install(TARGETS pyfastchess._core
    RUNTIME DESTINATION pyfastchess
    LIBRARY DESTINATION pyfastchess
    ARCHIVE DESTINATION pyfastchess)
else()
  install(TARGETS pyfastchess._core
    LIBRARY DESTINATION pyfastchess)
endif()
