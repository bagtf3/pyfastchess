cmake_minimum_required(VERSION 3.18)
project(pyfastchess LANGUAGES CXX)

# Find Python & NumPy
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NumPy_INCLUDE_DIR
)

# Find pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found via CONFIG; trying module mode")
  find_package(pybind11 REQUIRED)
endif()

add_library(pyfastchess MODULE
  src/binding.cpp
  src/backend.cpp
)

target_include_directories(pyfastchess PRIVATE
    ${NumPy_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/chess-library/include
)

target_link_libraries(pyfastchess PRIVATE pybind11::module)
target_compile_features(pyfastchess PRIVATE cxx_std_17)

# Optimize
if (MSVC)
  target_compile_options(pyfastchess PRIVATE /O2 /DNOMINMAX)
else()
  target_compile_options(pyfastchess PRIVATE -O3 -march=native)
endif()

# Proper Python extension name
pybind11_extension(pyfastchess)
pybind11_strip(pyfastchess)
set_target_properties(pyfastchess PROPERTIES OUTPUT_NAME "pyfastchess")
