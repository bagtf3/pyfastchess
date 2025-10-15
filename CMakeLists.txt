cmake_minimum_required(VERSION 3.18)
project(pyfastchess LANGUAGES CXX)

# --- Use modern FindPython for pybind11 ---
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(pyfastchess._core
    src/backend.cpp
    src/binding.cpp
    src/mcts.cpp
    src/evaluator.cpp
    src/cache.cpp
)

# --- Include directories (explicit) ---
target_include_directories(pyfastchess._core PRIVATE
    "${CMAKE_SOURCE_DIR}/extern/chess-library/include"
    "${CMAKE_SOURCE_DIR}/src"
)

# --- Compiler settings ---
target_compile_features(pyfastchess._core PRIVATE cxx_std_17)
if (MSVC)
  target_compile_options(pyfastchess._core PRIVATE /permissive- /Zc:__cplusplus)
endif()

# Keep the module name stable so "from pyfastchess import _core" works.
set_target_properties(pyfastchess._core PROPERTIES OUTPUT_NAME "_core")

# --- Optional ONNX Runtime linkage (ONLY if env var is set) ---
# Exported behavior: if you set ONNXRUNTIME_ROOT in your environment it will be used.
if(NOT DEFINED ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()

if(ONNXRUNTIME_ROOT)
  # Use the provided ONNXRUNTIME_ROOT for headers/libs; do NOT copy DLLs.
  target_include_directories(pyfastchess._core PRIVATE "${ONNXRUNTIME_ROOT}/include")
  target_link_directories(pyfastchess._core PRIVATE "${ONNXRUNTIME_ROOT}/lib")
  target_link_libraries(pyfastchess._core PRIVATE onnxruntime onnxruntime_providers_cuda)
  message(STATUS "Configuring pyfastchess._core to link against ONNX Runtime at: ${ONNXRUNTIME_ROOT}")
else()
  message(STATUS "ONNXRUNTIME_ROOT not defined; building without ONNX Runtime linkage.")
endif()


