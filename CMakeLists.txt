cmake_minimum_required(VERSION 3.18)
project(pyfastchess LANGUAGES CXX)

# ---- Python & pybind11 ----
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED CONFIG)

# ---- Sources ----
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(SOURCES
    ${SRC_DIR}/backend.cpp
    ${SRC_DIR}/binding.cpp
    ${SRC_DIR}/mcts.cpp
    ${SRC_DIR}/evaluator.cpp
    ${SRC_DIR}/cache.cpp
)

# ---- Extension module ----
pybind11_add_module(pyfastchess._core ${SOURCES})

target_include_directories(pyfastchess._core PRIVATE
    "${PROJECT_SOURCE_DIR}/extern/chess-library/include"
    "${SRC_DIR}"
)

target_compile_features(pyfastchess._core PRIVATE cxx_std_17)
if (MSVC)
  target_compile_options(pyfastchess._core PRIVATE /permissive- /Zc:__cplusplus)
endif()

# Name must be "_core" for "from pyfastchess import _core"
set_target_properties(pyfastchess._core PROPERTIES OUTPUT_NAME "_core")

# ---- ONNX Runtime (optional GPU) ----
if(NOT DEFINED ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()
# Set a default only if you actually use ORT; otherwise comment this block out.
if(NOT ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "C:/libs/onnxruntime-win-x64-gpu-1.19.0")
endif()

target_include_directories(pyfastchess._core PRIVATE "${ONNXRUNTIME_ROOT}/include")
target_link_directories(pyfastchess._core PRIVATE "${ONNXRUNTIME_ROOT}/lib")
target_link_libraries(pyfastchess._core PRIVATE onnxruntime onnxruntime_providers_cuda)

# ====== INSTALL RULES (no outputs into src/) ======

# Where to place the Python package inside the install prefix
set(PY_PKG_DIR "pyfastchess")

# 1) Install the compiled extension into <prefix>/pyfastchess/_core*.pyd
if (WIN32)
  install(TARGETS pyfastchess._core
    RUNTIME DESTINATION ${PY_PKG_DIR}
    LIBRARY DESTINATION ${PY_PKG_DIR}
    ARCHIVE DESTINATION ${PY_PKG_DIR})
else()
  install(TARGETS pyfastchess._core
    LIBRARY DESTINATION ${PY_PKG_DIR})
endif()

# 2) Install your pure-Python files (only .py) into <prefix>/pyfastchess
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/pyfastchess/"
        DESTINATION ${PY_PKG_DIR}
        FILES_MATCHING PATTERN "*.py")

# 3) Install ORT runtime DLLs next to the .pyd (install-time only; not in source tree)
if(WIN32)
  install(FILES
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll"
    DESTINATION ${PY_PKG_DIR}
    OPTIONAL)
endif()

message(STATUS "Configured. Build artifacts stay in the build tree; 'cmake --install' controls what lands in your install prefix.")
