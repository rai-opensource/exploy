# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

cmake_minimum_required(VERSION 3.22)

# Keep this aligned with the Python version defined in pyproject.toml.
project(exploy LANGUAGES CXX VERSION 0.1.0)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Enable shared libraries and position independent code by default.
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")

# Configure RPATH.
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
    list(APPEND CMAKE_INSTALL_RPATH "@loader_path/../lib")
else()
    list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

# Treat warnings as errors.
add_compile_options(-Werror -Wpedantic -Wunused-variable -Wunused-parameter -Wreorder)

# Find dependencies.
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
find_package(onnxruntime REQUIRED)

# ==============
# Exploy library
# ==============

# C++ shared library.
add_library(exploy SHARED
    src/controller.cpp
    src/context.cpp
    src/onnx_runtime.cpp
    src/components.cpp
    src/matcher.cpp
)

# Create an alias target for symmetry between the build tree and install tree.
add_library(exploy::exploy ALIAS exploy)

# Set the C++ standard.
target_compile_features(exploy PUBLIC cxx_std_20)
set_target_properties(exploy PROPERTIES
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

# Include directories and public headers.
target_include_directories(exploy PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Find all header files in the include directory.
file(GLOB EXPLOY_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/exploy/*.hpp")

# Add public headers to the target.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    target_sources(exploy PUBLIC
        FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES ${EXPLOY_HEADERS}
    )
endif()

# Link dependencies.
target_link_libraries(exploy PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    fmt::fmt
    onnxruntime::onnxruntime
)

# =================================
# Exported targets and installation
# =================================

# Install the target (the library itself) and public headers.
if(CMAKE_VERSION VERSION_LESS 3.23)
    install(
        FILES ${EXPLOY_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/exploy
    )
    install(
        TARGETS exploy
        EXPORT exployTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
else()
    install(TARGETS exploy
        EXPORT exployTargets
        FILE_SET HEADERS
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Export the targets for use by downstream packages.
install(EXPORT exployTargets
    FILE exployTargets.cmake
    NAMESPACE exploy::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/exploy
)

# Generate and install the package configuration files.
export(EXPORT exployTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/exployTargets.cmake"
    NAMESPACE exploy::
)

# Create a ConfigVersion.cmake file.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/exployConfig.cmake.in"
"@PACKAGE_INIT@

include(CMakeFindDependencyMacro)
find_dependency(Eigen3)
find_dependency(nlohmann_json)
find_dependency(fmt)
find_dependency(onnxruntime)

include(\"\${CMAKE_CURRENT_LIST_DIR}/exployTargets.cmake\")
")

# Configure and install the package configuration files.
configure_package_config_file(
    "${CMAKE_CURRENT_BINARY_DIR}/exployConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/exployConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/exploy
)

# Create a version file for the config file.
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/exployConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install the config and version files.
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/exployConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/exployConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/exploy
)

# =====
# Tests
# =====

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  set(GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/test/testdata/test_onnx_generator.py")
  set(GEN_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/test/testdata/test.onnx")

  add_custom_command(
      OUTPUT "${GEN_OUTPUT}"
      COMMAND ${CMAKE_COMMAND} -E env PYTHONWARNINGS="ignore" ${Python3_EXECUTABLE} "${GEN_SCRIPT}" "${GEN_OUTPUT}"
      DEPENDS "${GEN_SCRIPT}"
      COMMENT "Generating test ONNX model..."
  )

  add_executable(${PROJECT_NAME}_test
      test/controller_test.cpp
      test/components_test.cpp
      test/context_test.cpp
      test/logging_test.cpp
      test/metadata_test.cpp
      test/onnx_runtime_test.cpp
      "${GEN_OUTPUT}"
  )
  target_compile_definitions(${PROJECT_NAME}_test PRIVATE
      TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/testdata"
  )
  target_link_libraries(${PROJECT_NAME}_test PRIVATE
      exploy::exploy
      GTest::gtest_main
      GTest::gmock
  )
  add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test)
endif()
