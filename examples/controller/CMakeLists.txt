# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

cmake_minimum_required(VERSION 3.23)
project(loopback_controller_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the configuration directory for the exploy package.
# By default, set it to the build directory.
set(exploy_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../build/control"
    CACHE PATH "Path to the exploy package configuration directory")
message(STATUS "Using exploy package configuration from: ${exploy_DIR}")
find_package(exploy REQUIRED CONFIG)

add_executable(loopback_controller_example
    main.cpp
)

target_include_directories(loopback_controller_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(loopback_controller_example PRIVATE
    exploy::exploy
)

target_compile_options(loopback_controller_example PRIVATE
    -Wall -Wextra -Wunused-variable -Wunused-parameter
)
