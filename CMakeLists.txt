cmake_minimum_required(VERSION 3.15)
project(NavierStokesHPC VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_OPENMP "Enable OpenMP" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(ENABLE_PROFILING "Enable profiling" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -pedantic")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /W4")
endif()

if(ENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

# Source files
set(SOURCES
    src/core/grid.cpp
    src/core/solver.cpp
)

include_directories(include)

# Core library
add_library(nssolver STATIC ${SOURCES})

# OpenMP
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(nssolver PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(nssolver PUBLIC USE_OPENMP)
endif()

# Main executable
add_executable(ns_main src/main.cpp)
target_link_libraries(ns_main nssolver)

# Define M_PI for MSVC
if(MSVC)
    target_compile_definitions(nssolver PRIVATE _USE_MATH_DEFINES)
    target_compile_definitions(ns_main PRIVATE _USE_MATH_DEFINES)
endif()

# Install
install(TARGETS ns_main nssolver
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP: ${USE_OPENMP}")
