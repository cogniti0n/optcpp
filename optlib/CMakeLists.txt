cmake_minimum_required(VERSION 3.16)
project(optlib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Eigen
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)

FetchContent_MakeAvailable(eigen)

# optlib library (header-only for now)
add_library(optlib INTERFACE)

target_include_directories(optlib INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(optlib INTERFACE Eigen3::Eigen)

# Example executable
add_executable(gd_quadratic examples/gd_quadratic.cpp)
target_link_libraries(gd_quadratic PRIVATE optlib)
add_executable(newton_quadratic examples/newton_quadratic.cpp)
target_link_libraries(newton_quadratic PRIVATE optlib)