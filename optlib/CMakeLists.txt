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

file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp
)

foreach(example_src ${EXAMPLE_SOURCES})
  get_filename_component(example_name ${example_src} NAME_WE)
  add_executable(${example_name} ${example_src})
  target_link_libraries(${example_name} PRIVATE optlib)
endforeach()