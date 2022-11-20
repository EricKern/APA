cmake_minimum_required(VERSION 3.20.1)

set(CMAKE_CUDA_ARCHITECTURES "61;75")

project(NVBenchDemo CUDA)


find_package(nvbench REQUIRED)

add_executable(bandwidth.out bandwidth.cu)
target_link_libraries(bandwidth.out nvbench::main)
target_include_directories(bandwidth.out PRIVATE "${PROJECT_SOURCE_DIR}/include")

# add_executable(1_3cpy 1_3cpy.cu)
# # target_link_libraries(1_3cpy nvbench::main)
# target_include_directories(1_3cpy PRIVATE "${PROJECT_SOURCE_DIR}")