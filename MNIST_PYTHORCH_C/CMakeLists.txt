project(MNIST_PYTHORCH_C C)
cmake_minimum_required(VERSION 3.12)
# set(CMAKE_C_COMPILER gcc)
# include_directories(${CMAKE_SOURCE_DIR}/cc/pythorch)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -msse2")
add_definitions(-DOPTIMIZE_IM2COL=1)
add_definitions(-DOPTIMIZE_GEMM=2)


add_subdirectory(cc/pythorch)
link_libraries(pythorch m)
include_directories(cc)

aux_source_directory(tests/train_model/export_code DIR_MNIST_EXAMPLE_SRCS)
add_executable(mnist_example tests/mnist.c ${DIR_MNIST_EXAMPLE_SRCS})
target_include_directories(mnist_example PUBLIC cc tests/train_model/export_code)
target_link_libraries(mnist_example pythorch)

aux_source_directory(tests/train_model/export_code/ DIR_TEST_COMPILER_SRCS)
add_executable(test_compiler tests/test_compiler/mnist.c ${DIR_TEST_COMPILER_SRCS})
target_include_directories(test_compiler PUBLIC cc tests/train_model/export_code)
target_link_libraries(test_compiler pythorch)

add_executable(test_simd tests/test_simd/scratch.c)
target_link_libraries(test_simd pythorch)