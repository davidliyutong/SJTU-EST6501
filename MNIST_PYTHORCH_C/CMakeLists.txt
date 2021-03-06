project(MNIST_PYTHORCH_C C)
cmake_minimum_required(VERSION 3.12)
# set(CMAKE_C_COMPILER gcc)

# Optimization Options
set(CMAKE_BUILD_TYPE "Release") # Debug, Release, RelWithDebInfo, MinSizeRel
add_compile_options(-mavx)
add_definitions(-DOPTIMIZE_IM2COL)
add_definitions(-DOPTIMIZE_GEMM_AVX=2)
add_definitions(-DOPTIMIZE_INDEX)

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

aux_source_directory(tests/train_model/export_code_stm32/ DIR_TEST_STM32_EXPORT_SRCS)
add_executable(test_stm32_export tests/test_compiler/mnist.c ${DIR_TEST_STM32_EXPORT_SRCS})
target_include_directories(test_stm32_export PUBLIC cc tests/train_model/export_code_stm32)
target_link_libraries(test_stm32_export pythorch)