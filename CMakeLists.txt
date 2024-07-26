cmake_minimum_required(VERSION 3.20)

project(
  echoframe-tcbf
  DESCRIPTION "Echoframe tensor-core beamformer"
  VERSION 0.1
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")

include(FetchContent)

FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.2.0)
FetchContent_MakeAvailable(cxxopts)

FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers.git
  GIT_TAG main)
FetchContent_MakeAvailable(cudawrappers)

FetchContent_Declare(
  ccglib
  GIT_REPOSITORY https://git.astron.nl/RD/recruit/ccglib
  GIT_TAG packing_complex_last) # temporary to support complex-last data
FetchContent_MakeAvailable(ccglib)

add_library(tcbf SHARED src/tcbf.cu)
target_include_directories(tcbf PUBLIC include/)
target_link_libraries(tcbf ccglib cudawrappers::cu cudawrappers::nvrtc)

add_executable(${PROJECT_NAME}-standalone src/main.cu)
target_link_libraries(${PROJECT_NAME}-standalone cxxopts tcbf)

add_executable(${PROJECT_NAME}-prepare-a-matrix src/prepare_a_matrix.cu)
target_link_libraries(${PROJECT_NAME}-prepare-a-matrix cxxopts tcbf
                      OpenMP::OpenMP_CXX)
