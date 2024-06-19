cmake_minimum_required(VERSION 3.20)

project(
  echoframe-beamformer
  DESCRIPTION "Echoframe tensor-core beamformer"
  VERSION 0.1
  LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)

FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers.git
  GIT_TAG main)
FetchContent_MakeAvailable(cudawrappers)

FetchContent_Declare(
  ccglib
  GIT_REPOSITORY https://git.astron.nl/RD/recruit/ccglib
  GIT_TAG temp-medical)
FetchContent_MakeAvailable(ccglib)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} ccglib cudawrappers::cu
                      cudawrappers::nvrtc)
