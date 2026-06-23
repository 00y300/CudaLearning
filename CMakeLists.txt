cmake_minimum_required(VERSION 3.18)
project(cuda-learning LANGUAGES CXX CUDA C)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# stb (header-only image lib) --------------------------------------------
include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_MakeAvailable(stb)

add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${stb_SOURCE_DIR})

add_library(stb_image_write INTERFACE)
target_include_directories(stb_image_write INTERFACE ${stb_SOURCE_DIR})

# Chapter targets --------------------------------------------------------
function(add_chapter name)
    add_executable(${name} ${ARGN})
    target_link_libraries(
        ${name}
        PRIVATE CUDA::cudart stb_image stb_image_write
    )
    target_compile_definitions(
        ${name}
        PRIVATE ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets"
    )
    set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES 120)
endfunction()

add_chapter(07-tiled-convolution 07-tiled-convolution/tiled-convolution.cu)
add_chapter(08-3d-stencil 08-3d-stencil/3d-stencil.cu)
add_chapter(09-histogram 09-histogram/histogram.cu)
add_chapter(10-reduction 10-reduction/reduction_sum.cu)
add_chapter(11-prefix-sum 11-prefix-sum/prefix_sum.cu)

# .clangd generation -----------------------------------------------------
get_filename_component(_nvcc_bin "${CUDAToolkit_NVCC_EXECUTABLE}" DIRECTORY)
get_filename_component(CUDA_PATH_FOR_CLANGD "${_nvcc_bin}" DIRECTORY)
message(STATUS "clangd --cuda-path -> ${CUDA_PATH_FOR_CLANGD}")
message(STATUS "clangd CUDA includes -> ${CUDAToolkit_INCLUDE_DIRS}")

set(_clangd_includes "")
foreach(_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
    string(APPEND _clangd_includes "    - \"-I${_dir}\"\n")
endforeach()

file(
    GENERATE OUTPUT "${CMAKE_SOURCE_DIR}/.clangd"
    CONTENT
        "CompileFlags:
  Remove:
    - \"--options-file*\"
    - \"-forward-unknown-to-host-compiler\"
    - \"--generate-code*\"
    - \"-gencode*\"
    - \"-arch=*\"
    - \"-rdc=*\"
    - \"-ccbin*\"
    - \"--expt-*\"
    - \"--threads*\"
    - \"-Xcompiler*\"
    - \"-Xcudafe*\"
    - \"-Xfatbin*\"
    - \"-Xlinker*\"
    - \"-Xptxas*\"
    - \"--device-debug\"
    - \"-G\"
  Add:
    - \"-xcuda\"
    - \"-std=c++17\"
    - \"--cuda-gpu-arch=sm_120\"
    - \"--cuda-path=${CUDA_PATH_FOR_CLANGD}\"
    - \"--no-cuda-version-check\"
    - \"-I${stb_SOURCE_DIR}\"
${_clangd_includes}Diagnostics:
  UnusedIncludes: None
"
)
