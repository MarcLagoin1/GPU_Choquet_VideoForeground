set(CMAKE_CUDA_COMPILER "/nix/store/9gsyidi79iqvsczzdwnx13pj2cwfccrb-cudatoolkit-11.7.0/bin/nvcc")
set(CUDA_NVCC_EXECTUABLE "/nix/store/9gsyidi79iqvsczzdwnx13pj2cwfccrb-cudatoolkit-11.7.0/bin")
set(CUDA_TOOLKIT_ROOT_DIR $CUDA_PATH)
set(CUDA_CUDART_LIBRARY "/p/nix/store/cfwcn5kvvcg2j13hvf9cv7siwvkjgvni-cudatoolkit-11.7.0-lib")

#THANK YOU VERY MUCH FOR PUTTING THE CUDART LIB IN A COMPLETELY DIFFERENT DIRECTORY THAN THE CUDA_TOOLKIT DIR

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")

cmake_minimum_required(VERSION 3.24)
project(gpu CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "-arch=sm_52 -lineinfo")
set(CMAKE_CUDA_ARCHITECTURES "61")

find_package(PNG REQUIRED)
find_package(CUDA REQUIRED)

add_executable(cpu
        src/cpu/main.cpp
        src/cpu/Image.cpp src/cpu/Image.h
        src/cpu/ChoquetIntegral.cpp src/cpu/ChoquetIntegral.h
        src/cpu/FeaturesExtraction.cpp src/cpu/FeaturesExtraction.h
        src/cpu/FuzzyChoquetIntegral.cpp src/cpu/FuzzyChoquetIntegral.h
        src/cpu/VideoProcessingExecutor.cpp src/cpu/VideoProcessingExecutor.h
        src/benchmarks.cpp src/benchmarks.h
        src/utils.cpp src/utils.h)

add_executable(gpu
        src/gpu/main.cpp
        src/gpu/Image.cpp src/cpu/Image.h
        src/gpu/FeaturesExtraction.cu src/cpu/FeaturesExtraction.h
        src/gpu/FuzzyChoquetIntegral.cu src/cpu/FuzzyChoquetIntegral.h
        src/gpu/VideoProcessingExecutor.cpp src/gpu/VideoProcessingExecutor.h
        src/benchmarks.cpp src/benchmarks.h
        src/utils.cpp src/utils.h
        )

add_executable(optimized_cpu
        src/optimized_cpu/main.cpp
        src/optimized_cpu/Image.cpp src/optimized_cpu/Image.h
        src/optimized_cpu/FuzzyChoquetIntegral.cpp src/optimized_cpu/FuzzyChoquetIntegral.h
        src/optimized_cpu/FeaturesExtraction.cpp src/optimized_cpu/FeaturesExtraction.h
        src/optimized_cpu/VideoProcessingExecutor.cpp src/optimized_cpu/VideoProcessingExecutor.h
        src/benchmarks.cpp src/benchmarks.h
        src/utils.cpp src/utils.h
        )

add_executable(optimized_gpu
        src/optimized_gpu/main.cpp
        src/optimized_gpu/Image.cpp src/optimized_gpu/Image.h
        src/optimized_gpu/FuzzyChoquetIntegral.cu src/optimized_gpu/FuzzyChoquetIntegral.h
        src/optimized_gpu/FeaturesExtraction.cu src/optimized_gpu/FeaturesExtraction.h
        src/optimized_gpu/VideoProcessingExecutor.cpp src/optimized_gpu/VideoProcessingExecutor.h
        src/benchmarks.cpp src/benchmarks.h
        src/utils.cpp src/utils.h
        )

target_link_libraries(cpu PRIVATE PNG::PNG)
target_link_libraries(optimized_cpu PRIVATE PNG::PNG ${CUDA_cudart_LIBRARY})

target_link_libraries(gpu PRIVATE PNG::PNG ${CUDA_cudart_LIBRARY})
set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(optimized_gpu PRIVATE PNG::PNG ${CUDA_cudart_LIBRARY})
set_target_properties(optimized_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)