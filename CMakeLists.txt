cmake_minimum_required(VERSION 3.24.3)
project(SUPerman LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenMP REQUIRED)

# Source files
set(COMMON_HEADER_FILES args.hpp cpu_algos.hpp flags.h MatrixMarketIOLibrary.h read_matrix.hpp util.h AlgorithmRecommender.h)
set(COMMON_SOURCE_FILES main.cpp MatrixMarketIOLibrary.cpp AlgorithmRecommender.cpp)
set(CUDA_HEADER_FILES gpu_wrappers.h)
set(CUDA_SOURCE_FILES gpu_exact_dense.cu gpu_exact_sparse.cu)

# CPU-only Executable
add_executable(cpu_perman ${COMMON_HEADER_FILES} ${COMMON_SOURCE_FILES})
target_compile_definitions(cpu_perman PRIVATE ONLYCPU)
target_link_libraries(cpu_perman OpenMP::OpenMP_CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    find_package(MPI)

    # GPU-only Executable
    add_executable(gpu_perman ${COMMON_HEADER_FILES} ${COMMON_SOURCE_FILES} ${CUDA_HEADER_FILES} ${CUDA_SOURCE_FILES})
    target_link_libraries(gpu_perman OpenMP::OpenMP_CXX CUDA::cudart)
    target_link_libraries(gpu_perman ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})

    # Combined CPU-GPU-MPI Executable
    add_executable(mpi_perman ${COMMON_HEADER_FILES} ${COMMON_SOURCE_FILES} ${CUDA_HEADER_FILES} ${CUDA_SOURCE_FILES})
    target_compile_definitions(mpi_perman PRIVATE MPIENABLED)
    target_link_libraries(mpi_perman OpenMP::OpenMP_CXX CUDA::cudart MPI::MPI_CXX)
    target_link_libraries(mpi_perman ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})
endif()
