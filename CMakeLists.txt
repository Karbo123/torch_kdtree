cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_kdtree LANGUAGES CXX CUDA)

find_package(Torch 1.8 REQUIRED)
find_package(Python3 3.7 REQUIRED Interpreter Development)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w -Xptxas='-w' -Xcompiler='-fopenmp -funroll-loops'")

file(GLOB COMMON_SOURCES 
          ${CMAKE_SOURCE_DIR}/src/buildKdTree.cu
          ${CMAKE_SOURCE_DIR}/src/Gpu.cu
          ${CMAKE_SOURCE_DIR}/src/mergeSort.cu
          ${CMAKE_SOURCE_DIR}/src/removeDups.cu
    )

########################################################
add_executable(kdTreeGPUsms ${COMMON_SOURCES} ${CMAKE_SOURCE_DIR}/src/KdTreeGPUsms.cu)
target_compile_features(kdTreeGPUsms PUBLIC cxx_std_14) 

target_link_libraries(kdTreeGPUsms -lgomp)

target_include_directories(kdTreeGPUsms PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(kdTreeGPUsms PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(kdTreeGPUsms PUBLIC "/usr/local/cuda/samples/common/inc")

########################################################

add_library(${PROJECT_NAME} SHARED ${COMMON_SOURCES} ${CMAKE_SOURCE_DIR}/src/torch_kdtree.cu)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14) 
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")

target_link_libraries(${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(${PROJECT_NAME} -lgomp)

target_compile_definitions(${PROJECT_NAME} PUBLIC TORCH_EXTENSION_NAME=${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${TORCH_CXX_FLAGS})

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(${PROJECT_NAME} PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC "/usr/local/cuda/samples/common/inc")
