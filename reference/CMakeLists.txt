cmake_minimum_required(VERSION 3.10)
project(StereoVisionSLAM)

cmake_policy(SET CMP0079 NEW)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Добавляем флаги для отключения проверки выравнивания и векторизации Eigen
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -fopenmp")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Suitesparse
include_directories("/usr/include/suitesparse/")

# G2O
Find_Package(G2O REQUIRED)

# Eigen
Find_Package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Sophus
Find_Package(Sophus REQUIRED)

# csparse
Find_Package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# fmt
find_package(fmt REQUIRED)

# rerun (фиксируем версию 0.16.1)
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/download/0.16.1/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

# pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Для диагностики – выведем путь к PCL_COMMON_LIBRARY
message(STATUS "PCL_COMMON_LIBRARY = ${PCL_COMMON_LIBRARY}")

# Библиотеки g2o
set(G2O_LIBS
    g2o_csparse_extension
    g2o_stuff
    g2o_core
    g2o_types_slam3d
    g2o_solver_csparse
)

# Все сторонние библиотеки
set(THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
    Sophus::Sophus
    ${G2O_LIBS}
    pthread
    ${CSPARSE_LIBRARY}
    rerun_sdk
    fmt::fmt
    ${PCL_LIBRARIES}
)

# Глобальный флаг для линковщика
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")

############### source ######################
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
add_subdirectory(app)

# ---- НИЖЕ НЕТ target_link_options / target_link_libraries для приложений ----