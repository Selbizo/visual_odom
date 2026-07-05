find_package( OpenCV REQUIRED )

# Add subdirectories for sub-libraries
add_subdirectory(videoStabilization)
add_subdirectory(evaluate)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(videoStabilization)
include_directories(evaluate)

# Enable CUDA if requested
if(USE_CUDA)
    find_package(CUDA)
    include_directories("${CUDA_INCLUDE_DIRS}")
    add_definitions(-DUSE_CUDA)
endif(USE_CUDA)

# ---- Shared libraries ----
add_library(feature SHARED "feature.cpp")
target_link_libraries(feature ${OpenCV_LIBS} bucket)

add_library(bucket SHARED "bucket.cpp")
target_link_libraries(bucket ${OpenCV_LIBS})

add_library(utils SHARED "utils.cpp")
target_link_libraries(utils evaluate_odometry feature ${OpenCV_LIBS})

add_library(visualOdometry SHARED "visualOdometry.cpp")
target_link_libraries(visualOdometry utils bucket feature Frame ${OpenCV_LIBS})

add_library(rgbd_standalone SHARED "rgbd_standalone.cpp")
target_link_libraries(rgbd_standalone ${OpenCV_LIBS})

add_library(Frame SHARED "Frame.cpp")
target_link_libraries(Frame ${OpenCV_LIBS})

# ---- Output directory ----
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
message(STATUS "Project source dir: ${PROJECT_SOURCE_DIR}")

# ---- Main executable ----
add_executable(run main.cpp)
target_link_libraries(run ${OpenCV_LIBS} bucket feature utils visualOdometry Frame rgbd_standalone basicFunctions stabilizationFunctions wienerFilter kalmanSplitter)


# add_subdirectory(videoStabilization)
# include_directories(${OpenCV_INCLUDE_DIRS} )
# include_directories(videoStabilization)


# if(USE_CUDA)
#     find_package(CUDA)
#     include_directories("${CUDA_INCLUDE_DIRS}")
#     add_definitions(-DUSE_CUDA)
# endif(USE_CUDA)


# add_executable( run videoStabilization.cpp )
# target_link_libraries( run ${OpenCV_LIBS} basicFunctions stabilizationFunctions wienerFilter)
