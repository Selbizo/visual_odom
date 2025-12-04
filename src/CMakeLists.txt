#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# find_package(OpenCL REQUIRED)

# OpenCV
set(OpenCV_DIR "/media/pi/sd30gb/opencv340/opencv/build")
find_package(OpenCV REQUIRED)

add_executable(run main.cpp)
target_link_libraries(run ${OpenCV_LIBS})
