cmake_minimum_required(VERSION 3.21)
project(
  RedoxiTrack
  VERSION 1.0
  LANGUAGES CXX)

# use c++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# enable position independent code always, for easy linking
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# export compile commands
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add cmake modules
# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# build what?
option(BUILD_EXAMPLES "Build examples" OFF)

if(BUILD_EXAMPLES)
  # To build examples, require opencv >= 4.8
  set(OPENCV_REQUIRED_VERSION "4.8")
  message(
    STATUS
      "OpenCV with version ${OPENCV_REQUIRED_VERSION} is required for building examples"
  )
  find_package(OpenCV ${OPENCV_REQUIRED_VERSION} REQUIRED)
else()
  # If BUILD_EXAMPLES is OFF, use whatever opencv is installed
  find_package(OpenCV REQUIRED)
endif()

# build basic library
add_subdirectory(src)

# build examples
if(BUILD_EXAMPLES)
  set(REDOXI_TEST_DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data)
  add_subdirectory(examples)
endif()
