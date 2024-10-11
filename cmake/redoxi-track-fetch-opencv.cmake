# define fetch content location
include(redoxi-track-fetch-common)

function(fetch_opencv)
    set(oneValueArgs WITH_CONTRIB WITH_QT)
    cmake_parse_arguments(FETCH_OPENCV "" "${oneValueArgs}" "" ${ARGN})

    # use fetched opencv if OPENCV_USE_LATEST is set
    # set(BUILD_opencv_python3 "OFF")
    set(BUILD_opencv_apps "OFF")
    set(BUILD_EXAMPLES "OFF")
    set(BUILD_DOCS "OFF")
    set(BUILD_TESTS "OFF")
    set(BUILD_PERF_TESTS "OFF")
    set(BUILD_opencv_python3 "OFF")
    set(BUILD_opencv_python2 "OFF")
    set(WITH_QT OFF)
    if(DEFINED FETCH_OPENCV_WITH_QT)
        set(WITH_QT ${FETCH_OPENCV_WITH_QT})
    endif()

    FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG        4.x
        GIT_SHALLOW    1 # only fetch the latest commit
    )
    FetchContent_MakeAvailable(opencv)
    # message(STATUS "OpenCV source dir: ${opencv_SOURCE_DIR}")
    # message(STATUS "OpenCV binary dir: ${opencv_BINARY_DIR}")
    # make sure downstream find_package() can find this opencv
    set(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
endfunction()