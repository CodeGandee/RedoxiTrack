cmake_minimum_required(VERSION 3.21)
project(RedoxiTrackExamples VERSION 0.1.0 LANGUAGES CXX)

option(WITH_EXAMPLE_TRACK_PERSONS "Build example track_persons" ON)
# option(WITH_EXAMPLE_TRACK_PERSON_LANDMARKS "Build example track_person_landmarks" OFF)
# option(WITH_EXAMPLE_TRACK_FACE "Build example track_faces" OFF)

# is this top level project?
# if yes, this project is compiled as standalone, we need to find the RedoxiTrack package
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(RedoxiTrack REQUIRED)
endif()

# building shared libs? if no, define REDOXI_TRACK_STATIC_LIBS
if(NOT BUILD_SHARED_LIBS)
    message(STATUS "Using static linking for RedoxiTrack examples")
    add_definitions(-DREDOXI_TRACK_STATIC_LIBS)
else()
    message(STATUS "Using shared linking for RedoxiTrack examples")
endif()

find_package(spdlog REQUIRED)

# check if REDOXI_TEST_DATA_DIR is set, if not raise error
if(NOT DEFINED REDOXI_TEST_DATA_DIR)
    message(FATAL_ERROR "REDOXI_TEST_DATA_DIR is not set")
endif()

# check if REDOXI_TEST_OUTPUT_DIR is set, if not, warn it
if(NOT DEFINED REDOXI_TEST_OUTPUT_DIR)
    set(REDOXI_TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/output)
    message(WARNING "REDOXI_TEST_OUTPUT_DIR is not set, output will be saved to ${REDOXI_TEST_OUTPUT_DIR}")
endif()

# create the output directory and pass it to the c++ preprocessor
# define the REDOXI_TEST_DATA_DIR and REDOXI_TEST_OUTPUT_DIR c++ preprocessor
file(MAKE_DIRECTORY ${REDOXI_TEST_OUTPUT_DIR})
add_definitions(-DREDOXI_TEST_DATA_DIR="${REDOXI_TEST_DATA_DIR}")
add_definitions(-DREDOXI_TEST_OUTPUT_DIR="${REDOXI_TEST_OUTPUT_DIR}")

# create targets
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
set(common_source_files 
    ${CMAKE_CURRENT_LIST_DIR}/src/example_common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/example_person_detector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/opencv_demo_person_det.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/opencv_demo_yolox.cpp
)
set(common_deps spdlog::spdlog RedoxiTrack::RedoxiTrack)

if(WITH_EXAMPLE_TRACK_PERSONS)
    set(yolox_model_name "object_detection_yolox_2022nov_int8.onnx")
    set(yolox_model_dst "${REDOXI_TEST_DATA_DIR}/tmp/${yolox_model_name}")
    set(yolox_model_candidate_path "${REDOXI_TEST_DATA_DIR}/models/${yolox_model_name}")

    # check if yolox model is found in dst path
    if(EXISTS ${yolox_model_dst})
        message(STATUS "Found YOLOX model in ${yolox_model_dst}")
    elseif(EXISTS ${yolox_model_candidate_path})
        message(STATUS "Found YOLOX model in ${yolox_model_candidate_path}")
        message(STATUS "Copying YOLOX model to ${yolox_model_dst}")
        file(COPY ${yolox_model_candidate_path} DESTINATION ${REDOXI_TEST_DATA_DIR}/tmp)
    else()
        set(yolox_model_url "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_yolox/${yolox_model_name}")
        message(STATUS "Downloading YOLOX model to ${yolox_model_dst}")
        file(DOWNLOAD ${yolox_model_url} ${yolox_model_dst})
    endif()
    
    # track person in video
    add_executable(track_persons ${CMAKE_CURRENT_LIST_DIR}/track_persons.cpp ${common_source_files})
    target_link_libraries(track_persons PRIVATE ${common_deps})
endif()

# track face in video
# if(WITH_EXAMPLE_TRACK_FACE)
#     # download face detection model
#     set(face_model_url "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
#     set(face_model_dst ${REDOXI_TEST_DATA_DIR}/models/face_detection_yunet_2023mar.onnx)

#     # download the face model to REDOXI_TEST_DATA_DIR/tmp
#     if(NOT EXISTS ${face_model_dst})
#         message(STATUS "Downloading face detection model to ${face_model_dst}")
#         file(DOWNLOAD ${face_model_url} ${face_model_dst})
#     endif()

#     add_executable(track_faces ${CMAKE_CURRENT_LIST_DIR}/track_faces.cpp ${common_source_files})
#     target_link_libraries(track_faces PRIVATE ${common_deps})
# endif()



# if(WITH_EXAMPLE_TRACK_PERSON_LANDMARKS)
#     # track person landmarks
#     # download person detection model
#     set(person_model_url "https://github.com/opencv/opencv_zoo/raw/main/models/person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx")
#     set(person_model_dst ${REDOXI_TEST_DATA_DIR}/tmp/person_detection_mediapipe_2023mar.onnx)

#     if(NOT EXISTS ${person_model_dst})
#         message(STATUS "Downloading person detection model to ${person_model_dst}")
#         file(DOWNLOAD ${person_model_url} ${person_model_dst})
#     endif()
#     add_executable(track_person_landmarks ${CMAKE_CURRENT_LIST_DIR}/track_person_landmarks.cpp ${common_source_files})
#     target_link_libraries(track_person_landmarks PRIVATE ${common_deps})
# endif()