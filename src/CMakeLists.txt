include(CMakePackageConfigHelpers)

find_package(OpenCV REQUIRED)
# print where OpenCV was found
message(STATUS "Inside src, OpenCV found at ${OpenCV_DIR}")

find_package(Eigen3 REQUIRED)

# are we building shared libs? If yes, set REDOXI_TRACK_EXPORT
if(BUILD_SHARED_LIBS)
    message(STATUS "Building shared libraries")
    add_definitions(-DREDOXI_TRACK_EXPORT)
else()
    message(STATUS "Building static libraries, and linking statically")
    add_definitions(-DREDOXI_TRACK_STATIC_LIBS)
endif()

set(detection
    ${CMAKE_CURRENT_LIST_DIR}/detection/DeepSortTrackTarget.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detection/SimpleSortTrackTarget.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detection/KalmanTrackTarget.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detection/PersonDetection.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detection/SingleDetection.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detection/TrackTarget.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detection/BotsortTrackTarget.cpp)

set(external
    ${CMAKE_CURRENT_LIST_DIR}/external/Hungarian.cpp
    ${CMAKE_CURRENT_LIST_DIR}/external/lapjv.cpp)

set(tracker
    ${CMAKE_CURRENT_LIST_DIR}/tracker/FeatureBaseDetTraits.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/BotsortMotionPrediction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/BotsortTracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/BotsortTrackerParam.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/BotsortKalmanTracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/DeepSortMotionPrediction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/DeepSortTracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/DeepSortTrackerParam.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/SimpleSortMotionPrediction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/SimpleSortTracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/SimpleSortTrackerParam.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/KalmanTracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/OpencvOpticalFlow.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/OpticalFlowTracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/OpticalTrackerParam.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/SortMotionPrediction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/TrackerBase.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker/TrackerParam.cpp)

set(utils
    ${CMAKE_CURRENT_LIST_DIR}/utils/utility_functions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/CosineFeature.cpp)

set(REDOXI_TRACKER_LINK_LIBS  ${OpenCV_LIBS} Eigen3::Eigen)
set(REDOXI_TRACKER_SRC_FILES ${detection} ${external} ${tracker} ${utils})

add_library(RedoxiTrack ${REDOXI_TRACKER_SRC_FILES})

# alias with namespace
add_library(RedoxiTrack::RedoxiTrack ALIAS RedoxiTrack)

target_include_directories(RedoxiTrack PUBLIC ${OpenCV_INCLUDE_DIRS})

# set include directories, INSTALL_INTERFACE handled by install(EXPORT)
target_include_directories(RedoxiTrack PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(RedoxiTrack PUBLIC ${REDOXI_TRACKER_LINK_LIBS})

# ===== installation =====
set(ProjectName RedoxiTrack)

# export target
install(TARGETS RedoxiTrack EXPORT RedoxiTrackTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(EXPORT RedoxiTrackTargets
    FILE RedoxiTrackTargets.cmake
    NAMESPACE RedoxiTrack::
    DESTINATION lib/cmake/RedoxiTrack
)

# create config file
configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/RedoxiTrackConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/RedoxiTrackConfig.cmake
    INSTALL_DESTINATION lib/cmake/RedoxiTrack
)

# install config file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/RedoxiTrackConfig.cmake
    DESTINATION lib/cmake/RedoxiTrack
)

# create version file
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/RedoxiTrackConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# install version file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/RedoxiTrackConfigVersion.cmake
    DESTINATION lib/cmake/RedoxiTrack
)

# copy include to install directory
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/RedoxiTrack DESTINATION include)

# copy project/cmake to install directory
install(DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/ DESTINATION cmake/RedoxiTrack)
