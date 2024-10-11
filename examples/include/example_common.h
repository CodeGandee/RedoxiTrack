#pragma once
#include <array>
#include <filesystem>
#include <string>
#include <vector>

// if REDOXI_TEST_DATA_DIR is not defined, raise compile error
#ifndef REDOXI_TEST_DATA_DIR
#error \
    "REDOXI_TEST_DATA_DIR is not defined, please define it to point to the data directory"
#endif

// if REDOXI_TEST_OUTPUT_DIR is not defined, raise compile error
#ifndef REDOXI_TEST_OUTPUT_DIR
#error \
    "REDOXI_TEST_OUTPUT_DIR is not defined, please define it to point to the output directory"
#endif

namespace RedoxiExamples
{
namespace Paths
{
const std::filesystem::path DataDir = REDOXI_TEST_DATA_DIR;
const std::filesystem::path OutputDir = REDOXI_TEST_OUTPUT_DIR;
const std::filesystem::path TemporaryDataDir =
    std::string(REDOXI_TEST_DATA_DIR) + "/tmp";
const std::filesystem::path ModelDir =
    std::string(REDOXI_TEST_DATA_DIR) + "/models";
} // namespace Paths

struct VideoTrackingSample {
    std::filesystem::path video;    // video file
    std::filesystem::path track_gt; // ground truth track file
};

/** Types of example data */
enum class ExampleData {
    None,             // no data, default type
    DancetrackSample, // sample video with ground truth track
};

/**
 * Get and print environment variable
 * @param env_var: environment variable name
 * @return std::string: environment variable value
 */
std::string get_and_print_env(const std::string &env_var);

/**
 * Get video tracking sample data
 * @param example_data: example data type
 * @return VideoTrackingSample: video and ground truth track file
 */
VideoTrackingSample get_video_tracking_sample(ExampleData example_data);

/** Get face detection model from opencv model zoo */
std::filesystem::path get_face_detection_model();

/** Get person detection model from opencv model zoo */
std::filesystem::path get_person_detection_model();

/** Get YOLOX model from opencv model zoo */
std::filesystem::path get_yolox_model();

/** Get YOLOX model with INT8 quantization from opencv model zoo */
std::filesystem::path get_yolox_model_int8();

const std::vector<std::array<int, 3>> &get_distinct_colors();
} // namespace RedoxiExamples