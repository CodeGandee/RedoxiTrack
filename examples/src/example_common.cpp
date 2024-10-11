#include "example_common.h"
#include <cstdlib>
#include <map>
#include <spdlog/spdlog.h>
#include <stdexcept>

static std::vector<std::array<int, 3>> distinct_colors = {{
    {128, 128, 128}, // gray
    {192, 192, 192}, // silver
    {47, 79, 79},    // darkslategray
    {85, 107, 47},   // darkolivegreen
    {160, 82, 45},   // sienna
    {165, 42, 42},   // brown
    {128, 128, 0},   // olive
    {72, 61, 139},   // darkslateblue
    {0, 128, 0},     // green
    {60, 179, 113},  // mediumseagreen
    {184, 134, 11},  // darkgoldenrod
    {70, 130, 180},  // steelblue
    {0, 0, 128},     // navy
    {210, 105, 30},  // chocolate
    {154, 205, 50},  // yellowgreen
    {32, 178, 170},  // lightseagreen
    {50, 205, 50},   // limegreen
    {127, 0, 127},   // purple2
    {143, 188, 143}, // darkseagreen
    {176, 48, 96},   // maroon3
    {153, 50, 204},  // darkorchid
    {255, 0, 0},     // red
    {255, 140, 0},   // darkorange
    {255, 215, 0},   // gold
    {106, 90, 205},  // slateblue
    {255, 255, 0},   // yellow
    {0, 0, 205},     // mediumblue
    {222, 184, 135}, // burlywood
    {0, 255, 0},     // lime
    {0, 255, 127},   // springgreen
    {220, 20, 60},   // crimson
    {0, 255, 255},   // aqua
    {0, 191, 255},   // deepskyblue
    {244, 164, 96},  // sandybrown
    {0, 0, 255},     // blue
    {160, 32, 240},  // purple3
    {240, 128, 128}, // lightcoral
    {173, 255, 47},  // greenyellow
    {255, 99, 71},   // tomato
    {255, 0, 255},   // fuchsia
    {240, 230, 140}, // khaki
    {100, 149, 237}, // cornflower
    {221, 160, 221}, // plum
    {144, 238, 144}, // lightgreen
    {255, 20, 147},  // deeppink
    {175, 238, 238}, // paleturquoise
    {238, 130, 238}, // violet
    {127, 255, 212}, // aquamarine
    {255, 105, 180}, // hotpink
    {255, 182, 193}  // lightpink
}};

namespace RedoxiExamples
{
static const std::map<ExampleData, VideoTrackingSample> video_tracking_samples = {
    {ExampleData::DancetrackSample, {Paths::DataDir / "videos" / "dancetrack-0039.mp4", Paths::DataDir / "videos" / "dancetrack-0039.gt.txt"}}};

std::string get_and_print_env(const std::string &env_var)
{
    const char *env = std::getenv(env_var.c_str());
    if (env == nullptr) {
        spdlog::warn("Environment variable {} not set", env_var);
        return "";
    } else {
        spdlog::info("Env: {}={}", env_var, env);
        return env;
    }
}

VideoTrackingSample get_video_tracking_sample(ExampleData example_data)
{
    VideoTrackingSample sample;
    if (video_tracking_samples.find(example_data) == video_tracking_samples.end()) {
        throw std::runtime_error("Example data not found");
    } else {
        sample = video_tracking_samples.at(example_data);
        return sample;
    }
}

std::filesystem::path get_face_detection_model()
{
    return Paths::ModelDir / "face_detection_yunet_2023mar.onnx";
}

std::filesystem::path get_person_detection_model()
{
    return Paths::DataDir / "tmp" / "person_detection_mediapipe_2023mar.onnx";
}

std::filesystem::path get_yolox_model()
{
    return Paths::DataDir / "tmp" / "object_detection_yolox_2022nov.onnx";
}

std::filesystem::path get_yolox_model_int8()
{
    return Paths::DataDir / "tmp" / "object_detection_yolox_2022nov_int8.onnx";
}

const std::vector<std::array<int, 3>> &get_distinct_colors()
{
    return distinct_colors;
}
}; // namespace RedoxiExamples