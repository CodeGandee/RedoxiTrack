#include <RedoxiTrack/RedoxiTrack.h>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

#include "RedoxiTrack/tracker/BotsortTracker.h"
#include "RedoxiTrack/tracker/DeepSortTracker.h"
#include "RedoxiTrack/tracker/OpticalFlowTracker.h"
#include "RedoxiTrack/tracker/TrackingEventHandler.h"
#include "example_common.h"
#include "example_person_detector.h"
#include "opencv_demo_yolox.h"

namespace rxt = RedoxiTrack;
namespace ex = RedoxiExamples;
namespace fs = std::filesystem;

const int DO_N_FRAME = 3000;
struct YoloxConfig;

std::shared_ptr<cv_yolox::YoloX> load_model();
std::shared_ptr<RedoxiExamples::PersonBodyDetector> create_body_detector(const std::shared_ptr<cv_yolox::YoloX> &model);
std::shared_ptr<rxt::DeepSortTracker> create_deepsort_tracker_with_optical_flow(cv::Size image_size);
std::shared_ptr<rxt::DeepSortTracker> create_deepsort_tracker_wo_optical_flow(cv::Size image_size);
std::shared_ptr<rxt::BotsortTracker> create_botsort_tracker_with_optical_flow(cv::Size image_size);
std::shared_ptr<rxt::BotsortTracker> create_botsort_tracker_wo_optical_flow(cv::Size image_size);
std::shared_ptr<rxt::OpticalFlowTracker> create_optical_flow_tracker(cv::Size image_size);

int main()
{
    spdlog::info("OpenCV version: {}", CV_VERSION);

    // enable visualization?
    // disabled by default, unless explicitly enabled
    bool use_visualization =
        ex::get_and_print_env("REDOXI_EXAMPLE_ENABLE_VISUALIZATION") == "1";

    // save by default, unless explicitly disabled
    bool use_save_output =
        !(ex::get_and_print_env("REDOXI_EXAMPLE_SAVE_OUTPUT") == "0");
    ex::get_and_print_env("CUDA_VISIBLE_DEVICES");

    // load model and create person detector
    auto net = load_model();
    auto detector = create_body_detector(net);

    // load video
    auto video_sample =
        ex::get_video_tracking_sample(ex::ExampleData::DancetrackSample);
    spdlog::info("Loading video file: {}", video_sample.video.string());
    cv::VideoCapture cap(video_sample.video.string());
    if (!cap.isOpened()) {
        spdlog::error("Failed to open video file: {}", video_sample.video.string());
        return 1;
    }

    // get video frame size
    cv::Size frame_size = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH),
                                   cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // create tracker
    // auto tracker_with_optical_flow = create_deepsort_tracker_with_optical_flow(frame_size);
    // auto tracker_wo_optical_flow = create_deepsort_tracker_wo_optical_flow(frame_size);
    auto tracker_with_optical_flow = create_botsort_tracker_with_optical_flow(frame_size);
    auto tracker_wo_optical_flow = create_botsort_tracker_wo_optical_flow(frame_size);

    // create output dir
    fs::path output_dir =
        fs::path(ex::Paths::OutputDir) / "track_person_in_video";
    fs::create_directories(output_dir);

    // for each frame, detect faces
    cv::Mat frame;
    int ith_frame = 0;
    if (use_visualization) {
        cv::namedWindow("frames", cv::WINDOW_FREERATIO);
    }
    while (cap.read(frame)) {
        spdlog::info("Processing frame {}, size: {}x{}, channels: {}", ith_frame,
                     frame.cols, frame.rows, frame.channels());
        spdlog::info("Detecting persons ...");
        auto person_list = detector->detect(frame);
        spdlog::info("Detected {} persons", person_list.size());

        // if this is the first frame, call begin_track()
        std::vector<rxt::DetectionPtr> _detlist;
        std::transform(person_list.begin(), person_list.end(), std::back_inserter(_detlist),
                       [](const auto &p) { return std::dynamic_pointer_cast<rxt::Detection>(p); });
        if (ith_frame == 0) {
            tracker_with_optical_flow->begin_track(frame, _detlist, ith_frame);
            tracker_wo_optical_flow->begin_track(cv::Mat(), _detlist, ith_frame);  // 在不用光流的情况下，这里可以只给一个空的frame
        } else {
            tracker_with_optical_flow->track(frame, _detlist, ith_frame);
            tracker_wo_optical_flow->track(cv::Mat(), _detlist, ith_frame);  // 在不用光流的情况下，这里可以只给一个空的frame
        }

        // draw bounding boxes if requested
        if (use_visualization || use_save_output) {
            cv::Mat frame_annotated_with_optical_flow = frame.clone();
            cv::putText(frame_annotated_with_optical_flow, "optical flow", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 2);
            cv::Mat frame_annotated_wo_optical_flow = frame.clone();
            cv::putText(frame_annotated_wo_optical_flow, "no optical flow", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 2);
            for (const auto &person : person_list) {
                auto bbox = person->get_bbox();

                // create rotated rect from bbox
                cv::RotatedRect ellipse;
                ellipse.center = cv::Point2f(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
                ellipse.size = cv::Size2f(bbox.width, bbox.height);

                // draw
                cv::ellipse(frame_annotated_with_optical_flow, ellipse, cv::Scalar(0, 255, 0), 1);
                cv::ellipse(frame_annotated_wo_optical_flow, ellipse, cv::Scalar(0, 255, 0), 1);
            }

            const auto &id2color = RedoxiExamples::get_distinct_colors();

            auto track_targets_with_optical_flow = tracker_with_optical_flow->get_all_open_targets();
            for (const auto &it : track_targets_with_optical_flow) {
                auto target_id = it.first;
                auto target = it.second;
                auto bbox = target->get_bbox();
                auto color = id2color[target_id % id2color.size()];
                cv::rectangle(frame_annotated_with_optical_flow, bbox, cv::Scalar(color[0], color[1], color[2]), 2);
                cv::putText(frame_annotated_with_optical_flow, std::to_string(target_id), cv::Point(bbox.x, bbox.y), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(color[0], color[1], color[2]), 3);
            }

            auto track_targets_wo_optical_flow = tracker_wo_optical_flow->get_all_open_targets();
            for (const auto &it : track_targets_wo_optical_flow) {
                auto target_id = it.first;
                auto target = it.second;
                auto bbox = target->get_bbox();
                auto color = id2color[target_id % id2color.size()];
                cv::rectangle(frame_annotated_wo_optical_flow, bbox, cv::Scalar(color[0], color[1], color[2]), 2);
                cv::putText(frame_annotated_wo_optical_flow, std::to_string(target_id), cv::Point(bbox.x, bbox.y), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(color[0], color[1], color[2]), 3);
            }

            // concat two frames
            cv::Mat frame_annotated;
            cv::hconcat(frame_annotated_with_optical_flow, frame_annotated_wo_optical_flow, frame_annotated);

            if (use_visualization) {
                cv::imshow("frames", frame_annotated);
                cv::waitKey(1.0 / 60 * 1000); // 60 fps
            }

            // write to file?
            if (use_save_output) {
                auto filename = cv::format("frame_annotated_%08d.jpg", ith_frame);
                std::string output_path = (output_dir / filename).string();
                spdlog::info("Writing annotated frame to file: {}", output_path);
                cv::imwrite(output_path, frame_annotated);
            }
        }

        ith_frame++;

        if (ith_frame > DO_N_FRAME) {
            break;
        }
    }

    // finish tracking, must be called after tracking is done
    tracker_with_optical_flow->finish_track();
    tracker_wo_optical_flow->finish_track();
    if (use_visualization) {
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}

struct YoloxConfig {
    double confidence_threshold = 0.35;       // lower bound
    double nms_threshold = 0.7;               // non-maximum suppression threshold
    double object_threshold = 0.3;            // objectness threshold
    cv::Size input_size = cv::Size(640, 640); // required by the model
    cv::dnn::Backend backend = cv::dnn::Backend::DNN_BACKEND_OPENCV;
    cv::dnn::Target target = cv::dnn::Target::DNN_TARGET_CPU;

    YoloxConfig()
    {
        backend = get_backend();
        target = get_target();
    }

    cv::dnn::Backend get_backend() const {
        // 检查编译时是否启用了 CUDA
    #ifdef HAVE_CUDA
        // 运行时检查 CUDA 设备
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            return cv::dnn::DNN_BACKEND_CUDA;
        }
    #endif
        return cv::dnn::DNN_BACKEND_OPENCV;
    }

    cv::dnn::Target get_target() const {
    #ifdef HAVE_CUDA
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            return cv::dnn::DNN_TARGET_CUDA;
        }
    #endif
        return cv::dnn::DNN_TARGET_CPU;
    }
};

std::shared_ptr<cv_yolox::YoloX> load_model()
{
    auto model_file = ex::get_yolox_model_int8();

    // check models
    if (!fs::exists(model_file)) {
        spdlog::error("Model file not found: {}", model_file.string());
        throw std::runtime_error("Model file not found");
    }

    spdlog::info("Loading model from file: {}", model_file.string());
    YoloxConfig config;
    if (config.backend == cv::dnn::Backend::DNN_BACKEND_CUDA) {
        spdlog::info("Using CUDA backend");
    } else {
        spdlog::info("Using OpenCV backend");
    }
    auto net = std::make_shared<cv_yolox::YoloX>(
        model_file.string(), config.confidence_threshold, config.nms_threshold,
        config.object_threshold, config.backend, config.target);
    spdlog::info("Model loaded.");

    return net;
}

std::shared_ptr<rxt::DeepSortTracker> create_deepsort_tracker_with_optical_flow(cv::Size image_size)
{
    // auto tracker = std::make_shared<rxt::DeepSortTracker>();
    auto tracker = std::make_shared<rxt::DeepSortTracker>();

    // init tracker
    rxt::DeepSortTrackerParam params;
    // 创建光流param就使用光流
    params.m_use_optical_before_track = true; // 可开可不开
    params.m_optical_param = std::make_shared<rxt::OpticalTrackerParam>();
    params.m_kalman_param = std::make_shared<rxt::TrackerParam>();

    params.set_preferred_image_size(image_size);
    tracker->init(params);
    return tracker;
}

std::shared_ptr<rxt::DeepSortTracker> create_deepsort_tracker_wo_optical_flow(cv::Size image_size)
{
    // auto tracker = std::make_shared<rxt::DeepSortTracker>();
    auto tracker = std::make_shared<rxt::DeepSortTracker>();

    // init tracker
    rxt::DeepSortTrackerParam params;
    // 不创建光流param就不使用光流
    params.m_use_optical_before_track = false; // 不可开
    params.m_kalman_param = std::make_shared<rxt::TrackerParam>();

    params.set_preferred_image_size(image_size);
    tracker->init(params);
    return tracker;
}

std::shared_ptr<rxt::BotsortTracker> create_botsort_tracker_with_optical_flow(cv::Size image_size)
{
    auto tracker = std::make_shared<rxt::BotsortTracker>();

    // init tracker
    rxt::BotsortTrackerParam params;
    // 创建光流param就使用光流
    params.m_use_optical_before_track = true; // 可开可不开
    params.m_optical_param = std::make_shared<rxt::OpticalTrackerParam>();
    params.m_kalman_param = std::make_shared<rxt::TrackerParam>();
    params.set_preferred_image_size(image_size);
    tracker->init(params);
    return tracker;
}

std::shared_ptr<rxt::BotsortTracker> create_botsort_tracker_wo_optical_flow(cv::Size image_size)
{
    auto tracker = std::make_shared<rxt::BotsortTracker>();

    // init tracker
    rxt::BotsortTrackerParam params;
    // 不创建光流param就不使用光流
    params.m_use_optical_before_track = false; // 不可开
    params.m_kalman_param = std::make_shared<rxt::TrackerParam>();
    params.set_preferred_image_size(image_size);
    tracker->init(params);
    return tracker;
}

std::shared_ptr<rxt::OpticalFlowTracker> create_optical_flow_tracker(cv::Size image_size)
{
    auto tracker = std::make_shared<rxt::OpticalFlowTracker>();
    rxt::OpticalTrackerParam params;
    params.set_preferred_image_size(image_size);
    tracker->init(params);
    return tracker;
}

std::shared_ptr<RedoxiExamples::PersonBodyDetector> create_body_detector(const std::shared_ptr<cv_yolox::YoloX> &model)
{
    auto detector = std::make_shared<RedoxiExamples::PersonBodyDetector>();
    RedoxiExamples::PersonBodyDetectorConfig config;
    config.model_yolox = model;
    detector->init(config);
    return detector;
}
