#include "example_person_detector.h"

namespace RedoxiExamples
{

void PersonBodyDetector::init(const PersonBodyDetectorConfig &config)
{
    if (!config.model_yolox)
        throw std::runtime_error("Model not set");

    this->m_model_yolox = config.model_yolox;
}

void PersonBodyDetector::set_model(
    const std::shared_ptr<cv_yolox::YoloX> &model)
{
    this->m_model_yolox = model;
}

PersonBodyDetector::DetectionList
    PersonBodyDetector::detect(const cv::Mat &frame)
{
    return this->_detect_by_yolox(frame);
}

PersonBodyDetector::DetectionList
    PersonBodyDetector::_detect_by_yolox(const cv::Mat &frame)
{
    DetectionList detections;
    if (!this->m_model_yolox)
        throw std::runtime_error("Model not set");

    auto net = this->m_model_yolox;
    // resize frame
    auto _preproc = cv_yolox::letterBox(frame);
    auto _frame = std::get<0>(_preproc);

    // scale factor, from original to resized
    auto _scale = std::get<1>(_preproc);

    // each row is a bbox and body landmarks
    auto model_output = net->infer(_frame);

    // convert to DetectionList
    for (int i = 0; i < model_output.rows; i++) {
        auto row = model_output.row(i);

        // is this a person detection?
        int class_id = int(row.at<float>(5));
        bool is_person =
            cv_yolox::labelYolox[class_id] == "person" ? true : false;

        if (!is_person)
            continue;

        auto det = std::make_shared<Detection>();

        // bbox
        auto bbox = det->get_bbox();
        bbox.x = row.at<float>(0) / _scale;
        bbox.y = row.at<float>(1) / _scale;
        bbox.width = row.at<float>(2) / _scale - bbox.x;
        bbox.height = row.at<float>(3) / _scale - bbox.y;
        det->set_bbox(bbox);

        // confidence and type
        det->set_confidence(row.at<float>(4));
        det->set_type(RedoxiTrack::DetectionTypes::PersonBody);

        detections.push_back(det);
    }

    return detections;
}

} // namespace RedoxiExamples