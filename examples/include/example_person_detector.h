#pragma once
#include "example_common.h"
#include "opencv_demo_yolox.h"
#include <RedoxiTrack/detection/SingleDetection.h>
#include <vector>

namespace RedoxiExamples
{

/** Configuration for the person body detector
 */
struct PersonBodyDetectorConfig {
    std::shared_ptr<cv_yolox::YoloX> model_yolox;
};
class PersonBodyDetector
{
  public:
    using Detection = RedoxiTrack::SingleDetection;
    using DetectionPtr = std::shared_ptr<Detection>;
    using DetectionList = std::vector<DetectionPtr>;

    PersonBodyDetector(){};
    virtual ~PersonBodyDetector(){};

    virtual void init(const PersonBodyDetectorConfig &config);

    /** Set the yolox model to be used for detection
     */
    virtual void set_model(const std::shared_ptr<cv_yolox::YoloX> &model);

    /** Detect people in the given frame
     * @param frame the input frame
     * @return a list of detected people
     */
    virtual DetectionList detect(const cv::Mat &frame);

  protected:
    virtual DetectionList _detect_by_yolox(const cv::Mat &frame);

  protected:
    std::shared_ptr<cv_yolox::YoloX> m_model_yolox;
};

}; // namespace RedoxiExamples
