//
// Created by wangjing on 12/31/21.
//

#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/tracker/MotionPredictionByImageKeypoint.h"
#include "RedoxiTrack/utils/utility_functions.h"
#include "opencv2/opencv.hpp"


namespace RedoxiTrack
{
class REDOXI_TRACK_API OpticalFlowMotionPrediction : public MotionPredictionByImageKeypoint
{
  public:
    class REDOXI_TRACK_API Result : public MotionPredictionByImageKeypoint::Result
    {
      public:
        std::vector<uint8_t> keypoints_valid;
    };

  public:
    /**
     * set prev image after optical flow
     * @param img gray
     */
    virtual void set_prev_image(const cv::Mat &img) = 0;
    virtual void set_current_image(const cv::Mat &img) = 0;
    virtual void set_prev_image_by_current() = 0;
    virtual cv::Mat get_prev_image() = 0;

    using MotionPredictionByImageKeypoint::predict_keypoint_location;
    virtual void predict_keypoint_location(const cv::Mat &cur, const std::vector<POINT> &points,
                                           MotionPredictionByImageKeypoint::Result &output) const = 0;
};
using OpticalFlowMotionPredictionPtr = std::shared_ptr<OpticalFlowMotionPrediction>;
} // namespace RedoxiTrack
