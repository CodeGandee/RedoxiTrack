//
// Created by 18200 on 2022/2/21.
//

#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/tracker/OpticalFlowMotionPrediction.h"
#include "RedoxiTrack/tracker/OpticalTrackerParam.h"
namespace RedoxiTrack
{
class REDOXI_TRACK_API OpencvOpticalFlow : public OpticalFlowMotionPrediction
{
  public:
    /**
     * set prev image after optical flow
     * @param img
     */
    void set_prev_image(const cv::Mat &img) override;
    cv::Mat get_prev_image() override;

    void set_current_image(const cv::Mat &img) override;

    void set_prev_image_by_current() override;

    /**
     * predict points position in cur image, using optical flow between m_pre_img and m_cur_img;
     * @param points
     * @param output
     */
    void predict_keypoint_location(const std::vector<POINT> &points,
                                   MotionPredictionByImageKeypoint::Result &output) const override;

    /**
     * predict points position in cur image, using optical flow between m_pre_img and cur;
     * @param cur current image
     * @param points
     * @param output
     */
    void predict_keypoint_location(const cv::Mat &cur, const vector<POINT> &points,
                                   MotionPredictionByImageKeypoint::Result &output) const override;

  protected:
    OpticalTrackerParam m_param;
    cv::Mat m_pre_img;
    cv::Mat m_cur_img;
};
} // namespace RedoxiTrack
