//
// Created by 18200 on 2022/1/18.
//

#pragma once
#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/tracker/MotionPredictionByKalman.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API SortMotionPrediction : public MotionPredictionByKalman
{
  public:
    void init(cv::KalmanFilter &kf, const BBOX &bbox) override;

    void predict(cv::KalmanFilter &kf, BBOX &output_bbox, int delta_frame_number, const bool flag = false) override;

    void update(cv::KalmanFilter &kf, const BBOX &bbox) override;

    void get_bbox_state(cv::KalmanFilter &kf, BBOX &output_bbox) override;

    void project_state2measurement(cv::KalmanFilter &kf, cv::Mat &output_mean,
                                   cv::Mat &output_covariance) const override;

  protected:
    static void _get_rect_from_xysr(float cx, float cy, float s, float r, BBOX &output_bbox);

  protected:
    cv::Mat m_update_measurement;
};
using SortMotionPredictionPtr = std::shared_ptr<SortMotionPrediction>;
} // namespace RedoxiTrack
