//
// Created by 001730 chengxiao on 1/3/23.
//
#pragma once

#include "RedoxiTrack/detection/Detection.h"
#include "RedoxiTrack/detection/TrackTarget.h"
#include "RedoxiTrack/tracker/KalmanTracker.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API BotsortKalmanTracker : public KalmanTracker
{
  public:
    // 继承构造函数
    using KalmanTracker::track;
    using RedoxiTrack::KalmanTracker::KalmanTracker;

    void track(const cv::Mat &img,
               const std::vector<DetectionPtr> &targets,
               int frame_number) override;

    void track(const cv::Mat &img,
               const std::vector<TrackTargetPtr> &targets,
               int frame_number);

    void update_kalman(TrackTargetPtr &target, const BBOX &bbox, int delta_frame_number = 1);
};
using BotsortKalmanTrackerPtr = std::shared_ptr<BotsortKalmanTracker>;
} // namespace RedoxiTrack
