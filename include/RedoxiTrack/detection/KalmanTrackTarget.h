//
// Created by wangjing on 1/11/22.
//
#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/detection/TrackTarget.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"


namespace RedoxiTrack
{
class REDOXI_TRACK_API KalmanTrackTarget : public TrackTarget
{
  public:
    cv::KalmanFilter &get_kf()
    {
        return m_kf;
    }
    const cv::KalmanFilter &get_kf() const
    {
        return m_kf;
    }
    void set_kf(const cv::KalmanFilter &input_kf)
    {
        m_kf = input_kf;
    }
    virtual DetectionPtr clone() const override;
    virtual void copy_to(Detection &target) const override;

    void print() override;

  protected:
    cv::KalmanFilter m_kf;

  public:
    /**
     * if target has been predict, set m_can_be_update=true, means it can be update
     * kalman target can be predicted multi times, update match the last predict state(it must be predicted before update)
     */
    bool m_can_be_update = false;
};
using KalmanTrackTargetPtr = std::shared_ptr<KalmanTrackTarget>;
} // namespace RedoxiTrack
