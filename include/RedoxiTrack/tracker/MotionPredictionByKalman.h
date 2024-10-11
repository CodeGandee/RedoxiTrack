//
// Created by 18200 on 2022/1/18.
//

#pragma once
#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

namespace RedoxiTrack
{
class REDOXI_TRACK_API MotionPredictionByKalman;
using MotionPredictionByKalmanPtr = std::shared_ptr<MotionPredictionByKalman>;

class REDOXI_TRACK_API MotionPredictionByKalman
{
  public:
    virtual ~MotionPredictionByKalman()
    {
    }

    /**
     * init kalmanFilter, set state and covariance
     * @param kf
     * @param bbox
     */
    virtual void init(cv::KalmanFilter &kf, const BBOX &bbox) = 0;

    /**
     * kalmanFilter predict, x_t = A*x_{t-1}
     * @param kf
     * @param output_bbox
     * @param delta_frame_number
     * @param flag
     */
    virtual void predict(cv::KalmanFilter &kf, BBOX &output_bbox, int delta_frame_number, const bool flag = false) = 0;

    /**
     * update kalmanFilter, ^x_t = Ax_{t-1} + k(z_t - CAx_{t-1})
     * @param kf
     * @param bbox
     */
    virtual void update(cv::KalmanFilter &kf, const BBOX &bbox) = 0;

    /**
     * get updated bbox from kalmanFilter's statePost
     * @param kf
     * @param output_bbox
     */
    virtual void get_bbox_state(cv::KalmanFilter &kf, BBOX &output_bbox) = 0;

    /**
     * get kalmanFilter's measurement mean and covariance, c*x_t and cP_tc'+measurementNoiseCov
     * @param kf
     * @param output_mean
     * @param output_covariance
     */
    virtual void project_state2measurement(cv::KalmanFilter &kf, cv::Mat &output_mean,
                                           cv::Mat &output_covariance) const = 0;

    virtual MotionPredictionByKalmanPtr clone() const = 0;
    virtual void copy_to(MotionPredictionByKalman &to) const
    {
        to.m_stateNum = m_stateNum;
        to.m_measureNum = m_measureNum;
    };

  protected:
    int m_stateNum;
    int m_measureNum;
};

} // namespace RedoxiTrack
