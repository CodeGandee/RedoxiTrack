//
// Created by cx on 2025/1/6.
//

#pragma once
#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/tracker/MotionPredictionByKalman.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API SimpleSortMotionPrediction : public MotionPredictionByKalman
{
  public:
    /**
     * init kalmanFilter's state
     * @param kf
     * @param bbox
     */
    void init(cv::KalmanFilter &kf, const BBOX &bbox) override;

    /**
     * kalmanFilter predict, x_t = A*x_{t-1}
     * @param kf
     * @param output_bbox
     * @param delta_frame_number
     * @param flag
     */
    void predict(cv::KalmanFilter &kf, BBOX &output_bbox, int delta_frame_number, const bool flag = false) override;

    /**
     * update kalmanFilter, ^x_t = Ax_{t-1} + k(z_t - CAx_{t-1})
     * @param kf
     * @param bbox
     */
    void update(cv::KalmanFilter &kf, const BBOX &bbox) override;

    /**
     * get updated bbox from kalmanFilter's statePost
     * @param kf
     * @param output_bbox
     */
    void get_bbox_state(cv::KalmanFilter &kf, BBOX &output_bbox) override;

    /**
     * get kalmanFilter's measurement mean and covariance, c*x_t and cP_tc'+measurementNoiseCov
     * @param kf
     * @param output_mean
     * @param output_covariance
     */
    void project_state2measurement(cv::KalmanFilter &kf, cv::Mat &output_mean,
                                   cv::Mat &output_covariance) const override;

    MotionPredictionByKalmanPtr clone() const override;

    void copy_to(MotionPredictionByKalman &to) const override;

  protected:
    static void _bbox2xyah(const BBOX &bbox, std::vector<float> &output);
    static void _xyah2bbox(float x, float y, float a, float h, BBOX &output_bbox);

  protected:
    /**
     * weight for covariance element
     */
    float m_std_weight_position = 1.0 / 20.0;
    float m_std_weight_velocity = 1.0 / 160.0;
    cv::Mat m_update_measurement;
};
using SimpleSortMotionPredictionPtr = std::shared_ptr<SimpleSortMotionPrediction>;
} // namespace RedoxiTrack
