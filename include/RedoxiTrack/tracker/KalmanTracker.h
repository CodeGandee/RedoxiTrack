//
// Created by wangjing on 1/11/22.
//

#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/detection/KalmanTrackTarget.h"
#include "RedoxiTrack/tracker/DeepSortMotionPrediction.h"
#include "RedoxiTrack/tracker/MotionPredictionByKalman.h"
#include "RedoxiTrack/tracker/TrackerBase.h"


namespace RedoxiTrack
{
class REDOXI_TRACK_API KalmanTrackingState : public TrackerTrackingState
{
  public:
    MotionPredictionByKalmanPtr m_motion_predict;
};

/**
 * KalmanTracker 可以被多次无检测track(相当于predict多次)，
 * 而update_kalman仅对应最后一次track的状态，并且在进行update_kalman之前一定要被track一次。
 * 例子：
 *      1、改变跟踪对象的状态
 *          KalmanTracker->track(img, frame_number);
 *          KalmanTracker->update_kalman(kalman_target, bbox);
 *      2、不改变跟踪对象的状态
 *          KalmanTracker->push_tracking_state();
 *          KalmanTracker->track(img, frame_number);
 *          KalmanTracker->update_kalman(kalman_target, bbox);
 *          KalmanTracker->pop_tracking_state();
 *      3、改变跟踪对象的一部分状态
 *          KalmanTracker->track(img, frame_number);
 *          KalmanTracker->push_tracking_state();
 *          KalmanTracker->update_kalman(kalman_target, bbox);
 *          KalmanTracker->pop_tracking_state();
 */
class REDOXI_TRACK_API KalmanTracker : public TrackerBase
{
  public:
    KalmanTracker()
    {
        m_motion_predict = std::make_shared<DeepSortMotionPrediction>();
    }

    void init(const TrackerParam &param) override;

    /**
     * set motion prediction, responsible for kalman predict and update
     * @param motion_predict
     */
    void set_motion_prediction(const MotionPredictionByKalmanPtr motion_predict)
    {
        m_motion_predict = motion_predict;
    }

    const TrackerParam *get_tracker_param() const override;

    void set_tracker_param(const TrackerParam &param) override;
    void begin_track(const cv::Mat &img,
                     const std::vector<DetectionPtr> &detections,
                     int frame_number) override;
    void finish_track() override;

    void track(const cv::Mat &img, const std::vector<DetectionPtr> &detections, int frame_number) override;

    void
        track(const cv::Mat &img, int frame_number) override;

    TrackTargetPtr get_open_target(int path_id) const override;

    void add_target(const TrackTargetPtr &target) override;

    TrackTargetPtr create_target(const DetectionPtr &det, int frame_number) override;

    void delete_target(int path_id) override;

    void delete_all_targets() override;

    void add_event_handler(const TrackingEventHandlerPtr &handler) override;

    void remove_event_handler(const TrackingEventHandlerPtr &handler) override;

    /**
     * update target's kalmanFilter's state
     * @param target
     * @param bbox
     */
    void update_kalman(TrackTargetPtr &target, const BBOX &bbox);


    MotionPredictionByKalmanPtr get_motion_prediction() const
    {
        return m_motion_predict;
    }

  protected:
    virtual TrackerTrackingStatePtr _tracking_state_create() override;
    virtual void _tracking_state_fill(TrackerTrackingState &state) override;
    virtual void _tracking_state_recover(const TrackerTrackingState &state) override;
    /**
     * motion predict using kalmanFilter
     * @param delta_frame_number
     * @param target
     * @param notify_event_handler
     */
    void _motion_predict(int delta_frame_number, TrackTargetPtr &target, bool notify_event_handler = false);
    /**
     * target kalmanFilter predict
     * @param target
     * @param delta_frame_number
     * @param output_bbox
     */
    void _kalman_predict(TrackTargetPtr &target, const int &delta_frame_number, BBOX &output_bbox);

  private:
    MotionPredictionByKalmanPtr m_motion_predict;
};
using KalmanTrackerPtr = std::shared_ptr<KalmanTracker>;
} // namespace RedoxiTrack
