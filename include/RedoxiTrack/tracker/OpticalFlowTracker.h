#pragma once

#include "RedoxiTrack/tracker/OpencvOpticalFlow.h"
#include "RedoxiTrack/tracker/OpticalFlowMotionPrediction.h"
#include "RedoxiTrack/tracker/OpticalTrackerParam.h"
#include "RedoxiTrack/tracker/TrackerBase.h"
#include "RedoxiTrack/tracker/TrackerParam.h"

// #include "NNIEOpticalFlow.h"
#include "RedoxiTrack/utils/utility_functions.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API OpticalFlowTrackerTrackingSate : public TrackerTrackingState
{
  public:
    virtual ~OpticalFlowTrackerTrackingSate()
    {
    }
    cv::Mat m_prev_img;
};
using OpticalFlowTrackerTrackingSatePtr = std::shared_ptr<OpticalFlowTrackerTrackingSate>;

class REDOXI_TRACK_API OpticalFlowTracker : public TrackerBase
{
  public:
    OpticalFlowTracker()
    {
        m_motion_predict = std::make_shared<OpencvOpticalFlow>();
    }

    void init(const TrackerParam &param) override;

    /**
     * set motion prediction, using optical flow motion predictor
     * @param motion_predict
     */
    void set_motion_prediction(const OpticalFlowMotionPredictionPtr motion_predict)
    {
        m_motion_predict = motion_predict;
    }

    const TrackerParam *get_tracker_param() const override;

    void set_tracker_param(const TrackerParam &param) override;

    /**
     * initialize opticalflow image and bbox
     * @param img
     * @param detections
     * @param frame_number
     */
    void begin_track(const cv::Mat &img,
                     const std::vector<DetectionPtr> &detections,
                     int frame_number) override;

    void finish_track() override;

    /**
     * using sort to implement tracking, todo debug sort track
     * @param img
     * @param detections
     * @param frame_number
     */
    void
        track(const cv::Mat &img, const std::vector<DetectionPtr> &detections, int frame_number) override;

    /**
     * using opticalflow to predict bbox
     * @param img
     * @param frame_number
     */
    void
        track(const cv::Mat &img, int frame_number) override;

    TrackTargetPtr get_open_target(int path_id) const override;

    void add_target(const TrackTargetPtr &target) override;
    TrackTargetPtr create_target(const DetectionPtr &det, int frame_number) override;

    void delete_target(int path_id) override;

    void delete_all_targets() override;

    void add_event_handler(const TrackingEventHandlerPtr &handler) override;

    void remove_event_handler(const TrackingEventHandlerPtr &handler) override;

  protected:
    virtual TrackerTrackingStatePtr _tracking_state_create() override;
    virtual void _tracking_state_fill(TrackerTrackingState &state) override;
    virtual void _tracking_state_recover(const TrackerTrackingState &state) override;
    /**
     * given new image, compute the moved bboxes of the track targets, if bbox out of image, use its origin bbox
     * @param img
     * @param frame_number
     * @param id2target
     * @return
     */
    std::map<int, BBOX> _advance_bbox_with_motion_prediction(const cv::Mat &img, int frame_number, const std::map<int, TrackTargetPtr> &id2target);

    /**
     * motion predict, target's m_bbox is be set to predicted bbox
     * @param img
     * @param frame_number
     * @param id2target
     */
    void _motion_predict(const cv::Mat &img, int frame_number, const std::map<int, TrackTargetPtr> &id2target);
    void _delete_target(std::map<int, TrackTargetPtr> &id2target, const int id);

    OpticalFlowMotionPredictionPtr m_motion_predict;
};
using OpticalFlowTrackerPtr = std::shared_ptr<OpticalFlowTracker>;

} // namespace RedoxiTrack
