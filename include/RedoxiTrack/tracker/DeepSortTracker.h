//
// Created by wangjing on 1/11/22.
//

#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/detection/DeepSortTrackTarget.h"
#include "RedoxiTrack/tracker/DetectionTraits.h"
#include "RedoxiTrack/tracker/KalmanTracker.h"
#include "RedoxiTrack/tracker/OpticalFlowTracker.h"
#include "RedoxiTrack/tracker/TrackerBase.h"
#include "RedoxiTrack/tracker/TrackingEventHandler.h"

namespace RedoxiTrack
{

class REDOXI_TRACK_API DeepSortTracker : public TrackerBase
{

  protected:
    class DefaultDetectionTraits : public FeatureBasedDetTraits
    {
      public:
        DefaultDetectionTraits(DeepSortTracker *p);
        FeatureTraitsPtr get_feature_traits() const override;

      public:
      protected:
        DeepSortTracker *m_tracker = nullptr;
    };

  public:
    class REDOXI_TRACK_API OpticalFlowEventHandler
        : public TrackingEventHandler
    {
      public:
        int evt_target_association_after(
            TrackerBase *sender,
            const TrackingEvent::TargetAssociation &evt_data) override;

        int evt_target_created_after(
            TrackerBase *sender,
            const TrackingEvent::TargetAssociation &evt_data) override;

        int evt_target_closed_after(
            TrackerBase *sender,
            const TrackingEvent::TargetClosed &evt_data) override;

        int evt_target_motion_predict_after(
            TrackerBase *sender,
            const TrackingEvent::TargetMotionPredict &evt_data) override;

        virtual void clear()
        {
            m_det2target_create.clear();
            m_det2target_assiciate.clear();
            m_target_close.clear();
            m_target_motion_predict.clear();
        }

      public:
        // target creation
        std::map<DetectionPtr, TrackTargetPtr> m_det2target_create;
        std::map<DetectionPtr, TrackTargetPtr> m_det2target_assiciate;
        std::vector<TrackTargetPtr> m_target_close;
        std::vector<TrackTargetPtr> m_target_motion_predict;
    };
    class REDOXI_TRACK_API KalmanEventHandler : public TrackingEventHandler
    {
      public:
        int evt_target_association_after(
            TrackerBase *sender,
            const TrackingEvent::TargetAssociation &evt_data) override;

        int evt_target_created_after(
            TrackerBase *sender,
            const TrackingEvent::TargetAssociation &evt_data) override;

        int evt_target_motion_predict_after(
            TrackerBase *sender,
            const TrackingEvent::TargetMotionPredict &evt_data) override;

        virtual void clear()
        {
            m_det2target_create.clear();
            m_det2target_assiciate.clear();
            m_target_motion_predict.clear();
        }

      public:
        // target creation
        std::map<DetectionPtr, TrackTargetPtr> m_det2target_create;
        std::map<DetectionPtr, TrackTargetPtr> m_det2target_assiciate;
        std::vector<TrackTargetPtr> m_target_motion_predict;
    };

  public:
    void init(const TrackerParam &param) override;

    OpticalFlowTrackerPtr get_optical_flow_tracker() const
    {
        return m_optical_flow_tracker;
    }
    KalmanTrackerPtr get_kalman_tracker() const
    {
        return m_kalman_tracker;
    }

    const TrackerParam *get_tracker_param() const override;

    void set_tracker_param(const TrackerParam &param) override;

    void begin_track(const cv::Mat &img,
                     const std::vector<DetectionPtr> &detections,
                     int frame_number) override;
    void finish_track() override;

    void track(const cv::Mat &img, const std::vector<DetectionPtr> &detections,
               int frame_number) override;

    void track(const cv::Mat &img, int frame_number) override;

    virtual void push_tracking_state() override;

    virtual void pop_tracking_state(bool apply = true) override;

    TrackTargetPtr get_open_target(int path_id) const override;

    void add_target(const TrackTargetPtr &target) override;

    TrackTargetPtr create_target(const DetectionPtr &det,
                                 int frame_number) override;
    TrackTargetPtr create_target(const DetectionPtr &det, int frame_number,
                                 const TrackTargetPtr &kalman_target,
                                 const TrackTargetPtr &optical_target);

    void delete_target(int path_id) override;

    void delete_all_targets() override;

    void add_event_handler(const TrackingEventHandlerPtr &handler) override;

    void remove_event_handler(const TrackingEventHandlerPtr &handler) override;

    /**
     * set detection comparision handler which compute distance of two
     * detections, default: (1 - cos(a.feature, b.feature))/2.0
     * @param hander
     */
    void set_detection_comparision(const DetectionTraitsPtr &hander);
    /**
     * reset detection comparision, reset to default: cosine (1 - cos(a.feature,
     * b.feature))/2.0
     */
    void reset_detection_comparision();

    FeatureTraitsPtr get_feature_traits();
    void set_feature_traits(const FeatureTraitsPtr &p);

  protected:
    void _update_features(DeepSortTrackTargetPtr &target,
                          const fVECTOR &features);

    void _bbox2xyah(const BBOX &bbox, cv::Mat &output);

    void
        _match_maha_distance(const std::vector<DetectionPtr> &sources,
                             const std::vector<TrackTargetPtr> &targets,
                             std::vector<std::pair<int, int>> &output_matched_pair,
                             std::vector<int> &output_unmatched_source,
                             std::vector<int> &output_unmatched_target);

    void
        _match_iou_distance(const std::vector<DetectionPtr> &sources,
                            const std::vector<TrackTargetPtr> &targets,
                            std::vector<std::pair<int, int>> &output_matched_pair,
                            std::vector<int> &output_unmatched_source,
                            std::vector<int> &output_unmatched_target);

    void _remove_targets(const int frame_number);

    void _update_target(TrackTargetPtr &deepsort_target_ptr,
                        const DetectionPtr &det, const int &frame_number);
    /**
     * 将光流的预测作为kalman滤波器的观测值，获取更新后的kalman滤波器状态作为跟踪对象的运动预测；
     * 同时改变kalman的预测状态，而又不改变kalman的update状态。
     * @param img
     * @param id2target
     * @param frame_number
     */
    void _motion_predict(const cv::Mat &img,
                         std::map<int, TrackTargetPtr> &id2target,
                         int frame_number);

  protected:
    OpticalFlowTrackerPtr m_optical_flow_tracker;
    KalmanTrackerPtr m_kalman_tracker;
    std::shared_ptr<OpticalFlowEventHandler> m_optical_flow_handler;
    std::shared_ptr<KalmanEventHandler> m_kalman_handler;

    DetectionTraitsPtr m_detection_comparision;
    FeatureTraitsPtr m_feature_traits;
};
using DeepSortTrackerPtr = std::shared_ptr<DeepSortTracker>;
} // namespace RedoxiTrack
