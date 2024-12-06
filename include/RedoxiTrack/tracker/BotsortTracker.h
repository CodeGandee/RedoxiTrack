//
// Created by 001730 chengxiao on 12/13/22.
//
#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/detection/BotsortTrackTarget.h"
#include "RedoxiTrack/detection/KalmanTrackTarget.h"
#include "RedoxiTrack/tracker/BotsortKalmanTracker.h"
#include "RedoxiTrack/tracker/BotsortTrackerParam.h"
#include "RedoxiTrack/tracker/DetectionTraits.h"
#include "RedoxiTrack/tracker/OpticalFlowTracker.h"
#include "RedoxiTrack/tracker/TrackerBase.h"
#include "RedoxiTrack/tracker/TrackingEventHandler.h"
#include "RedoxiTrack/utils/CosineFeature.h"
#include "opencv2/core/core_c.h"
// #include "opencv2/highgui.hpp"



namespace RedoxiTrack
{

class REDOXI_TRACK_API BotsortTracker : public TrackerBase
{

  protected:
    class DefaultDetectionTraits : public FeatureBasedDetTraits
    {
      public:
        DefaultDetectionTraits(BotsortTracker *p);
        FeatureTraitsPtr get_feature_traits() const override;

      public:
      protected:
        BotsortTracker *m_tracker = nullptr;
    };

  public:
    class REDOXI_TRACK_API OpticalFlowEventHandler : public TrackingEventHandler
    {
      public:
        int evt_target_association_after(TrackerBase *sender,
                                         const TrackingEvent::TargetAssociation &evt_data) override;

        int evt_target_created_after(TrackerBase *sender, const TrackingEvent::TargetAssociation &evt_data) override;

        int evt_target_closed_after(TrackerBase *sender, const TrackingEvent::TargetClosed &evt_data) override;

        int evt_target_motion_predict_after(TrackerBase *sender, const TrackingEvent::TargetMotionPredict &evt_data) override;

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
        int evt_target_association_after(TrackerBase *sender,
                                         const TrackingEvent::TargetAssociation &evt_data) override;

        int evt_target_created_after(TrackerBase *sender, const TrackingEvent::TargetAssociation &evt_data) override;

        int evt_target_motion_predict_after(TrackerBase *sender, const TrackingEvent::TargetMotionPredict &evt_data) override;

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
    BotsortKalmanTrackerPtr get_kalman_tracker() const
    {
        return m_kalman_tracker;
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

    virtual void push_tracking_state() override;

    virtual void pop_tracking_state(bool apply = true) override;

    const std::map<int, TrackTargetPtr> &get_all_open_targets() const override;

    TrackTargetPtr get_open_target(int path_id) const override;

    void add_target(const TrackTargetPtr &target) override;

    TrackTargetPtr create_target(const DetectionPtr &det, int frame_number) override;
    TrackTargetPtr create_target(const DetectionPtr &det, int frame_number,
                                 const TrackTargetPtr &kalman_target, const TrackTargetPtr &optical_target);

    void delete_target(int path_id) override;

    void delete_all_targets() override;

    void add_event_handler(const TrackingEventHandlerPtr &handler) override;

    void remove_event_handler(const TrackingEventHandlerPtr &handler) override;

    /**
     * set detection comparision handler which compute distance of two detections, default: (1 - cos(a.feature, b.feature))/2.0
     * @param hander
     */
    void set_detection_comparision(const DetectionTraitsPtr &hander);
    /**
     * reset detection comparision, reset to default: cosine (1 - cos(a.feature, b.feature))/2.0
     */
    void reset_detection_comparision();

    FeatureTraitsPtr get_feature_traits();
    void set_feature_traits(const FeatureTraitsPtr &p);

  protected:
    void _update_features(BotsortTrackTargetPtr &target, const fVECTOR &features);

    void _bbox2xcycwh(const BBOX &bbox, cv::Mat &output);

    void _match_maha_distance(const std::vector<DetectionPtr> &sources,
                              const std::vector<TrackTargetPtr> &targets,
                              const float match_thresh,
                              std::vector<std::pair<int, int>> &output_matched_pair,
                              std::vector<int> &output_unmatched_source,
                              std::vector<int> &output_unmatched_target);

    void _match_iou_distance(const std::vector<DetectionPtr> &sources,
                             const std::vector<TrackTargetPtr> &targets,
                             const float match_thresh,
                             std::vector<std::pair<int, int>> &output_matched_pair,
                             std::vector<int> &output_unmatched_source,
                             std::vector<int> &output_unmatched_target);
    void _remove_duplicate_targets();
    void _remove_targets(vector<TrackTargetPtr> &removed);

    void _update_target(TrackTargetPtr &botsort_target_ptr, const DetectionPtr &det, const int &frame_number,
                        bool add_refind, std::vector<TrackTargetPtr> &activated, std::vector<TrackTargetPtr> &refind);
    /**
     * 将光流的预测作为kalman滤波器的观测值，获取更新后的kalman滤波器状态作为跟踪对象的运动预测；
     * 同时改变kalman的预测状态，而又不改变kalman的update状态。
     * @param img
     * @param id2target
     * @param frame_number
     */
    void _motion_predict(const cv::Mat &img, std::vector<TrackTargetPtr> &target_pool, int frame_number);

    void _fuse_score(std::vector<std::vector<float>> &dist_matrix_iou,
                     const std::vector<DetectionPtr> &detections);


  protected:
    std::map<int, TrackTargetPtr> m_tracked_targets;
    std::map<int, TrackTargetPtr> m_lost_targets;
    std::map<int, TrackTargetPtr> m_removed_targets;

    OpticalFlowTrackerPtr m_optical_flow_tracker;
    BotsortKalmanTrackerPtr m_kalman_tracker;
    std::shared_ptr<OpticalFlowEventHandler> m_optical_flow_handler;
    std::shared_ptr<KalmanEventHandler> m_kalman_handler;

    DetectionTraitsPtr m_detection_comparision;
    FeatureTraitsPtr m_feature_traits;
};
using BotsortTrackerPtr = std::shared_ptr<BotsortTracker>;
} // namespace RedoxiTrack
