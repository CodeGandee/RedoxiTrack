//
// Created by cx on 2025/1/6.
//

#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/detection/SimpleSortTrackTarget.h"
#include "RedoxiTrack/tracker/DetectionTraits.h"
#include "RedoxiTrack/tracker/KalmanTracker.h"
#include "RedoxiTrack/tracker/TrackerBase.h"
#include "RedoxiTrack/tracker/TrackingEventHandler.h"

namespace RedoxiTrack
{

class REDOXI_TRACK_API SimpleSortTracker : public TrackerBase
{

  protected:
    class DefaultDetectionTraits : public FeatureBasedDetTraits
    {
      public:
        DefaultDetectionTraits(SimpleSortTracker *p);
        FeatureTraitsPtr get_feature_traits() const override;

      public:
      protected:
        SimpleSortTracker *m_tracker = nullptr;
    };

  public:
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
                                 const TrackTargetPtr &kalman_target);

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
    void _update_features(SimpleSortTrackTargetPtr &target,
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

  protected:
    KalmanTrackerPtr m_kalman_tracker;
    std::shared_ptr<KalmanEventHandler> m_kalman_handler;

    DetectionTraitsPtr m_detection_comparision;
    FeatureTraitsPtr m_feature_traits;
};
using SimpleSortTrackerPtr = std::shared_ptr<SimpleSortTracker>;
} // namespace RedoxiTrack
