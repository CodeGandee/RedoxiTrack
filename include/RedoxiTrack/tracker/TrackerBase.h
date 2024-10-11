#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include <set>

#include "RedoxiTrack/detection/Detection.h"
#include "RedoxiTrack/detection/TrackTarget.h"
#include "RedoxiTrack/tracker/TrackerParam.h"
#include "RedoxiTrack/tracker/TrackingEventHandler.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API TrackerTrackingState
{
  public:
    virtual ~TrackerTrackingState()
    {
    }
    int m_frame_number;
    /**
     * save target's address
     * tracker's m_id2target can be add or delete directly, it can be recover
     * from this
     */
    std::map<int, TrackTargetPtr> m_id2target;
    /**
     * save target's state, it's new object, has different address
     */
    std::map<int, TrackTargetPtr> m_id2target_clone;
};
using TrackerTrackingStatePtr = std::shared_ptr<TrackerTrackingState>;

class REDOXI_TRACK_API TrackerBase
{
  public:
    TrackerBase();

    virtual ~TrackerBase();

  public:
    /**
     * tracker must init
     * @param param
     */
    virtual void init(const TrackerParam &param) = 0;
    virtual const TrackerParam *get_tracker_param() const
    {
        return m_param.get();
    }

    virtual void set_tracker_param(const TrackerParam &param) = 0;

    /**
     * first track
     * @param img
     * @param detections
     * @param frame_number
     */
    virtual void begin_track(const cv::Mat &img,
                             const std::vector<DetectionPtr> &detections,
                             int frame_number) = 0;
    /**
     * finish track, close all targets
     */
    virtual void finish_track() = 0;
    /**
     * save current tracking state
     */
    virtual void push_tracking_state();

    /**
     * recover previous tracking state
     */
    virtual void pop_tracking_state(bool apply = true);

    /**
     * track to future frames based on detections
     * @param img
     * @param detections
     * @param frame_number
     */
    virtual void track(const cv::Mat &img,
                       const std::vector<DetectionPtr> &detections,
                       int frame_number) = 0;

    /**
     * track to future based on motion prediction
     * @param img
     * @param frame_number
     */
    virtual void track(const cv::Mat &img, int frame_number) = 0;

    /**
     * get all targets still being tracked
     * @return
     */
    virtual const std::map<int, TrackTargetPtr> &
        get_all_open_targets() const = 0;

    /**
     * return NULL if target is not found
     * @param path_id
     * @return
     */
    virtual TrackTargetPtr get_open_target(int path_id) const;

    /**
     * add a target to m_id2target
     * the target will then be managed by tracker
     * @param target
     */
    virtual void add_target(const TrackTargetPtr &target) = 0;

    /**
     * create a new track target by det
     * @param det
     * @param frame_number
     * @return
     */
    virtual TrackTargetPtr create_target(const DetectionPtr &det,
                                         int frame_number) = 0;

    /**
     * delete target from m_id2target
     * @param path_id
     */
    virtual void delete_target(int path_id) = 0;

    virtual void delete_all_targets() = 0;

    virtual void add_event_handler(const TrackingEventHandlerPtr &handler) = 0;

    virtual void
        remove_event_handler(const TrackingEventHandlerPtr &handler) = 0;

    virtual const std::set<TrackingEventHandlerPtr> &get_event_handlers() const
    {
        return m_event_handlers;
    }

    /**
     * reset tracking state, call init()
     */
    virtual void reset_tracking_state();

    int get_current_frame_number()
    {
        return m_frame_number;
    }

    const std::map<int, TrackTargetPtr> &get_all_targets() const
    {
        return m_id2target;
    }

  protected:
    /**
     * create a new tracking state
     * @return
     */
    virtual TrackerTrackingStatePtr _tracking_state_create();

    /**
     * fill state content by current tracking state
     * @param state
     */
    virtual void _tracking_state_fill(TrackerTrackingState &state);

    /**
     * recover previous tracking state by param state
     * @param state
     */
    virtual void _tracking_state_recover(const TrackerTrackingState &state);

    void _update_frame_number(int frame_number)
    {
        assert(m_frame_number <= frame_number);
        m_frame_number = frame_number;
    }

    int _generate_path_id()
    {
        // FIXME: why cannot compile this?
        // assert(m_id2target.size() < std::numeric_limits<size_t>::max());
        int x = 0;
        while (true) {
            x = m_path_id_for_generate_unique_id++;
            if (m_id2target.find(x) == m_id2target.end())
                break;
        }
        return x;
    }

    /**
     * m_frame_number -1 used for judge if its first created
     */
    int m_frame_number = INIT_TRACKING_FRAME;
    int m_path_id_for_generate_unique_id = 1;

    /**
     * tracker tracks all targets
     */
    std::map<int, TrackTargetPtr> m_id2target;
    /**
     * tracker saves previous tracking state
     */
    std::vector<TrackerTrackingStatePtr> m_state_stack;

    TrackerParamPtr m_param;

    std::set<TrackingEventHandlerPtr> m_event_handlers;
};

using TrackerBasePtr = std::shared_ptr<TrackerBase>;
} // namespace RedoxiTrack
