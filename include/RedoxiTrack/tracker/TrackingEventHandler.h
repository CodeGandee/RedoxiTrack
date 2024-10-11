//
// Created by sfj on 2022/4/29.
//

#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/detection/TrackTarget.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API TrackerBase;

namespace TrackingEvent
{
class REDOXI_TRACK_API TrackingEventData
{
  public:
    virtual ~TrackingEventData() = default;
};

class REDOXI_TRACK_API TargetAssociation : public TrackingEventData
{
  public:
    DetectionPtr m_detection;
    TrackTargetPtr m_target;
};

class REDOXI_TRACK_API TargetClosed : public TrackingEventData
{
  public:
    TrackTargetPtr m_target;
};

class REDOXI_TRACK_API TargetMotionPredict : public TrackingEventData
{
  public:
    TrackTargetPtr m_target;
};
} // namespace TrackingEvent

class REDOXI_TRACK_API TrackingEventHandler
{
  public:
    /**
     * Return EventHandlerResultTypes, None is not advice, and EventHandlerResultTypes is only used for
     * evt_target_association_before
     * @param sender tracker pointer who send target association message
     * @param evt_data association message, detPtr: targetPtr match
     * @return
     */
    virtual int evt_target_association_before(TrackerBase *sender, const TrackingEvent::TargetAssociation &evt_data)
    {
        return 0;
    };

    /**
     * Now return 0, is not used.
     * @param sender tracker pointer who send target association message
     * @param evt_data association message, detPtr: targetPtr match
     * @return
     */
    virtual int evt_target_association_after(TrackerBase *sender, const TrackingEvent::TargetAssociation &evt_data)
    {
        return 0;
    };

    /**
     * Now return 0, is not used.
     * @param sender tracker pointer who send target created message
     * @param evt_data created message, detPtr: targetPtr match
     * @return
     */
    virtual int evt_target_created_before(TrackerBase *sender, const TrackingEvent::TargetAssociation &evt_data)
    {
        return 0;
    };
    /**
     * Now return 0, is not used.
     * @param sender tracker pointer who send target created message
     * @param evt_data created message, detPtr: targetPtr match
     * @return
     */
    virtual int evt_target_created_after(TrackerBase *sender, const TrackingEvent::TargetAssociation &evt_data)
    {
        return 0;
    };
    /**
     * Now return 0, is not used.
     * @param sender tracker pointer who send target closed message
     * @param evt_data closed message, targetPtr
     * @return
     */
    virtual int evt_target_closed_before(TrackerBase *sender, const TrackingEvent::TargetClosed &evt_data)
    {
        return 0;
    };
    /**
     * Now return 0, is not used.
     * @param sender tracker pointer who send target closed message
     * @param evt_data closed message, targetPtr
     * @return
     */
    virtual int evt_target_closed_after(TrackerBase *sender, const TrackingEvent::TargetClosed &evt_data)
    {
        return 0;
    };
    /**
     * Now return 0, is not used.
     * @param sender tracker pointer who send target motion predict message
     * @param evt_data motion preidct message, targetPtr
     * @return
     */
    virtual int evt_target_motion_predict_before(TrackerBase *sender, const TrackingEvent::TargetMotionPredict &evt_data)
    {
        return 0;
    };
    /**
     * Now return 0, is not used.
     * @param sender tracker pointer who send target motion predict message
     * @param evt_data motion preidct message, targetPtr
     * @return
     */
    virtual int evt_target_motion_predict_after(TrackerBase *sender, const TrackingEvent::TargetMotionPredict &evt_data)
    {
        return 0;
    };
    virtual ~TrackingEventHandler() = default;
};
using TrackingEventHandlerPtr = std::shared_ptr<TrackingEventHandler>;
} // namespace RedoxiTrack
