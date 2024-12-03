//
// Created by sfj on 2022/4/29.
//

#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/detection/TrackTarget.h"
#include <functional>

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

/**
 * @brief This class is used to handle tracking events from internal callbacks,
 * you need to inherit this class to customize your own event handling logic.
 *
 * The TrackingEventHandler class provides virtual methods to handle various tracking events such as target
 * association, target creation, target closure, and target motion prediction. These methods can be overridden by derived
 * classes to implement custom event handling logic.
 */
class REDOXI_TRACK_API TrackingEventHandler
{
  public:
    using Ptr = std::shared_ptr<TrackingEventHandler>;
    using ConstPtr = std::shared_ptr<const TrackingEventHandler>;

    virtual ~TrackingEventHandler() = default;

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
};
using TrackingEventHandlerPtr = std::shared_ptr<TrackingEventHandler>;


/**
 * @brief This class is used to handle tracking events from external callbacks, without inheriting from TrackingEventHandler.
 *
 * The ExternalTrackingEventHandler class provides implementations for handling various tracking events such as target
 * association, target creation, target closure, and target motion prediction. These implementations can be customized
 * by setting the corresponding callback functions.
 */
class REDOXI_TRACK_API ExternalTrackingEventHandler : public TrackingEventHandler
{
  public:
    using TrackerBase_t = TrackerBase;
    using TargetAssociation_t = TrackingEvent::TargetAssociation;
    using TargetClosed_t = TrackingEvent::TargetClosed;
    using TargetMotionPredict_t = TrackingEvent::TargetMotionPredict;

    using Ptr = std::shared_ptr<ExternalTrackingEventHandler>;
    using ConstPtr = std::shared_ptr<const ExternalTrackingEventHandler>;

  public:
    int evt_target_association_before(TrackerBase_t *sender, const TargetAssociation_t &evt_data) override
    {
        if (on_target_association_before) {
            return on_target_association_before(sender, evt_data);
        }
        return 0;
    }

    int evt_target_association_after(TrackerBase_t *sender, const TargetAssociation_t &evt_data) override
    {
        if (on_target_association_after) {
            return on_target_association_after(sender, evt_data);
        }
        return 0;
    }

    int evt_target_created_before(TrackerBase_t *sender, const TargetAssociation_t &evt_data) override
    {
        if (on_target_created_before) {
            return on_target_created_before(sender, evt_data);
        }
        return 0;
    }

    int evt_target_created_after(TrackerBase_t *sender, const TargetAssociation_t &evt_data) override
    {
        if (on_target_created_after) {
            return on_target_created_after(sender, evt_data);
        }
        return 0;
    }

    int evt_target_closed_before(TrackerBase_t *sender, const TargetClosed_t &evt_data) override
    {
        if (on_target_closed_before) {
            return on_target_closed_before(sender, evt_data);
        }
        return 0;
    }

    int evt_target_closed_after(TrackerBase_t *sender, const TargetClosed_t &evt_data) override
    {
        if (on_target_closed_after) {
            return on_target_closed_after(sender, evt_data);
        }
        return 0;
    }

    int evt_target_motion_predict_before(TrackerBase_t *sender, const TargetMotionPredict_t &evt_data) override
    {
        if (on_target_motion_predict_before) {
            return on_target_motion_predict_before(sender, evt_data);
        }
        return 0;
    }

    int evt_target_motion_predict_after(TrackerBase_t *sender, const TargetMotionPredict_t &evt_data) override
    {
        if (on_target_motion_predict_after) {
            return on_target_motion_predict_after(sender, evt_data);
        }
        return 0;
    }

  public:
    using OnTargetAssociationCallback_t = std::function<int(TrackerBase_t *sender, const TargetAssociation_t &evt_data)>;
    using OnTargetClosedCallback_t = std::function<int(TrackerBase_t *sender, const TargetClosed_t &evt_data)>;
    using OnTargetMotionPredictCallback_t = std::function<int(TrackerBase_t *sender, const TargetMotionPredict_t &evt_data)>;

    OnTargetAssociationCallback_t on_target_association_before;
    OnTargetAssociationCallback_t on_target_association_after;
    OnTargetAssociationCallback_t on_target_created_before;
    OnTargetAssociationCallback_t on_target_created_after;
    OnTargetClosedCallback_t on_target_closed_before;
    OnTargetClosedCallback_t on_target_closed_after;
    OnTargetMotionPredictCallback_t on_target_motion_predict_before;
    OnTargetMotionPredictCallback_t on_target_motion_predict_after;
};
// using ExternalTrackingEventHandlerPtr = std::shared_ptr<ExternalTrackingEventHandler>;

} // namespace RedoxiTrack
