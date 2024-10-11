#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/detection/TrackTarget.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API PathStateChange
{
  public:
    int state_before = TrackPathStateBitmask::None;
    int state_after = TrackPathStateBitmask::None;
};
/**
 * not use, now using event handler
 */
class REDOXI_TRACK_API TrackingDecision
{
  public:
    TrackingDecision()
    {
    }
    virtual ~TrackingDecision()
    {
    }

  public:
    std::map<int, int> detection_index2path_id;
    std::vector<int> closed_id;
    std::map<int, RedoxiTrack::TrackTargetPtr> targets;
    // std::map<int, PathStateChange> path_id_state_change;
};
} // namespace RedoxiTrack
