//
// Created by 001730 chengxiao on 22/8/30.
//
#pragma once

#include "RedoxiTrack/detection/TrackTarget.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API BotsortTrackTarget : public TrackTarget
{
  public:
    TrackTargetPtr m_optical_target;
    TrackTargetPtr m_kalman_target;

    bool m_is_activated = false;

    virtual DetectionPtr clone() const override;
    virtual void copy_to(Detection &target) const override;

    void print() override;
};
using BotsortTrackTargetPtr = std::shared_ptr<BotsortTrackTarget>;
} // namespace RedoxiTrack