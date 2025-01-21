//
// Created by cx on 2025/1/6.
//

#pragma once

#include "RedoxiTrack/detection/TrackTarget.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API SimpleSortTrackTarget : public TrackTarget
{
  public:
    TrackTargetPtr m_kalman_target;

    virtual DetectionPtr clone() const override;
    virtual void copy_to(Detection &target) const override;

    void print() override;
};
using SimpleSortTrackTargetPtr = std::shared_ptr<SimpleSortTrackTarget>;
} // namespace RedoxiTrack
