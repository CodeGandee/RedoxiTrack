//
// Created by sfj on 2022/5/5.
//

#pragma once

#include "RedoxiTrack/detection/TrackTarget.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API DeepSortTrackTarget : public TrackTarget
{
  public:
    TrackTargetPtr m_optical_target;
    TrackTargetPtr m_kalman_target;

    virtual DetectionPtr clone() const override;
    virtual void copy_to(Detection &target) const override;

    void print() override;
};
using DeepSortTrackTargetPtr = std::shared_ptr<DeepSortTrackTarget>;
} // namespace RedoxiTrack
