//
// Created by 18200 on 2022/2/8.
//

#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/tracker/TrackerParam.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API OpticalTrackerParam : public TrackerParam
{
  public:
    void copy_to(TrackerParam &p) const override;

    std::shared_ptr<TrackerParam> clone() const override;

  public:
    /**
     * numbers of points in unit height
     */
    int m_pts_per_height = 5;
    int m_pts_per_width = 5;
};

using OpticalTrackerParamPtr = std::shared_ptr<OpticalTrackerParam>;
} // namespace RedoxiTrack
