//
// Created by cx on 2025/1/6.
//

#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/tracker/TrackerParam.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API SimpleSortTrackerParam : public TrackerParam
{
  public:
    void copy_to(TrackerParam &p) const override;

    std::shared_ptr<TrackerParam> clone() const override;

    const TrackerParam &get_kalman_param() const
    {
        return m_kalman_param;
    }

  public:
    // bbox mahalanobis distance, larger than this, the boxes are NOT considered matched
    float get_gating_threshold() const
    {
        float ratio = (m_preferred_image_size.width + m_preferred_image_size.height) / 2.0;
        return m_base_gating_threshold * ratio;
    }

  public:
    float m_max_gating_distance = 0.3;

    float m_base_gating_threshold = 6.325 * 1e-3;

    // bbox mahalanobis distance, 9.4877 in paper
    // float m_gating_threshold = 6.325 * 1e-3 * (1080 + 1920) / 2;

    float m_alpha_smooth_features = 0.9;
    float m_gating_dist_lambda = 0.98;
    float m_duplicate_iou_dist = 0.15;
    TrackerParam m_kalman_param;
};

using SimpleSortTrackerParamPtr = std::shared_ptr<SimpleSortTrackerParam>;
} // namespace RedoxiTrack
