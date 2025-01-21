//
// Created by cx on 2025/1/6.
//

#include "RedoxiTrack/tracker/SimpleSortTrackerParam.h"

namespace RedoxiTrack
{
void SimpleSortTrackerParam::copy_to(TrackerParam &p) const
{
    TrackerParam::copy_to(p);
    SimpleSortTrackerParam *m = dynamic_cast<SimpleSortTrackerParam *>(&p);
    if (m) {
        m->m_max_gating_distance = m_max_gating_distance;
        m->m_base_gating_threshold = m_base_gating_threshold;
        m->m_alpha_smooth_features = m_alpha_smooth_features;
        m->m_gating_dist_lambda = m_gating_dist_lambda;
        m->m_duplicate_iou_dist = m_duplicate_iou_dist;
        m_kalman_param.copy_to(m->m_kalman_param);
    }
}

std::shared_ptr<TrackerParam> SimpleSortTrackerParam::clone() const
{
    std::shared_ptr<SimpleSortTrackerParam> m = std::make_shared<SimpleSortTrackerParam>();
    copy_to(*m);
    return m;
}


} // namespace RedoxiTrack
