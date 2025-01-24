//
// Created by 18200 on 2022/2/9.
//

#include "RedoxiTrack/tracker/DeepSortTrackerParam.h"

namespace RedoxiTrack
{
void DeepSortTrackerParam::copy_to(TrackerParam &p) const
{
    TrackerParam::copy_to(p);
    DeepSortTrackerParam *m = dynamic_cast<DeepSortTrackerParam *>(&p);
    if (m) {
        m->m_max_gating_distance = m_max_gating_distance;
        m->m_base_gating_threshold = m_base_gating_threshold;
        m->m_alpha_smooth_features = m_alpha_smooth_features;
        m->m_gating_dist_lambda = m_gating_dist_lambda;
        m->m_duplicate_iou_dist = m_duplicate_iou_dist;
        m->m_use_optical_before_track = m_use_optical_before_track;
        if (m_optical_param) {
            m->m_optical_param = std::make_shared<OpticalTrackerParam>();
            m_optical_param->copy_to(*m->m_optical_param);
            std::cout << "open optical tracker" << std::endl;
        }
        if (m_kalman_param) {
            m->m_kalman_param = std::make_shared<TrackerParam>();
            m_kalman_param->copy_to(*m->m_kalman_param);
        }
    }
}

std::shared_ptr<TrackerParam> DeepSortTrackerParam::clone() const
{
    std::shared_ptr<DeepSortTrackerParam> m = std::make_shared<DeepSortTrackerParam>();
    copy_to(*m);
    return m;
}


} // namespace RedoxiTrack
