//
// Created by 001730 chengxiao on 2022/8/30.
//

#include "RedoxiTrack/tracker/BotsortTrackerParam.h"
#include "RedoxiTrack/tracker/OpticalTrackerParam.h"
#include "RedoxiTrack/tracker/TrackerParam.h"

namespace RedoxiTrack{
    void BotsortTrackerParam::copy_to(TrackerParam &p) const {
        TrackerParam::copy_to(p);
        BotsortTrackerParam* m = dynamic_cast<BotsortTrackerParam*>(&p);
        if(m){
            m->m_track_high_thresh = m_track_high_thresh;
            m->m_track_low_thresh = m_track_low_thresh;
            m->m_new_track_thresh = m_new_track_thresh;
            m->m_keep_track_buffer = m_keep_track_buffer;
            m->m_max_time_lost = m_max_time_lost;
            m->m_match_thresh = m_match_thresh;
            m->m_aspect_ratio_thresh = m_aspect_ratio_thresh;
            m->m_min_box_area = m_min_box_area;
            m->m_proximity_thresh = m_proximity_thresh;
            m->m_appearance_thresh = m_appearance_thresh;
            m->m_alpha_smooth_features = m_alpha_smooth_features;
            m->m_use_optical_before_track = m_use_optical_before_track;
            m->m_fuse_score = m_fuse_score;
            m->m_use_reid_feature = m_use_reid_feature;
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

    std::shared_ptr<TrackerParam> BotsortTrackerParam::clone() const {
        std::shared_ptr<BotsortTrackerParam> m = std::make_shared<BotsortTrackerParam>();
        copy_to(*m);
        return m;
    }


}