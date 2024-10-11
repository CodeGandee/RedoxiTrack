//
// Created by 18200 on 2022/2/9.
//
#include "RedoxiTrack/tracker/OpticalTrackerParam.h"

namespace RedoxiTrack{
    void OpticalTrackerParam::copy_to(TrackerParam &p) const {
        TrackerParam::copy_to(p);
        OpticalTrackerParam* m = dynamic_cast<OpticalTrackerParam*>(&p);
        if(m){
            m->m_pts_per_width = m_pts_per_width;
            m->m_pts_per_height = m_pts_per_height;
        }
    }

    std::shared_ptr<TrackerParam> OpticalTrackerParam::clone() const {
        std::shared_ptr<OpticalTrackerParam> m = std::make_shared<OpticalTrackerParam>();
        copy_to(*m);
        return m;
    }
}