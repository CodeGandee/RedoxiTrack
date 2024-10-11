//
// Created by 001730 chengxiao on 22/8/30.
//

#include "RedoxiTrack/detection/BotsortTrackTarget.h"

namespace RedoxiTrack {
    DetectionPtr BotsortTrackTarget::clone() const {
        auto output = std::make_shared<BotsortTrackTarget>();
        copy_to(*output);
        return output;
    }

    void BotsortTrackTarget::copy_to(Detection &target) const {
        auto p =dynamic_cast<BotsortTrackTarget*>(&target);
        assert_throw(p, "failed to convert Detection to BotsortTrackTarget");
        TrackTarget::copy_to(target);
        p->m_optical_target = m_optical_target;
        p->m_kalman_target = m_kalman_target;
        p->m_is_activated = m_is_activated;
    }

    void BotsortTrackTarget::print() {
        std::cout<<"botsort target"<<std::endl;
        TrackTarget::print();
        std::cout<<" optical target "<<std::endl;
        m_optical_target->print();
        std::cout<<" kalman target "<<std::endl;
        m_kalman_target->print();
    }

}