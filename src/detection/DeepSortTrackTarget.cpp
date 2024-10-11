//
// Created by sfj on 2022/5/11.
//

#include "RedoxiTrack/detection/DeepSortTrackTarget.h"

namespace RedoxiTrack {
    DetectionPtr DeepSortTrackTarget::clone() const {
        auto output = std::make_shared<DeepSortTrackTarget>();
        copy_to(*output);
        return output;
    }

    void DeepSortTrackTarget::copy_to(Detection &target) const {
        auto p =dynamic_cast<DeepSortTrackTarget*>(&target);
        assert_throw(p, "failed to convert Detection to DeepSortTrackTarget");
        TrackTarget::copy_to(target);
        p->m_optical_target = m_optical_target;
        p->m_kalman_target = m_kalman_target;
    }

    void DeepSortTrackTarget::print() {
        std::cout<<"deepsort target"<<std::endl;
        TrackTarget::print();
        std::cout<<" optical target "<<std::endl;
        m_optical_target->print();
        std::cout<<" kalman target "<<std::endl;
        m_kalman_target->print();
    }

}

