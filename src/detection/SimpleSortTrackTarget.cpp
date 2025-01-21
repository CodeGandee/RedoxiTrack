//
// Created by cx on 2025/1/6.
//

#include "RedoxiTrack/detection/SimpleSortTrackTarget.h"

namespace RedoxiTrack {
    DetectionPtr SimpleSortTrackTarget::clone() const {
        auto output = std::make_shared<SimpleSortTrackTarget>();
        copy_to(*output);
        return output;
    }

    void SimpleSortTrackTarget::copy_to(Detection &target) const {
        auto p =dynamic_cast<SimpleSortTrackTarget*>(&target);
        assert_throw(p, "failed to convert Detection to SimpleSortTrackTarget");
        TrackTarget::copy_to(target);
        p->m_kalman_target = m_kalman_target;
    }

    void SimpleSortTrackTarget::print() {
        std::cout<<"simplesort target"<<std::endl;
        TrackTarget::print();
        std::cout<<" kalman target "<<std::endl;
        m_kalman_target->print();
    }

}
