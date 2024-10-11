//
// Created by wangjing on 1/11/22.
//

#include "RedoxiTrack/detection/KalmanTrackTarget.h"
#include "RedoxiTrack/utils/utility_functions.h"

namespace RedoxiTrack{
    void KalmanTrackTarget::print() {
        TrackTarget::print();
        std::cout<<"kf state "<<std::endl;
        std::cout<<m_kf.statePost<<std::endl;
    }

    DetectionPtr KalmanTrackTarget::clone() const {
        auto output = std::make_shared<KalmanTrackTarget>();
        copy_to(*output);
        return output;
    }

    void KalmanTrackTarget::copy_to(Detection &target) const {
        auto p = dynamic_cast<KalmanTrackTarget*>(&target);
        assert_throw(p, "failed to convert Detection to Kalman!");
        TrackTarget::copy_to(target);
        copy_kalmanFilter(m_kf, p->m_kf);
        p->m_can_be_update = m_can_be_update;
    }
}
