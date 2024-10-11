#include "RedoxiTrack/detection/TrackTarget.h"
#include "RedoxiTrack/utils/utility_functions.h"

namespace RedoxiTrack
{
    void TrackTarget::set_underlying_detection(const DetectionPtr &det, bool update_properties) {
        m_detection = det;
        if(update_properties){
            set_bbox(det->get_bbox());
            set_quality(det->get_quality());
            set_confidence(det->get_confidence());
            set_feature(det->get_feature());
            set_type(det->get_type());
        }
    }

    DetectionPtr TrackTarget::clone() const {
        auto output = clone(true);
        return output;
    }

    void TrackTarget::copy_to(Detection &target) const {
        copy_to(target, true);
    }

    DetectionPtr TrackTarget::clone(bool with_detection) const {
        auto output = std::make_shared<TrackTarget>();
        copy_to(*output, with_detection);
        return output;
    }

    void TrackTarget::copy_to(Detection &target, bool with_detection) const {
        auto* _target_ptr = dynamic_cast<TrackTarget*>(&target);
        assert_throw(_target_ptr, "Detection convert target failed!");
        SingleDetection::copy_to(target);
        auto& _target = *_target_ptr;
        if(m_detection)
        {
            if(with_detection) {
                if (_target.m_detection)
                    m_detection->copy_to(*_target.m_detection);
                else
                    _target.m_detection = m_detection->clone();
            }
            else
                _target.m_detection = m_detection;
        }
        _target.m_path_id = m_path_id;
        _target.m_start_frame_number = m_start_frame_number;
        _target.m_end_frame_number = m_end_frame_number;
        _target.m_path_state = m_path_state;
    }

    void TrackTarget::print() {
        std::cout<<"id "<<m_path_id<<" bbox "<<m_bbox<<std::endl;
    }

    TrackTarget::TrackTarget()
    {
        //ctor
    }

    TrackTarget::~TrackTarget()
    {
        //dtor
    }

}

