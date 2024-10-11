#include "RedoxiTrack/detection/SingleDetection.h"

namespace RedoxiTrack
{
    BBOX SingleDetection::get_bbox() const
    {
        return m_bbox;
    }

    float SingleDetection::get_confidence() const
    {
        return m_confidence;
    }

    float SingleDetection::get_quality() const
    {
        return m_quality;
    }

    void SingleDetection::get_feature(fVECTOR& output) const
    {
        output = m_feature;
    }


    void SingleDetection::set_bbox(const BBOX& box){
        m_bbox = box;
    }

    void SingleDetection::set_feature(const fVECTOR& x){
        m_feature = x;
    }

    void SingleDetection::set_confidence(const float& conf){
        m_confidence = conf;
    }

    void SingleDetection::set_quality(const float& q){
        m_quality = q;
    }

    fVECTOR SingleDetection::get_feature() const {
        fVECTOR x;
        get_feature(x);
        return x;
    }

    DetectionPtr SingleDetection::clone() const {
        auto p = std::make_shared<SingleDetection>();
        copy_to(*p);
        return p;
    }

    void SingleDetection::copy_to(Detection &to) const {
        auto p = dynamic_cast<SingleDetection*>(&to);
        assert_throw(p, "Failed convert Detection to SingleDetection");
        Detection::copy_to(to);
        p->m_feature = m_feature;
        p->m_confidence = m_confidence;
        p->m_quality = m_quality;
        p->m_bbox = m_bbox;
    }

}
