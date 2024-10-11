#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/detection/Detection.h"
#include <vector>

namespace RedoxiTrack
{

class REDOXI_TRACK_API SingleDetection : public Detection
{
  protected:
    BBOX m_bbox;
    fVECTOR m_feature;
    float m_confidence;
    float m_quality;

  public:
    virtual void set_bbox(const BBOX &box);
    virtual void set_feature(const fVECTOR &x);
    virtual void set_confidence(const float &conf);
    virtual void set_quality(const float &q);

    virtual BBOX get_bbox() const override;
    virtual float get_quality() const override;
    float get_confidence() const override;

    /**
     * get feature without copy
     * @param output
     */
    virtual void get_feature(fVECTOR &output) const override;
    virtual fVECTOR get_feature() const override;

    DetectionPtr clone() const override;

    void copy_to(Detection &to) const override;
};

using SingleDetectionPtr = std::shared_ptr<SingleDetection>;

} // namespace RedoxiTrack
