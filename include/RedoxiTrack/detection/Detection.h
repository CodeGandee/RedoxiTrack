#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/detection/IDObject.h"

namespace RedoxiTrack
{
class Detection;
using DetectionPtr = std::shared_ptr<Detection>;

/**
 * a general detection
 */
class REDOXI_TRACK_API Detection : public IDObject
{
  protected:
    int m_type = DetectionTypes::None;

  public:
    /**
     * get bounding box of detection
     * @return
     */
    virtual BBOX get_bbox() const = 0;
    virtual float get_confidence() const = 0;
    virtual float get_quality() const = 0;

    /**
     * user-defined semantic type of the detection, 0 means no-type
     * @return
     */
    virtual int get_type()
    {
        return m_type;
    }
    virtual void set_type(int type)
    {
        m_type = type;
    }

    /**
     * get feature without copy
     * @param output
     */
    virtual void get_feature(fVECTOR &output) const = 0;

    /**
     * get a copy of the feature
     * @return
     */
    virtual fVECTOR get_feature() const
    {
        fVECTOR x;
        get_feature(x);
        return x;
    }

    /**
     * create new detection which has same content
     * @return
     */
    virtual DetectionPtr clone() const = 0;
    virtual void copy_to(Detection &to) const
    {
        to.m_type = m_type;
    }
};

} // namespace RedoxiTrack
