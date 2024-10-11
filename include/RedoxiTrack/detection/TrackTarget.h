#pragma once

#include "RedoxiTrack/detection/SingleDetection.h"


namespace RedoxiTrack
{
class TrackTarget;
using TrackTargetPtr = std::shared_ptr<TrackTarget>;
class REDOXI_TRACK_API TrackTarget : public SingleDetection
{
  public:
    TrackTarget();
    virtual ~TrackTarget();

    virtual int get_path_id() const
    {
        return m_path_id;
    }
    virtual void set_path_id(int x)
    {
        m_path_id = x;
    }

    virtual int get_start_frame_number() const
    {
        return m_start_frame_number;
    }
    virtual void set_start_frame_number(int x)
    {
        m_start_frame_number = x;
    }

    virtual int get_end_frame_number() const
    {
        return m_end_frame_number;
    }
    virtual void set_end_frame_number(int x)
    {
        m_end_frame_number = x;
    }

    virtual int get_path_state() const
    {
        return m_path_state;
    }
    virtual void set_path_state(int state)
    {
        m_path_state = state;
    }

    virtual void print();

    virtual const DetectionPtr &get_underlying_detection() const
    {
        return m_detection;
    }
    /**
     * set the actual object being tracked, this will update the bounding box of this track target
     * if update_properties=true, propertiex inculding bbox, feature, confidence and quality etc.
     * @param det
     * @param update_properties
     */
    virtual void set_underlying_detection(const DetectionPtr &det, bool update_properties);

    virtual DetectionPtr clone() const override;
    virtual void copy_to(Detection &target) const override;

    virtual DetectionPtr clone(bool with_detection) const;
    virtual void copy_to(Detection &target, bool with_detection) const;


  protected:
    virtual TrackTargetPtr _create_empty_target() const
    {
        return std::make_shared<TrackTarget>();
    };

    DetectionPtr m_detection;
    int m_path_id = 0;
    int m_start_frame_number = -1;
    int m_end_frame_number = -1;
    int m_path_state = TrackPathStateBitmask::None;
};


} // namespace RedoxiTrack
