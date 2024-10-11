#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"

namespace RedoxiTrack
{

// parameters for the tracker
class REDOXI_TRACK_API TrackerParam
{
  public:
    TrackerParam()
    {
    }
    virtual ~TrackerParam()
    {
    }

    virtual void copy_to(TrackerParam &p) const;
    virtual std::shared_ptr<TrackerParam> clone() const;
    virtual void set_preferred_image_size(const cv::Size &size);

  public:
    /**
     * if the time of target has not be detected greater than m_max_time_since_update, it will be delete
     */
    int m_max_time_since_update = 30;
    float m_max_iou_distance = 0.5;

    cv::Size m_preferred_image_size{1920, 1080};
};
using TrackerParamPtr = std::shared_ptr<TrackerParam>;

} // namespace RedoxiTrack
