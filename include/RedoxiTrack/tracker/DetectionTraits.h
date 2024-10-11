//
// Created by sfj on 2022/5/23.
//

#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/detection/Detection.h"
#include "RedoxiTrack/utils/FeatureTraits.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API DetectionTraits
{
  public:
    virtual ~DetectionTraits()
    {
    }

  public:
    /**
     * compute distance of two detections
     * @param a
     * @param b
     * @return
     */
    virtual double compute_detection_distance(const Detection *a, const Detection *b) = 0;
};

class REDOXI_TRACK_API FeatureBasedDetTraits : public DetectionTraits
{
  public:
    virtual ~FeatureBasedDetTraits()
    {
    }

  public:
    /**
     * compute distance of two detections, according to detection's feature
     * @param a
     * @param b
     * @return
     */
    virtual double compute_detection_distance(const Detection *a, const Detection *b) override;
    virtual FeatureTraitsPtr get_feature_traits() const = 0;
};
using DetectionTraitsPtr = std::shared_ptr<DetectionTraits>;
} // namespace RedoxiTrack
