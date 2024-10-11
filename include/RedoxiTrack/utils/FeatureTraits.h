//
// Created by sfj on 2022/7/29.
//

#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API FeatureTraits
{
  public:
    virtual ~FeatureTraits()
    {
    }
    virtual double distance(const fVECTOR &input1, const fVECTOR &input2) const = 0;
    virtual void linear_combine(fVECTOR *output, const fVECTOR &fa, const fVECTOR &fb, double wa = 1, double wb = 1) const = 0;
    virtual double max_distance() const = 0;
    virtual double min_distance() const = 0;
};
using FeatureTraitsPtr = std::shared_ptr<FeatureTraits>;
} // namespace RedoxiTrack
