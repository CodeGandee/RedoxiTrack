//
// Created by sfj on 2022/7/29.
//
#pragma once

#include "RedoxiTrack/utils/FeatureTraits.h"

namespace RedoxiTrack
{
class REDOXI_TRACK_API CosineFeature : public FeatureTraits
{
  public:
    virtual double distance(const fVECTOR &input1, const fVECTOR &input2) const override;
    virtual void linear_combine(fVECTOR *output, const fVECTOR &fa, const fVECTOR &fb, double wa = 1, double wb = 1) const override;
    virtual double max_distance() const override;
    virtual double min_distance() const override;
};
} // namespace RedoxiTrack
