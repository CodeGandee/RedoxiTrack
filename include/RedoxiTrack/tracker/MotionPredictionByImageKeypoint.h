//
// Created by wangjing on 12/31/21.
//

#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
namespace RedoxiTrack
{
class REDOXI_TRACK_API MotionPredictionByImageKeypoint
{
  public:
    class REDOXI_TRACK_API Result
    {
      public:
        virtual ~Result()
        {
        }
        std::vector<POINT> keypoints_predicted;
    };

  public:
    virtual ~MotionPredictionByImageKeypoint()
    {
    }

    virtual void predict_keypoint_location(const std::vector<POINT> &points, Result &output) const = 0;

    virtual void predict_keypoint_location(const std::vector<POINT> &points, Result &output)
    {
        const MotionPredictionByImageKeypoint *x = this;
        x->predict_keypoint_location(points, output);
    }
};

} // namespace RedoxiTrack
