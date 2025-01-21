//
// Created by 001730 chengxiao on 2022/8/30.
//
#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/tracker/OpticalTrackerParam.h"
#include "RedoxiTrack/tracker/TrackerParam.h"


namespace RedoxiTrack
{
class REDOXI_TRACK_API BotsortTrackerParam : public TrackerParam
{
  public:
    void copy_to(TrackerParam &p) const override;

    std::shared_ptr<TrackerParam> clone() const override;

    std::shared_ptr<const OpticalTrackerParam> get_optical_param() const
    {
        return m_optical_param;
    }

    std::shared_ptr<const TrackerParam> get_kalman_param() const
    {
        return m_kalman_param;
    }

  public:
    float m_track_high_thresh = 0.6; // botsort nni 0.35  bytetrack0.5  botsort 0.6
    float m_track_low_thresh = 0.1;
    float m_new_track_thresh = 0.7; // botsort nni 0.5   bytetrack0.6  botsort 0.7
    int m_keep_track_buffer = 30;
    int m_max_time_lost = 30;
    float m_match_thresh = 0.8; // botsort nni 0.6   bytetrack0.8  botsort 0.8
    float m_aspect_ratio_thresh = 1.6;
    float m_min_box_area = 10.0;
    float m_proximity_thresh = 0.5;
    float m_appearance_thresh = 0.25; // botsort nni 0.5   botsort 0.25
    float m_alpha_smooth_features = 0.9;
    bool m_use_optical_before_track = false;
    bool m_fuse_score = false; // botsort/bytetrack false
    bool m_use_reid_feature = true;
    std::shared_ptr<OpticalTrackerParam> m_optical_param;
    std::shared_ptr<TrackerParam> m_kalman_param;
};

using BotsortTrackerParamPtr = std::shared_ptr<BotsortTrackerParam>;
} // namespace RedoxiTrack
