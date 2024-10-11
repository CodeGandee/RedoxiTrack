#pragma once

#include "RedoxiTrack/detection/Detection.h"
#include "RedoxiTrack/tracker/DetectionComparision.h"

namespace RedoxiTrack
{

enum softnms_method { NMS,
                      SOFTNMS,
                      SIGMA_SOFTNMS };

class REDOXI_TRACK_API SoftNMS
{
  private:
    enum softnms_method m_method = SOFTNMS;
    float m_sigma = 0.5;
    float m_iou_thres = 0.5;
    float m_iou_scale = 0.5;
    bool m_use_IDfeature = true;
    std::vector<RedoxiTrack::DetectionPtr> m_detections;
    std::vector<float> m_scores;
    std::map<RedoxiTrack::DetectionPtr,
             std::vector<RedoxiTrack::DetectionPtr>>
        m_map_influenced_box;
    RedoxiTrack::DetectionComparisionPtr m_compute_IDdistance;

  public:
    void set_method(enum softnms_method method);
    void set_sigma(float sigma);
    void set_iou_thres(float iou_thres);
    void set_iou_scale(float iou_scale);
    void set_use_IDfeature(bool use_IDfeature);
    void set_compute_IDdistance(const RedoxiTrack::DetectionComparisionPtr &compute_IDdistance);
    std::vector<float> apply(std::vector<RedoxiTrack::DetectionPtr> &detections);
    std::map<RedoxiTrack::DetectionPtr, std::vector<RedoxiTrack::DetectionPtr>>
        get_influenced_boxes();
    std::vector<RedoxiTrack::DetectionPtr> filter(const float thres, std::vector<int> &del_inds);
};

} // namespace RedoxiTrack
