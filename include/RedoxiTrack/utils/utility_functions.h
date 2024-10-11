#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"
#include "RedoxiTrack/detection/Detection.h"
#include "RedoxiTrack/external/Hungarian.h"
#include "RedoxiTrack/external/lapjv.h"

namespace RedoxiTrack
{
REDOXI_TRACK_API void copy_kalmanFilter(const cv::KalmanFilter &from, cv::KalmanFilter &to);

REDOXI_TRACK_API float median(std::vector<float> &v, const float empty_output = -9999);
REDOXI_TRACK_API float compute_iou(const Detection &source, const Detection &target);
REDOXI_TRACK_API float compute_iou(const BBOX &source, const BBOX &target);

// out_distance[i][j] = IOU of source[i] and target[j]
REDOXI_TRACK_API void compute_pairwise_iou(const std::vector<DetectionPtr> &source,
                                           const std::vector<DetectionPtr> &target,
                                           fMATRIX *out_distance);

// return (u,v) means source[u] matches to target[v]
REDOXI_TRACK_API std::vector<std::pair<int, int>>
    match_detecion_by_iou(const std::vector<DetectionPtr> &source, const std::vector<DetectionPtr> &target);

REDOXI_TRACK_API std::vector<std::pair<int, int>>
    match_detecion_by_center_position(const std::vector<DetectionPtr> &source, const std::vector<DetectionPtr> &target);

// return (u,v) means source[u] matches to target[v]
REDOXI_TRACK_API void hungarian_match(const std::vector<std::vector<float>> &matrix_source2target,
                                      const int source_length, const int target_length, const float thresh,
                                      std::vector<std::pair<int, int>> &output_matched_pair,
                                      std::vector<int> &output_unmatched_source,
                                      std::vector<int> &output_unmatched_target);

REDOXI_TRACK_API void lapjv_match(const std::vector<std::vector<float>> &matrix_source2target,
                                  const int source_length, const int target_length, const float thresh,
                                  std::vector<std::pair<int, int>> &output_matched_pair,
                                  std::vector<int> &output_unmatched_source,
                                  std::vector<int> &output_unmatched_target);

REDOXI_TRACK_API std::vector<POINT> generate_uniform_keypoints(const BBOX &bbox, int pts_width, int pts_height, float margin = 0.25);

REDOXI_TRACK_API BBOX predict_bbox_by_keypoints(const BBOX &bbox,
                                                const POINT *point1,
                                                const POINT *point2,
                                                int number_points,
                                                const uint8_t *valid_points);

REDOXI_TRACK_API bool is_bbox_inside_image(const BBOX &bbox, const int img_width, const int img_height, float thresh = 1);

REDOXI_TRACK_API BBOX crop_bbox_inside_image(const BBOX &bbox, const int img_width, const int img_height);


} // namespace RedoxiTrack
