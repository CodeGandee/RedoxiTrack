#include "RedoxiTrack/utils/utility_functions.h"

namespace RedoxiTrack{
    float median(std::vector<float> &v, const float empty_output)
    {
        if (v.empty())
            return empty_output;
        int n = floor(v.size() / 2);
        nth_element(v.begin(), v.begin() + n, v.end());
        return v[n];
    }

    void copy_kalmanFilter(const cv::KalmanFilter &from, cv::KalmanFilter &to){
         from.transitionMatrix.copyTo(to.transitionMatrix);
         from.measurementMatrix.copyTo(to.measurementMatrix);
         from.processNoiseCov.copyTo(to.processNoiseCov);
         from.measurementNoiseCov.copyTo(to.measurementNoiseCov);
         from.errorCovPost.copyTo(to.errorCovPost);
         from.statePre.copyTo(to.statePre);
         from.statePost.copyTo(to.statePost);
         from.controlMatrix.copyTo(to.controlMatrix);
         from.errorCovPre.copyTo(to.errorCovPre);
         from.gain.copyTo(to.gain);
         from.temp1.copyTo(to.temp1);
         from.temp2.copyTo(to.temp2 );
         from.temp3.copyTo(to.temp3);
         from.temp4.copyTo(to.temp4 );
         from.temp5.copyTo(to.temp5);
    }

    float compute_iou(const Detection& source, const Detection& target){
        return compute_iou(source.get_bbox(), target.get_bbox());
    }

    float compute_iou(const BBOX& source, const BBOX& target)
    {
        // float in = (source & target).area();
        // float un = source.area() + target.area() - in;
        // if (un < DBL_EPSILON){
        //     return 0;
        // }
        // else{
        //     return (float)(in / un);
        // }
        float iou = 0.0;
        float source_x1 = source.x, source_y1 = source.y, source_x2 = source.br().x, source_y2 = source.br().y;
        float target_x1 = target.x, target_y1 = target.y, target_x2 = target.br().x, target_y2 = target.br().y;

        float box_area = (target_x2 - target_x1 + 1) * (target_y2 - target_y1 + 1);
        float iw = min(source_x2, target_x2) - max(source_x1, target_x1) + 1;
        if (iw > 0) {
            float ih = min(source_y2, target_y2) - max(source_y1, target_y1) + 1;
            if (ih > 0) {
                float ua = float((source_x2 - source_x1 + 1) * (source_y2 - source_y1 + 1) + box_area - iw * ih);
                iou = iw * ih / ua;
            }
        }
        return iou;
    }
    // in this namespace, source represent now(detection), target represent prev predict(tracker)
    void
    hungarian_match(const std::vector<std::vector<float>> &matrix_source2target,
                    const int source_length, const int target_length, const float thresh,
                    std::vector<std::pair<int, int>> &output_matched_pair,
                    std::vector<int> &output_unmatched_source,
                    std::vector<int> &output_unmatched_target) {
        std::vector<int> assignment;
        RedoxiTrack::HungarianAlgorithm HungAlgo;
        // assignment: target index
        HungAlgo.Solve(const_cast<std::vector<std::vector<float>>&>(matrix_source2target), assignment);

        std::vector<bool> unmatched_rows(source_length, true);
        std::vector<bool> unmatched_cols(target_length, true);
        for (unsigned int row = 0; row < source_length; row++) {
            int &col = assignment[row];
            if (col == -1 || col >= target_length)
                continue;
            if (matrix_source2target[row][col] <= thresh) {
                output_matched_pair.push_back(std::pair<int, int>(row, col));
                unmatched_rows[row] = false;
                unmatched_cols[col] = false;
            }
        }
        for (unsigned int i = 0; i < source_length; i++)
            if (unmatched_rows[i])
                output_unmatched_source.push_back(i);
        for (unsigned int i = 0; i < target_length; i++)
            if (unmatched_cols[i])
                output_unmatched_target.push_back(i);
    }

    void lapjv_match( const std::vector<std::vector<float>> &matrix_source2target,
                          const int source_length, const int target_length, const float thresh,
                          std::vector<std::pair<int, int>> &output_matched_pair,
                          std::vector<int> &output_unmatched_source,
                          std::vector<int> &output_unmatched_target) {
        if (source_length == 0 && target_length == 0) return;
        else if (source_length == 0) {
            for (int i = 0; i < target_length; i++) {
                output_unmatched_target.push_back(i);
            }
            return;
        }
        else if (target_length == 0) {
            for (int i = 0; i < source_length; i++) {
                output_unmatched_source.push_back(i);
            }
            return;
        }
        uint_t n = source_length + target_length;
        std::vector<std::vector<double>> cost_matrix(n, std::vector<double>(n, thresh / 2));
        for (unsigned int row = 0; row < n; row++) {
            for (unsigned int col = 0; col < n; col++) {
                if (row < source_length && col < target_length) {
                    cost_matrix[row][col] = matrix_source2target[row][col];
                }
                else if (row >= source_length && col >= target_length) {
                    cost_matrix[row][col] = 0.0;
                }
            }
        }

        double **cost_ptr = (double **) malloc(n * sizeof(double *));
        for (int i = 0; i < n; i++) {
            cost_ptr[i] = cost_matrix[i].data();
        }

        int_t *x = (int_t *) malloc(n * sizeof(int_t));
        int_t *y = (int_t *) malloc(n * sizeof(int_t));

        int ret = lapjv_internal(n, cost_ptr, x, y);
        free(cost_ptr);
        assert_throw(ret == 0, "Unknown error (lapjv_internal returned %d).");

        if (n != source_length) {
            for (int j = 0; j < n; j++) {
                if (x[j] >= target_length)
                    x[j] = -1;
                if (y[j] >= source_length)
                    y[j] = -1;
            }
        }

        for (int i = 0; i < source_length; i++) {
            if (x[i] != -1) {
                output_matched_pair.push_back(std::pair<int, int>(i, x[i]));
            }
            else {
                output_unmatched_source.push_back(i);
            }
        }
        for (int i = 0; i < target_length; i++) {
            if (y[i] == -1) {
                output_unmatched_target.push_back(i);
            }
        }
    }

    std::vector<POINT>
    generate_uniform_keypoints(const BBOX &bbox, int pts_width, int pts_height, float margin) {
        std::vector<POINT> output;
        float margin_w = bbox.width * margin;
        float margin_h = bbox.height * margin;
        float stepx = (bbox.width - 2 * margin_w) / pts_width;
        float stepy = (bbox.height - 2 * margin_h) / pts_height;
        for(int j = 0; j < pts_height; j++){
            for(int i = 0; i < pts_width; i++){
                float x = bbox.x + margin_w + i * stepx;
                float y = bbox.y + margin_h + j * stepy;
//                int x = int(bbox.x + margin_w) + i * ceil(stepx);
//                int y = int(bbox.y + margin_h) + j * ceil(stepy);
                output.push_back(POINT(x, y));
            }
        }
        return output;
    }

    BBOX
    predict_bbox_by_keypoints(const BBOX &bbox, const POINT *point1, const POINT *point2,
                                                           int number_points, const uint8_t *valid_points) {
        std::vector<float> xoff;
        std::vector<float> yoff;
        for(int i = 0; i < number_points; i++){
            if (valid_points[i]){
                xoff.push_back(point2[i].x - point1[i].x);
                yoff.push_back(point2[i].y - point1[i].y);
            }
        }

        std::vector<float> d;
        for(int i = 0; i < number_points; i++){
            for(int j = i+1; j < number_points; j++){
                if (valid_points[i])
                    d.push_back(norm(point2[i] - point2[j]) / norm(point1[i] - point1[j]));
            }
        }

        float dx = median(xoff);
        float dy = median(yoff);
        float s = median(d);
        if (s > 1.1f)
            s = 1.f;
        if (s < 0.9f)
            s = 1.f;

        BBOX output;
        s = 1;
        float s1 = 0.5f * (s - 1.f) * bbox.width;
        float s2 = 0.5f * (s - 1.f) * bbox.height;
        output.x = round(bbox.x + dx - s1);
        output.y = round(bbox.y + dy - s2);
        output.width = round(bbox.width * s);
        output.height = round(bbox.height * s);
        return output;
    }

    bool is_bbox_inside_image(const BBOX &bbox, const int img_width, const int img_height,
                                                           float thresh) {
        BBOX img_rect(0, 0, img_width, img_height);
        return (bbox & img_rect).area() > thresh;
    }

    BBOX
    crop_bbox_inside_image(const BBOX &bbox, const int img_width, const int img_height) {
        BBOX output;
        BBOX img_rect(0, 0, img_width, img_height);
        output = img_rect & bbox;
        return output;
    }
}