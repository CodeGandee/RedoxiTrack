//
// Created by 18200 on 2022/2/21.
//

#include "RedoxiTrack/tracker/OpencvOpticalFlow.h"
namespace RedoxiTrack {

    void OpencvOpticalFlow::set_prev_image(const cv::Mat &img) {
        img.copyTo(m_pre_img);
    }

    cv::Mat OpencvOpticalFlow::get_prev_image() {
        return m_pre_img.clone();
    }

    void OpencvOpticalFlow::set_current_image(const cv::Mat &img) {
        img.copyTo(m_cur_img);
    }

    void OpencvOpticalFlow::set_prev_image_by_current() {
        m_cur_img.copyTo(m_pre_img);
    }

    void OpencvOpticalFlow::predict_keypoint_location(const std::vector<POINT> &points,
                                                      MotionPredictionByImageKeypoint::Result &output) const {
        this->predict_keypoint_location(m_cur_img, points, output);
    }

    void OpencvOpticalFlow::predict_keypoint_location(const cv::Mat &cur, const vector<POINT> &points,
                                                      MotionPredictionByImageKeypoint::Result &output) const {
        auto p = dynamic_cast<OpticalFlowMotionPrediction::Result *>(&output);
        std::vector<float> similarity;
        p->keypoints_predicted.clear();
        p->keypoints_valid.clear();
        calcOpticalFlowPyrLK(m_pre_img, cur, points, p->keypoints_predicted, p->keypoints_valid, similarity);
    }

}

