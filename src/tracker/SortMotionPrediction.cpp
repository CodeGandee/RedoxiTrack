//
// Created by 18200 on 2022/1/18.
//
#include "RedoxiTrack/tracker/SortMotionPrediction.h"

namespace RedoxiTrack{
    void SortMotionPrediction::init(cv::KalmanFilter &kf, const BBOX &bbox) {
        m_stateNum = 7;
        m_measureNum = 4;
        m_update_measurement = cv::Mat::zeros(m_measureNum, 1, CV_32F);
        kf = cv::KalmanFilter(m_stateNum, m_measureNum, 0);

        kf.transitionMatrix = (cv::Mat_<float>(m_stateNum, m_stateNum) << 1, 0, 0, 0, 1, 0, 0,
                0, 1, 0, 0, 0, 1, 0,
                0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 1);

        setIdentity(kf.measurementMatrix);
        setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
        setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
        setIdentity(kf.errorCovPost, cv::Scalar::all(1));

        // initialize state vector with bounding box in [cx,cy,s,r] style
        kf.statePost.at<float>(0, 0) = bbox.x + bbox.width / 2.f;
        kf.statePost.at<float>(1, 0) = bbox.y + bbox.height / 2.f;
        kf.statePost.at<float>(2, 0) = bbox.area();
        kf.statePost.at<float>(3, 0) = (float)bbox.width / (float)bbox.height;
    }

    void SortMotionPrediction::predict(cv::KalmanFilter &kf, BBOX &output_bbox, int delta_frame_number, const bool flag) {
        cv::Mat p = kf.predict();
        _get_rect_from_xysr(p.at<float>(0, 0), p.at<float>(1, 0),
                                   p.at<float>(2, 0), p.at<float>(3, 0),
                                   output_bbox);
    }

    void SortMotionPrediction::update(cv::KalmanFilter &kf, const BBOX &bbox) {
        // measurement
        m_update_measurement.at<float>(0, 0) = bbox.x + bbox.width / 2.f;
        m_update_measurement.at<float>(1, 0) = bbox.y + bbox.height / 2.f;
        m_update_measurement.at<float>(2, 0) = bbox.area();
        m_update_measurement.at<float>(3, 0) = (float)bbox.width / (float)bbox.height;
        // update
        kf.correct(m_update_measurement);
    }

    void SortMotionPrediction::get_bbox_state(cv::KalmanFilter &kf, BBOX &output_bbox) {
        const cv::Mat &s = kf.statePost;
        _get_rect_from_xysr(s.at<float>(0, 0), s.at<float>(1, 0),
                             s.at<float>(2, 0), s.at<float>(3, 0),
                             output_bbox);
    }

    void SortMotionPrediction::project_state2measurement(cv::KalmanFilter &kf, cv::Mat &output_mean,
                                                         cv::Mat &output_covariance) const {
        output_mean = kf.measurementMatrix * kf.statePre;
        output_covariance = kf.measurementMatrix * kf.errorCovPre * kf.measurementMatrix.t();
    }

    void SortMotionPrediction::_get_rect_from_xysr(float cx, float cy, float s, float r, BBOX &output_bbox) {
//        if (std::isnan(cx) || std::isinf(cx))
//            return false;
//        if (std::isnan(cy) || std::isinf(cy))
//            return false;
//        if (std::isnan(s) || std::isinf(s) || s <= 1.f)
//            return false;
//        if (std::isnan(r) || std::isinf(r) || r <= 0.01f || r >= 100.f)
//            return false;

        float w = sqrt(s * r);
        float h = s / w;
        float x = (cx - w / 2);
        float y = (cy - h / 2);

        if (x < 0 && cx > 0)
            x = 0;
        if (y < 0 && cy > 0)
            y = 0;

        output_bbox.x = ceil(x);
        output_bbox.y = ceil(y);
        output_bbox.width = floor(w);
        output_bbox.height = floor(h);
//        return true;
    }
}
