//
// Created by cx on 2025/1/6.
//
#include "RedoxiTrack/tracker/SimpleSortMotionPrediction.h"

namespace RedoxiTrack{
    void SimpleSortMotionPrediction::init(cv::KalmanFilter &kf, const BBOX &bbox) {
        m_stateNum = 8;
        m_measureNum = 4;
        m_update_measurement = cv::Mat::zeros(m_measureNum, 1, CV_32F);
        // state space: x, y, a(aspect ratio), h(height), vx, vy, va, vh
        kf = cv::KalmanFilter(m_stateNum, m_measureNum, 0);
        std::vector<float> n_xyah;
        _bbox2xyah(bbox, n_xyah);

        kf.transitionMatrix = (cv::Mat_<float>(m_stateNum, m_stateNum) <<   1, 0, 0, 0, 1, 0, 0, 0,
                                                                            0, 1, 0, 0, 0, 1, 0, 0,
                                                                            0, 0, 1, 0, 0, 0, 1, 0,
                                                                            0, 0, 0, 1, 0, 0, 0, 1,
                                                                            0, 0, 0, 0, 1, 0, 0, 0,
                                                                            0, 0, 0, 0, 0, 1, 0, 0,
                                                                            0, 0, 0, 0, 0, 0, 1, 0,
                                                                            0, 0, 0, 0, 0, 0, 0, 1);

        kf.measurementMatrix = (cv::Mat_<float>(m_measureNum, m_stateNum) <<1, 0, 0, 0, 0, 0, 0, 0,
                                                                            0, 1, 0, 0, 0, 0, 0, 0,
                                                                            0, 0, 1, 0, 0, 0, 0, 0,
                                                                            0, 0, 0, 1, 0, 0, 0, 0);

        kf.errorCovPost = (cv::Mat_<float>(m_stateNum, m_stateNum) <<   std::pow(2*m_std_weight_position*n_xyah[3], 2), 0, 0, 0, 0, 0, 0, 0,
                                                                        0, std::pow(2*m_std_weight_position*n_xyah[3], 2), 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 1, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, std::pow(2*m_std_weight_position*n_xyah[3], 2), 0, 0, 0, 0,
                                                                        0, 0, 0, 0, std::pow(10*m_std_weight_velocity*n_xyah[3], 2), 0, 0, 0,
                                                                        0, 0, 0, 0, 0, std::pow(10*m_std_weight_velocity*n_xyah[3], 2), 0, 0,
                                                                        0, 0, 0, 0, 0, 0, std::pow(1e-5, 2), 0,
                                                                        0, 0, 0, 0, 0, 0, 0, std::pow(10*m_std_weight_velocity*n_xyah[3], 2));
        kf.processNoiseCov = cv::Mat_<float>::zeros(m_stateNum, m_stateNum);
        kf.measurementNoiseCov = cv::Mat_<float>::zeros(m_measureNum, m_measureNum);
        // initialize state vector with bounding box in [x,y,a,h] styl
        kf.statePost.at<float>(0, 0) = n_xyah[0];
        kf.statePost.at<float>(1, 0) = n_xyah[1];
        kf.statePost.at<float>(2, 0) = n_xyah[2];
        kf.statePost.at<float>(3, 0) = n_xyah[3];
    }

    void SimpleSortMotionPrediction::predict(cv::KalmanFilter &kf, BBOX &output_bbox, int delta_frame_number, const bool flag) {
        // init Q matrix using height
        //Uncertainty is related to the height of the bbox
        //reference to deepsort code:https://github.com/ifzhang/FairMOT/blob/master/src/lib/tracking_utils/kalman_filter.py
        auto pose_cov = m_std_weight_position*kf.statePost.at<float>(3, 0) * delta_frame_number;
        auto velocity_cov = m_std_weight_velocity*kf.statePost.at<float>(3, 0) * delta_frame_number;
        kf.processNoiseCov.at<float>(0, 0) = std::pow(pose_cov, 2);
        kf.processNoiseCov.at<float>(1, 1) = std::pow(pose_cov, 2);
        kf.processNoiseCov.at<float>(2, 2) = 1;
        kf.processNoiseCov.at<float>(3, 3) = std::pow(pose_cov, 2);
        kf.processNoiseCov.at<float>(4, 4) = std::pow(velocity_cov, 2);
        kf.processNoiseCov.at<float>(5, 5) = std::pow(velocity_cov, 2);
        kf.processNoiseCov.at<float>(6, 6) = std::pow(1e-5, 2);
        kf.processNoiseCov.at<float>(7, 7) = std::pow(velocity_cov, 2);
        if (flag)
            kf.statePost.at<float>(7, 0) = 0;

        cv::Mat p = kf.predict();
        _xyah2bbox(p.at<float>(0, 0), p.at<float>(1, 0),
                          p.at<float>(2, 0), p.at<float>(3, 0),
                          output_bbox);
    }

    void SimpleSortMotionPrediction::update(cv::KalmanFilter &kf, const BBOX &bbox) {
//        cv::Mat measurement = cv::Mat::zeros(m_measureNum, 1, CV_32F);
        // measurement
        std::vector<float> n_xyah;
        _bbox2xyah(bbox, n_xyah);
        m_update_measurement.at<float>(0, 0) = n_xyah[0];
        m_update_measurement.at<float>(1, 0) = n_xyah[1];
        m_update_measurement.at<float>(2, 0) = n_xyah[2];
        m_update_measurement.at<float>(3, 0) = n_xyah[3];
        // init R matrix using height
        kf.measurementNoiseCov.at<float>(0, 0) = std::pow(m_std_weight_position*kf.statePre.at<float>(3, 0), 2);
        kf.measurementNoiseCov.at<float>(1, 1) = std::pow(m_std_weight_position*kf.statePre.at<float>(3, 0), 2);
        kf.measurementNoiseCov.at<float>(2, 2) = 1.0/30;
        kf.measurementNoiseCov.at<float>(3, 3) = std::pow(m_std_weight_position*kf.statePre.at<float>(3, 0), 2);

        // update
        kf.correct(m_update_measurement);
    }

    void SimpleSortMotionPrediction::get_bbox_state(cv::KalmanFilter &kf, BBOX &output_bbox) {
        const cv::Mat &s = kf.statePost;
        _xyah2bbox(s.at<float>(0, 0), s.at<float>(1, 0),
                          s.at<float>(2, 0), s.at<float>(3, 0),
                          output_bbox);
    }

    void SimpleSortMotionPrediction::project_state2measurement(cv::KalmanFilter &kf, cv::Mat &output_mean,
                                                            cv::Mat &output_covariance) const {
        cv::Mat_<float> innovation_cov = (cv::Mat_<float>(m_measureNum, m_measureNum) <<   std::pow(m_std_weight_position*kf.statePre.at<float>(3, 0), 2), 0, 0, 0,
                                                                                0, std::pow(m_std_weight_position*kf.statePre.at<float>(3, 0), 2), 0, 0,
                                                                                0, 0, std::pow(1e-1, 2), 0,
                                                                                0, 0, 0, std::pow(m_std_weight_position*kf.statePre.at<float>(3, 0), 2));

        output_mean = kf.measurementMatrix * kf.statePre;
        output_covariance = kf.measurementMatrix * kf.errorCovPre * kf.measurementMatrix.t() + innovation_cov;
    }

    void SimpleSortMotionPrediction::_bbox2xyah(const BBOX &bbox, std::vector<float> &output) {
        output.clear();
        output.push_back(bbox.x + bbox.width / 2);
        output.push_back(bbox.y + bbox.height / 2);
        output.push_back(bbox.width / bbox.height);
        output.push_back(bbox.height);
    }

    void SimpleSortMotionPrediction::_xyah2bbox(float x, float y, float a, float h, BBOX &output_bbox) {
//        if (std::isnan(x) || std::isinf(x))
//            return false;
//        if (std::isnan(y) || std::isinf(y))
//            return false;
//        if (std::isnan(a) || std::isinf(a))
//            return false;
//        if (std::isnan(h) || std::isinf(h))
//            return false;
        output_bbox.height = h;
        output_bbox.width = a * h;
        output_bbox.x = x - a * h / 2.0;
        output_bbox.y = y - h / 2.0;
//        return true;
    }

    MotionPredictionByKalmanPtr SimpleSortMotionPrediction::clone() const {
        auto output = std::make_shared<SimpleSortMotionPrediction>();
        copy_to(*output);
        return output;
    }

    void SimpleSortMotionPrediction::copy_to(MotionPredictionByKalman &to) const {
        auto p = dynamic_cast<SimpleSortMotionPrediction*>(&to);
        assert_throw(p, "failed to convert MotionPredictionByKalman to SimpleSortMotionPrediction");
        MotionPredictionByKalman::copy_to(to);
        p->m_std_weight_position = m_std_weight_position;
        p->m_update_measurement = m_update_measurement;
        p->m_std_weight_velocity = m_std_weight_velocity;
    }
}
