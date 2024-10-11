//
// Created by 001730 chengxiao on 22/8/30.
//
#include "RedoxiTrack/tracker/BotsortMotionPrediction.h"

namespace RedoxiTrack{
    void BotsortMotionPrediction::init(cv::KalmanFilter &kf, const BBOX &bbox) {
        m_stateNum = 8;
        m_measureNum = 4;
        m_update_measurement = cv::Mat::zeros(m_measureNum, 1, CV_32F);
        std::vector<float> n_xcycwh;
        _bbox2xcycwh(bbox, n_xcycwh);
        m_update_measurement.at<float>(0, 0) = n_xcycwh[0];
        m_update_measurement.at<float>(1, 0) = n_xcycwh[1];
        m_update_measurement.at<float>(2, 0) = n_xcycwh[2];
        m_update_measurement.at<float>(3, 0) = n_xcycwh[3];

        // state space: xc(center x), yc(center y), w(width), h(height), vxc, vyc, vw, vh
        kf = cv::KalmanFilter(m_stateNum, m_measureNum, 0);
        // A
        kf.transitionMatrix = (cv::Mat_<float>(m_stateNum, m_stateNum) <<   1, 0, 0, 0, 1, 0, 0, 0,
                                                                            0, 1, 0, 0, 0, 1, 0, 0,
                                                                            0, 0, 1, 0, 0, 0, 1, 0,
                                                                            0, 0, 0, 1, 0, 0, 0, 1,
                                                                            0, 0, 0, 0, 1, 0, 0, 0,
                                                                            0, 0, 0, 0, 0, 1, 0, 0,
                                                                            0, 0, 0, 0, 0, 0, 1, 0,
                                                                            0, 0, 0, 0, 0, 0, 0, 1);
        // H
        kf.measurementMatrix = (cv::Mat_<float>(m_measureNum, m_stateNum) <<1, 0, 0, 0, 0, 0, 0, 0,
                                                                            0, 1, 0, 0, 0, 0, 0, 0,
                                                                            0, 0, 1, 0, 0, 0, 0, 0,
                                                                            0, 0, 0, 1, 0, 0, 0, 0);
        // posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
        kf.errorCovPost = (cv::Mat_<float>(m_stateNum, m_stateNum) <<   std::pow(2 * m_std_weight_position * n_xcycwh[2], 2), 0, 0, 0, 0, 0, 0, 0,
                                                                        0, std::pow(2 * m_std_weight_position * n_xcycwh[3], 2), 0, 0, 0, 0, 0, 0,
                                                                        0, 0, std::pow(2 * m_std_weight_position * n_xcycwh[2], 2), 0, 0, 0, 0, 0,
                                                                        0, 0, 0, std::pow(2 * m_std_weight_position * n_xcycwh[3], 2), 0, 0, 0, 0,
                                                                        0, 0, 0, 0, std::pow(10 * m_std_weight_velocity * n_xcycwh[2], 2), 0, 0, 0,
                                                                        0, 0, 0, 0, 0, std::pow(10 * m_std_weight_velocity * n_xcycwh[3], 2), 0, 0,
                                                                        0, 0, 0, 0, 0, 0, std::pow(10 * m_std_weight_velocity * n_xcycwh[2], 2), 0,
                                                                        0, 0, 0, 0, 0, 0, 0, std::pow(10 * m_std_weight_velocity * n_xcycwh[3], 2));
        kf.processNoiseCov = cv::Mat_<float>::zeros(m_stateNum, m_stateNum);
        kf.measurementNoiseCov = cv::Mat_<float>::zeros(m_measureNum, m_measureNum);
        // initialize state vector with bounding box in [xc,yc,w,h] style
        kf.statePost.at<float>(0, 0) = n_xcycwh[0];
        kf.statePost.at<float>(1, 0) = n_xcycwh[1];
        kf.statePost.at<float>(2, 0) = n_xcycwh[2];
        kf.statePost.at<float>(3, 0) = n_xcycwh[3];
    }

    void BotsortMotionPrediction::predict(cv::KalmanFilter &kf, BBOX &output_bbox, int delta_frame_number, const bool flag) {
        // init Q matrix using height
        //Uncertainty is related to the height of the bbox
        kf.processNoiseCov.at<float>(0, 0) = std::pow(m_std_weight_position * kf.statePost.at<float>(2, 0) * delta_frame_number, 2);
        kf.processNoiseCov.at<float>(1, 1) = std::pow(m_std_weight_position * kf.statePost.at<float>(3, 0) * delta_frame_number, 2);
        kf.processNoiseCov.at<float>(2, 2) = std::pow(m_std_weight_position * kf.statePost.at<float>(2, 0) * delta_frame_number, 2);
        kf.processNoiseCov.at<float>(3, 3) = std::pow(m_std_weight_position * kf.statePost.at<float>(3, 0) * delta_frame_number, 2);
        kf.processNoiseCov.at<float>(4, 4) = std::pow(m_std_weight_velocity * kf.statePost.at<float>(2, 0) * delta_frame_number, 2);
        kf.processNoiseCov.at<float>(5, 5) = std::pow(m_std_weight_velocity * kf.statePost.at<float>(3, 0) * delta_frame_number, 2);
        kf.processNoiseCov.at<float>(6, 6) = std::pow(m_std_weight_velocity * kf.statePost.at<float>(2, 0) * delta_frame_number, 2);
        kf.processNoiseCov.at<float>(7, 7) = std::pow(m_std_weight_velocity * kf.statePost.at<float>(3, 0) * delta_frame_number, 2);
        if (flag) {
            kf.statePost.at<float>(6, 0) = 0;
            kf.statePost.at<float>(7, 0) = 0;
        }

        cv::Mat p = kf.predict();
        _xcycwh2bbox(p.at<float>(0, 0), p.at<float>(1, 0),
                          p.at<float>(2, 0), p.at<float>(3, 0),
                          output_bbox);
    }

    void BotsortMotionPrediction::update(cv::KalmanFilter &kf, const BBOX &bbox) {
//        cv::Mat measurement = cv::Mat::zeros(m_measureNum, 1, CV_32F);
        // measurement
        std::vector<float> n_xcycwh;
        _bbox2xcycwh(bbox, n_xcycwh);
        m_update_measurement.at<float>(0, 0) = n_xcycwh[0];
        m_update_measurement.at<float>(1, 0) = n_xcycwh[1];
        m_update_measurement.at<float>(2, 0) = n_xcycwh[2];
        m_update_measurement.at<float>(3, 0) = n_xcycwh[3];
        // init R matrix using height
        kf.measurementNoiseCov.at<float>(0, 0) = std::pow(m_std_weight_position*kf.statePre.at<float>(2, 0), 2);
        kf.measurementNoiseCov.at<float>(1, 1) = std::pow(m_std_weight_position*kf.statePre.at<float>(3, 0), 2);
        kf.measurementNoiseCov.at<float>(2, 2) = std::pow(m_std_weight_position*kf.statePre.at<float>(2, 0), 2);
        kf.measurementNoiseCov.at<float>(3, 3) = std::pow(m_std_weight_position*kf.statePre.at<float>(3, 0), 2);

        // update
        kf.correct(m_update_measurement);
    }

    void BotsortMotionPrediction::get_bbox_state(cv::KalmanFilter &kf, BBOX &output_bbox) {
        const cv::Mat &s = kf.statePost;
        _xcycwh2bbox(s.at<float>(0, 0), s.at<float>(1, 0),
                          s.at<float>(2, 0), s.at<float>(3, 0),
                          output_bbox);
    }

    void BotsortMotionPrediction::project_state2measurement(cv::KalmanFilter &kf, cv::Mat &output_mean,
                                                            cv::Mat &output_covariance) const {
        cv::Mat_<float> innovation_cov = (cv::Mat_<float>(m_measureNum, m_measureNum) <<   std::pow(m_std_weight_position*kf.statePre.at<float>(2, 0), 2), 0, 0, 0,
                                                                                0, std::pow(m_std_weight_position*kf.statePre.at<float>(3, 0), 2), 0, 0,
                                                                                0, 0, std::pow(m_std_weight_position*kf.statePre.at<float>(2, 0), 2), 0,
                                                                                0, 0, 0, std::pow(m_std_weight_position*kf.statePre.at<float>(3, 0), 2));

        output_mean = kf.measurementMatrix * kf.statePre;
        output_covariance = kf.measurementMatrix * kf.errorCovPre * kf.measurementMatrix.t() + innovation_cov;
    }

    void BotsortMotionPrediction::_bbox2xcycwh(const BBOX &bbox, std::vector<float> &output) {
        output.clear();
        output.push_back(bbox.x + bbox.width / 2);
        output.push_back(bbox.y + bbox.height / 2);
        output.push_back(bbox.width);
        output.push_back(bbox.height);
    }

    void BotsortMotionPrediction::_xcycwh2bbox(float xc, float yc, float w, float h, BBOX &output_bbox) {
        output_bbox.height = h;
        output_bbox.width = w;
        output_bbox.x = xc - w / 2.0;
        output_bbox.y = yc - h / 2.0;
    }

    MotionPredictionByKalmanPtr BotsortMotionPrediction::clone() const {
        auto output = std::make_shared<BotsortMotionPrediction>();
        copy_to(*output);
        return output;
    }

    void BotsortMotionPrediction::copy_to(MotionPredictionByKalman &to) const {
        auto p = dynamic_cast<BotsortMotionPrediction*>(&to);
        assert_throw(p, "failed to convert MotionPredictionByKalman to BotsortMotionPrediction");
        MotionPredictionByKalman::copy_to(to);
        p->m_std_weight_position = m_std_weight_position;
        p->m_update_measurement = m_update_measurement;
        p->m_std_weight_velocity = m_std_weight_velocity;
    }
}