//
// Created by 001730 chengxiao on 1/3/23.
//

#include "RedoxiTrack/tracker/BotsortKalmanTracker.h"
#include "RedoxiTrack/detection/Detection.h"
#include "RedoxiTrack/detection/TrackTarget.h"
#include "RedoxiTrack/RedoxiTrackConfig.h"
#include <assert.h>
#include <memory>

namespace RedoxiTrack {
    void BotsortKalmanTracker::track(const cv::Mat &img,
                    const std::vector<DetectionPtr>& targets,
                    int frame_number) {
        // assert_throw(m_frame_number != INIT_TRACKING_FRAME, "m frame number is INIT_TRACKING_FRAME");
        // assert_throw(m_frame_number <= frame_number, "frame number less than m frame number");

        // int delta_frame_number = frame_number - m_frame_number;
        // for (auto &p : targets) {
        //     auto _p = std::dynamic_pointer_cast<TrackTarget>(p);
        //     _motion_predict(delta_frame_number, _p, true);
        // }

        // _update_frame_number(frame_number);

        std::vector<TrackTargetPtr> _targets;
        for(auto& p : targets)
        {
            auto _p = std::dynamic_pointer_cast<TrackTarget>(p);
            assert_throw(_p != nullptr, "failed to cast to TrackTargetPtr");
            _targets.push_back(_p);
        }
        this->track(img, _targets, frame_number);
    }

    void BotsortKalmanTracker::track(const cv::Mat &img,
                    const std::vector<TrackTargetPtr>& targets,
                    int frame_number) {
        assert_throw(m_frame_number != INIT_TRACKING_FRAME, "m frame number is INIT_TRACKING_FRAME");
        assert_throw(m_frame_number <= frame_number, "frame number less than m frame number");

        int delta_frame_number = frame_number - m_frame_number;
        for (auto p : targets) {
            _motion_predict(delta_frame_number, p, true);
        }

        _update_frame_number(frame_number);
    }

    void BotsortKalmanTracker::update_kalman(TrackTargetPtr& target, const BBOX &bbox, int delta_frame_number) {
        KalmanTrackTargetPtr n_single_kalman_target = std::dynamic_pointer_cast<KalmanTrackTarget>(target);
        auto& kf = n_single_kalman_target->get_kf();
        // assert_throw(n_single_kalman_target->m_can_be_update, "failed kalman target can not be update, please predict before update");
        if (!n_single_kalman_target->m_can_be_update) {
            kf.statePost.copyTo(kf.statePre);
            kf.errorCovPost.copyTo(kf.errorCovPre);
            kf.processNoiseCov.at<float>(0, 0) = std::pow(1.0 / 20.0 * kf.statePost.at<float>(2, 0) * delta_frame_number, 2);
            kf.processNoiseCov.at<float>(1, 1) = std::pow(1.0 / 20.0 * kf.statePost.at<float>(3, 0) * delta_frame_number, 2);
            kf.processNoiseCov.at<float>(2, 2) = std::pow(1.0 / 20.0 * kf.statePost.at<float>(2, 0) * delta_frame_number, 2);
            kf.processNoiseCov.at<float>(3, 3) = std::pow(1.0 / 20.0 * kf.statePost.at<float>(3, 0) * delta_frame_number, 2);
            kf.processNoiseCov.at<float>(4, 4) = std::pow(1.0 / 160.0 * kf.statePost.at<float>(2, 0) * delta_frame_number, 2);
            kf.processNoiseCov.at<float>(5, 5) = std::pow(1.0 / 160.0 * kf.statePost.at<float>(3, 0) * delta_frame_number, 2);
            kf.processNoiseCov.at<float>(6, 6) = std::pow(1.0 / 160.0 * kf.statePost.at<float>(2, 0) * delta_frame_number, 2);
            kf.processNoiseCov.at<float>(7, 7) = std::pow(1.0 / 160.0 * kf.statePost.at<float>(3, 0) * delta_frame_number, 2);
        }
        get_motion_prediction()->update(n_single_kalman_target->get_kf(), bbox);
        BBOX n_temp_bbox;
        get_motion_prediction()->get_bbox_state(n_single_kalman_target->get_kf(), n_temp_bbox);
        n_single_kalman_target->set_bbox(n_temp_bbox);
        n_single_kalman_target->m_can_be_update = false;
    }
}