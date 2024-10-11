//
// Created by wangjing on 1/11/22.
//

#include "RedoxiTrack/tracker/KalmanTracker.h"
#include "RedoxiTrack/utils/utility_functions.h"

namespace RedoxiTrack {
    void KalmanTracker::init(const TrackerParam &param) {
        m_param = param.clone();
    }

    const TrackerParam *KalmanTracker::get_tracker_param() const {
        return TrackerBase::get_tracker_param();
    }

    void KalmanTracker::set_tracker_param(const TrackerParam &param) {

    }

    // Track the first frame. should always be called first
    void KalmanTracker::begin_track(const cv::Mat &img,
                                    const std::vector<DetectionPtr> &detections,
                                    int frame_number) {
        m_id2target.clear();
        _update_frame_number(frame_number);
        for (size_t i = 0; i < detections.size(); i++) {
            TrackTargetPtr track_target_ptr = create_target(detections[i], frame_number);
            add_target(track_target_ptr);
        }
    }

    void KalmanTracker::finish_track() {
        std::vector<int> delete_ids;
        for (auto &p : m_id2target) {
            delete_ids.push_back(p.first);
        }
        for(auto del_id: delete_ids){
            TrackingEvent::TargetClosed event_data = TrackingEvent::TargetClosed();
            event_data.m_target = m_id2target[del_id];
            for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                auto res = (*iter)->evt_target_closed_before(this, event_data);
            }

            m_id2target.erase(del_id);

            for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                auto res = (*iter)->evt_target_closed_after(this, event_data);
            }
        }
        m_frame_number = INIT_TRACKING_FRAME;
        m_path_id_for_generate_unique_id = 0;
    }

    void KalmanTracker::update_kalman(TrackTargetPtr& target, const BBOX &bbox) {
        KalmanTrackTargetPtr n_single_kalman_target = dynamic_pointer_cast<KalmanTrackTarget>(target);
        assert_throw(n_single_kalman_target->m_can_be_update, "failed kalman target can not be update, please predict before update");
        m_motion_predict->update(n_single_kalman_target->get_kf(), bbox);
        BBOX n_temp_bbox;
        m_motion_predict->get_bbox_state(n_single_kalman_target->get_kf(), n_temp_bbox);
        n_single_kalman_target->set_bbox(n_temp_bbox);
        n_single_kalman_target->m_can_be_update = false;
    }

    void KalmanTracker::_kalman_predict(TrackTargetPtr &target, const int &delta_frame_number, BBOX &output_bbox) {
        auto kalman_target = dynamic_cast<KalmanTrackTarget*>(target.get());

        auto& kf = kalman_target->get_kf();

        kf.transitionMatrix.at<float>(0,4) = delta_frame_number;
        kf.transitionMatrix.at<float>(1,5) = delta_frame_number;
        kf.transitionMatrix.at<float>(2,6) = delta_frame_number;
        kf.transitionMatrix.at<float>(3,7) = delta_frame_number;
        // kalman filter predict m_id2target

        m_motion_predict->predict(kf, output_bbox, delta_frame_number,
                                  kalman_target->get_path_state() == TrackPathStateBitmask::Lost);
        kalman_target->m_can_be_update = true;
    }

    void KalmanTracker::track(const cv::Mat &img, const std::vector<DetectionPtr> &detections, int frame_number) {

        // delete lost trackers
        std::vector<int> delete_id;
        for (auto &p : m_id2target) {
            auto time_since_update = frame_number - p.second->get_end_frame_number();
            if (time_since_update > m_param->m_max_time_since_update)
                delete_id.push_back(p.first);
        }
        for (auto &p : delete_id) {
            TrackingEvent::TargetClosed event_data = TrackingEvent::TargetClosed();
            event_data.m_target = m_id2target[p];
            for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                auto res = (*iter)->evt_target_closed_before(this, event_data);
            }

            m_id2target.erase(p);

            for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                auto res = (*iter)->evt_target_closed_after(this, event_data);
            }
        }

        // first motion prediction
        int delta_frame_number = frame_number - m_frame_number;
        for(auto &p: m_id2target) {
            _motion_predict(delta_frame_number, p.second, false);
        }

        _update_frame_number(frame_number);

        if (m_id2target.empty()) {
            for (size_t i = 0; i < detections.size(); i++) {
                TrackTargetPtr track_target_ptr = create_target(detections[i], frame_number);
                add_target(track_target_ptr);
            }
            return;
        }

        if (detections.empty()) {
            return;
        }

//        std::map<int, KalmanTrackTargetPtr> kalman_track_target;
//        _trans_target2kalman_target((*id2target_ptr), kalman_track_target);
        // calculate iou
        auto n_det_predict = m_id2target.size();
        auto n_det_now = detections.size();
        std::vector<KalmanTrackTargetPtr> targets; //all previous targets
        for (auto &p : m_id2target) {
            auto kalman_target = dynamic_pointer_cast<KalmanTrackTarget>(p.second);
            targets.push_back(kalman_target);
        }
        std::vector<std::vector<float>> dist_matrix_now2prev(n_det_now, std::vector<float>(n_det_predict, 0));

        for (size_t i = 0; i < detections.size(); i++) {
            for (size_t j = 0; j < targets.size(); j++) {
                auto bbox_after_predict = targets[j]->get_bbox();
                auto bbox_now = detections[i]->get_bbox();
                auto iou = compute_iou(bbox_now, bbox_after_predict);
                dist_matrix_now2prev[i][j] = 1 - iou;
            }
        }

        // match detection and target after predict
        std::vector<std::pair<int, int>> matched_pair;
        std::vector<int> unmatched_detection_now;
        std::vector<int> unmatched_detection_predict;
        hungarian_match(dist_matrix_now2prev, n_det_now, n_det_predict, m_param->m_max_iou_distance, matched_pair,
                        unmatched_detection_now, unmatched_detection_predict);

        //update tracker state
            // update matched detection_now and detection_predict
            for (size_t i = 0; i < matched_pair.size(); i++) {
                auto single_target = targets[matched_pair[i].second];
                auto single_detection = detections[matched_pair[i].first];

                TrackingEvent::TargetAssociation event_data = TrackingEvent::TargetAssociation();
                event_data.m_detection = single_detection;
                event_data.m_target = single_target;
                for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                    auto res = (*iter)->evt_target_association_before(this, event_data);
                }

                // update kalman filter
                TrackTargetPtr temp_p = single_target;
                update_kalman(temp_p, single_detection->get_bbox());
                single_target->set_path_state(TrackPathStateBitmask::Open);
                single_target->set_end_frame_number(frame_number);

                for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                    auto res = (*iter)->evt_target_association_after(this, event_data);
                }
                // update tracking decision
            }

            // create target from detection
            for (size_t i = 0; i < unmatched_detection_now.size(); i++) {
                TrackTargetPtr track_target_ptr = create_target(
                        detections[unmatched_detection_now[i]], frame_number);
                add_target(track_target_ptr);
            }


        return;
    }

    // motion prediction results. will change and delete m_id2target.
    void KalmanTracker::track(const cv::Mat &img, int frame_number) {
        assert_throw(m_frame_number != INIT_TRACKING_FRAME, "m frame number is INIT_TRACKING_FRAME");
        assert_throw(m_frame_number <= frame_number, "frame number less than m frame number");

        int delta_frame_number = frame_number - m_frame_number;
        for (auto &p : m_id2target) {

            _motion_predict(delta_frame_number, p.second, true);
        }

        _update_frame_number(frame_number);
    }

    void KalmanTracker::_motion_predict(int delta_frame_number, TrackTargetPtr &target, bool notify_event_handler){
        assert_throw(m_frame_number != INIT_TRACKING_FRAME, "m frame number is INIT_TRACKING_FRAME");
        assert_throw(delta_frame_number >= 0, "frame number less than m frame number");

        BBOX single_target_predict_bbox;
        _kalman_predict(target, delta_frame_number, single_target_predict_bbox);

        if(notify_event_handler){
            TrackingEvent::TargetMotionPredict event_data = TrackingEvent::TargetMotionPredict();
            event_data.m_target = target;
            for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                auto res = (*iter)->evt_target_motion_predict_before(this, event_data);
            }

            target->set_bbox(single_target_predict_bbox);

            for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                auto res = (*iter)->evt_target_motion_predict_after(this, event_data);
            }
        }
        else{
            target->set_bbox(single_target_predict_bbox);
        }
    }

    const std::map<int, TrackTargetPtr> &KalmanTracker::get_all_open_targets() const {
        return m_id2target;
    }

    TrackTargetPtr KalmanTracker::get_open_target(int path_id) const {
        return TrackerBase::get_open_target(path_id);
    }

    void KalmanTracker::add_target(const TrackTargetPtr &target) {
        TrackingEvent::TargetAssociation event_data = TrackingEvent::TargetAssociation();
        event_data.m_detection = target->get_underlying_detection();
        event_data.m_target = target;
        for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
            auto res = (*iter)->evt_target_created_before(this, event_data);
        }

        m_id2target[target->get_path_id()] = target;

        for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
            auto res = (*iter)->evt_target_created_after(this, event_data);
        }
    }

    TrackTargetPtr KalmanTracker::create_target(const DetectionPtr &det, int frame_number) {
        KalmanTrackTargetPtr kalman_target_ptr = std::make_shared<KalmanTrackTarget>();
        m_motion_predict->init(kalman_target_ptr->get_kf(), det->get_bbox());
        TrackTargetPtr output = kalman_target_ptr;
        output->set_underlying_detection(det, true);
        output->set_start_frame_number(frame_number);
        output->set_end_frame_number(frame_number);
        output->set_path_id(_generate_path_id());
        output->set_path_state(TrackPathStateBitmask::New);
        return output;
    }

    TrackerTrackingStatePtr KalmanTracker::_tracking_state_create() {
        auto output = std::make_shared<KalmanTrackingState>();
        return output;
    }

    void KalmanTracker::_tracking_state_fill(TrackerTrackingState &state) {
        auto p = dyncast_with_check<KalmanTrackingState>(&state);
        TrackerBase::_tracking_state_fill(state);
        p->m_motion_predict = m_motion_predict->clone();
    }

    void KalmanTracker::_tracking_state_recover(const TrackerTrackingState &state) {
        auto p = dyncast_with_check<KalmanTrackingState>(&state);
        TrackerBase::_tracking_state_recover(state);
        p->m_motion_predict->copy_to(*m_motion_predict);
    }

    void KalmanTracker::add_event_handler(const TrackingEventHandlerPtr& handler) {
        m_event_handlers.insert(handler);

    }

    void KalmanTracker::remove_event_handler(const TrackingEventHandlerPtr& handler) {
        m_event_handlers.erase(handler);
    }

    void KalmanTracker::delete_target(int path_id) {
        m_id2target.erase(path_id);
        }

        void KalmanTracker::delete_all_targets() {

        }
}