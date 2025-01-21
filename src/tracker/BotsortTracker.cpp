//
// Created by 001730 chengxiao on 22/8/31.
//
#include "RedoxiTrack/tracker/BotsortTracker.h"
#include "RedoxiTrack/utils/utility_functions.h"
#include <algorithm>
#define MAX_COST_MATRIX_NUM 9999

#define DEBUG 1

namespace RedoxiTrack
{
void BotsortTracker::init(const TrackerParam &param)
{
    m_param = param.clone();
    auto p = dynamic_cast<BotsortTrackerParam *>(m_param.get());
    assert_throw(p, "TrackerParam type is wrong.");

    if (p->get_optical_param()) {
        m_optical_flow_tracker = std::make_shared<OpticalFlowTracker>();
        m_optical_flow_tracker->init(*p->get_optical_param());
    } else {
        m_optical_flow_tracker = nullptr;
    }


    m_kalman_tracker = std::make_shared<BotsortKalmanTracker>();
    m_kalman_tracker->init(*p->get_kalman_param());

    if (p->get_optical_param()) {
        m_optical_flow_handler = std::make_shared<OpticalFlowEventHandler>();
        m_optical_flow_tracker->add_event_handler(m_optical_flow_handler);
    } else {
        m_optical_flow_handler = nullptr;
    }

    m_kalman_handler = std::make_shared<KalmanEventHandler>();
    m_kalman_tracker->add_event_handler(m_kalman_handler);

    m_feature_traits = std::make_shared<CosineFeature>();
    m_detection_comparision = std::make_shared<DefaultDetectionTraits>(this);
}

const TrackerParam *BotsortTracker::get_tracker_param() const
{
    return TrackerBase::get_tracker_param();
}

void BotsortTracker::set_tracker_param(const TrackerParam &param)
{
    assert_throw(false, "not implemented");
}

void BotsortTracker::set_detection_comparision(const DetectionTraitsPtr &hander)
{
    m_detection_comparision = hander;
}

void BotsortTracker::reset_detection_comparision()
{
    m_detection_comparision = std::make_shared<DefaultDetectionTraits>(this);
}

void BotsortTracker::begin_track(const cv::Mat &img,
                                 const std::vector<DetectionPtr> &detections,
                                 int frame_number)
{
    m_id2target.clear();

    _update_frame_number(frame_number);

    auto p_param = dynamic_cast<BotsortTrackerParam *>(m_param.get());

    std::vector<DetectionPtr> new_detections;
    for (auto det : detections) {
        if (det->get_confidence() >= p_param->m_new_track_thresh)
            new_detections.push_back(det);
    }
    if (m_optical_flow_tracker) {
        m_optical_flow_handler->clear();
        m_optical_flow_tracker->begin_track(img, new_detections, frame_number);
    }
    m_kalman_handler->clear();
    m_kalman_tracker->begin_track(img, new_detections, frame_number);

    for (auto det : new_detections) {
        TrackTargetPtr optical_target = nullptr;
        if (m_optical_flow_tracker) {
            optical_target = m_optical_flow_handler->m_det2target_create[det];
        }
        auto kalman_target = m_kalman_handler->m_det2target_create[det];

        TrackTargetPtr botsort_target = create_target(det, frame_number, kalman_target, optical_target);
        auto single_botsort_target = dynamic_pointer_cast<BotsortTrackTarget>(botsort_target);
        single_botsort_target->m_is_activated = true;
        add_target(botsort_target);
    }
}

void BotsortTracker::finish_track()
{
    if (m_optical_flow_tracker) {
        m_optical_flow_tracker->finish_track();
    }
    m_kalman_tracker->finish_track();

    std::vector<int> delete_ids;
    for (auto &p : m_id2target) {
        delete_ids.push_back(p.first);
    }

    for (auto del_id : delete_ids) {
        TrackingEvent::TargetClosed event_data = TrackingEvent::TargetClosed();
        event_data.m_target = m_id2target[del_id];
        for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++) {
            (*iter)->evt_target_closed_before(this, event_data);
        }

        m_id2target.erase(del_id);

        for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++) {
            (*iter)->evt_target_closed_after(this, event_data);
        }
    }
    // 重置frame_number
    m_frame_number = INIT_TRACKING_FRAME;
    m_path_id_for_generate_unique_id = 0;

    // 重置m_lost_targets
    m_lost_targets.clear();
    // 重置m_removed_targets
    m_removed_targets.clear();
    // 重置m_tracked_targets
    m_tracked_targets.clear();
}

void BotsortTracker::track(const cv::Mat &img, const std::vector<DetectionPtr> &detections, int frame_number)
{
    assert_throw(m_frame_number != INIT_TRACKING_FRAME, "m frame number is INIT_TRACKING_FRAME");
    assert_throw(m_frame_number <= frame_number, "m frame number less than frame number");

#if DEBUG
    std::cout << "--------------------------------- frame id " << frame_number + 1 << " ---------------------------------" << std::endl;

    std::cout << "1. self.tracked_stracks" << std::endl;
    for (auto &p : m_tracked_targets) {
        // auto& single_botsort_target = p.second;
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.second.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        // std::cout << std::endl;
        // std::cout << p.second->get_feature() << std::endl;
    }
    std::cout << "2. self.lost_stracks" << std::endl;
    for (auto &p : m_lost_targets) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.second.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        // std::cout << std::endl;
        // std::cout << single_botsort_target->get_feature() << std::endl;
    }
    std::cout << "3. self.removed_stracks" << std::endl;
    for (auto &p : m_removed_targets) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.second.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        // std::cout << std::endl;
        // std::cout << single_botsort_target->get_feature() << std::endl;
    }
#endif

    std::vector<TrackTargetPtr> activated;
    std::vector<TrackTargetPtr> refind;
    std::vector<TrackTargetPtr> lost;
    std::vector<TrackTargetPtr> removed;

    // filter detections by low score thresh and high
    std::vector<DetectionPtr> detections_low_conf_filter;
    auto p_param = dynamic_cast<BotsortTrackerParam *>(m_param.get());
    for (auto &det : detections) {
        if (det->get_confidence() > p_param->m_track_low_thresh) {
            detections_low_conf_filter.push_back(det);
        }
    }

    // find high score detections and low scroe detections
    std::vector<DetectionPtr> detections_high;
    std::vector<DetectionPtr> detections_low;
    for (auto &det : detections_low_conf_filter) {
        if (det->get_confidence() > p_param->m_track_high_thresh) {
            detections_high.push_back(det);
        } else {
            // low detection not update track object's feature
            auto single_detection_ptr = dynamic_pointer_cast<SingleDetection>(det);
            fVECTOR x;
            single_detection_ptr->set_feature(x);
            detections_low.push_back(det);
        }
    }

#if DEBUG
    std::cout << "4. HIGH SCORE DETECTIONS" << std::endl;
    for (auto &p : detections_high) {
        auto &single_botsort_target = p;
        std::cout << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        // std::cout << std::endl;
        // std::cout << single_botsort_target->get_feature() << std::endl;
    }
#endif

    // Add newly detected tracklets to tracked_stracks
    std::vector<TrackTargetPtr> tracked_targets; // targets not include lost None close targets
    std::vector<TrackTargetPtr> unconfirmed;     // targets include lost targets
    for (auto &p : m_tracked_targets) {
        auto &botsort_target = p.second;
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.second.get());
        if (single_botsort_target->m_is_activated)
            tracked_targets.push_back(botsort_target);
        else
            unconfirmed.push_back(botsort_target);
    }

    // STEP1 : first association with high score detection bboxes
    std::vector<TrackTargetPtr> target_pool(tracked_targets);
    std::vector<TrackTargetPtr> kalman_target_pool;

    for (auto &t : m_lost_targets) {
        target_pool.push_back(t.second);
    }

    for (auto &t : target_pool) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(t.get());
        kalman_target_pool.push_back(single_botsort_target->m_kalman_target);
    }

#if DEBUG
    std::cout << "5. unconfirmed" << std::endl;
    for (auto &p : unconfirmed) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        // std::cout << std::endl;
        // std::cout << single_botsort_target->get_feature() << std::endl;
    }

    std::cout << "6. strack_pool" << std::endl;
    for (auto &p : target_pool) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        // std::cout << std::endl;
        // std::cout << single_botsort_target->get_feature() << std::endl;
    }
#endif

    // kalman target predict first
    if (p_param->m_use_optical_before_track) {
        _motion_predict(img, target_pool, frame_number);
    } else {
        // first kalman predict, set botsort bbox, delete untracked tracker
        m_kalman_handler->clear();
        m_kalman_tracker->track(img, kalman_target_pool, frame_number);
        // m_kalman_tracker->track(img, frame_number);

        for (auto &p : target_pool) {
            auto &botsort_target = p;
            auto kalman_target = dynamic_cast<BotsortTrackTarget *>(botsort_target.get())->m_kalman_target;
            botsort_target->set_bbox(kalman_target->get_bbox());
        }
    }
    _update_frame_number(frame_number);

#if DEBUG
    std::cout << "7. after multi_predict strack_pool" << std::endl;
    for (auto &p : target_pool) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        std::cout << std::endl;
        // std::cout << single_botsort_target->get_feature() << std::endl;
    }
#endif

    std::vector<std::pair<int, int>> matched_pair;
    std::vector<int> unmatched_detection_now;
    std::vector<int> unmatched_detection_predict;
    _match_maha_distance(detections_high, target_pool, p_param->m_match_thresh, matched_pair, unmatched_detection_now, unmatched_detection_predict);

#if DEBUG
    std::cout << "11. first matched" << std::endl;
    for (auto &match : matched_pair) {
        std::cout << match.first << "," << match.second << std::endl;
    }
    std::cout << "12. first u_track" << std::endl;
    for (auto &match : unmatched_detection_predict) {
        std::cout << match << ",";
    }
    std::cout << std::endl;
    std::cout << "13. first u_detection" << std::endl;
    for (auto &match : unmatched_detection_now) {
        std::cout << match << ",";
    }
    std::cout << std::endl;

    std::cout << "14. low score detections" << std::endl;
    for (auto &p : detections_low) {
        auto single_botsort_target = p;
        std::cout << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        // std::cout << std::endl;
        // std::cout << single_botsort_target->get_feature() << std::endl;
    }
#endif

    // update tracker state and traklet feature
    // update matched detection_now and detection_predict
    for (size_t i = 0; i < matched_pair.size(); i++) {
        _update_target(target_pool[matched_pair[i].second], detections_high[matched_pair[i].first], frame_number, true, activated, refind);
    }

    // STEP2 : second association with low score detection bboxes by iou matching
    std::vector<TrackTargetPtr> first_unmatched_track; // from first association unmatched targets
    for (auto p : unmatched_detection_predict) {
        if (target_pool[p]->get_path_state() == TrackPathStateBitmask::Open)
            first_unmatched_track.push_back(target_pool[p]);
    }

    // calculate iou distance
    std::vector<std::pair<int, int>> second_matched_pair;
    std::vector<int> unmatched_iou_detection_now;
    std::vector<int> unmatched_iou_detection_predict;
    _match_iou_distance(detections_low, first_unmatched_track, 0.5, second_matched_pair, unmatched_iou_detection_now, unmatched_iou_detection_predict);

#if DEBUG
    std::cout << "16. second matched" << std::endl;
    for (auto &match : second_matched_pair) {
        std::cout << match.first << "," << match.second << std::endl;
    }
    std::cout << "17. second u_track" << std::endl;
    for (auto &match : unmatched_iou_detection_predict) {
        std::cout << match << ",";
    }
    std::cout << std::endl;
    std::cout << "18. second u_detection" << std::endl;
    for (auto &match : unmatched_iou_detection_now) {
        std::cout << match << ",";
    }
    std::cout << std::endl;
#endif

    // update matched track targets
    for (size_t i = 0; i < second_matched_pair.size(); i++) {
        _update_target(first_unmatched_track[second_matched_pair[i].second], detections_low[second_matched_pair[i].first], frame_number, true, activated, refind);
    }

    // update unmatched track targets
    for (size_t i = 0; i < unmatched_iou_detection_predict.size(); i++) {
        auto botsort_target_ptr = first_unmatched_track[unmatched_iou_detection_predict[i]];
        auto single_botsort_target = dynamic_pointer_cast<BotsortTrackTarget>(botsort_target_ptr);
        if (single_botsort_target->get_path_state() != TrackPathStateBitmask::Lost) {
            single_botsort_target->set_path_state(TrackPathStateBitmask::Lost);
            single_botsort_target->m_kalman_target->set_path_state(TrackPathStateBitmask::Lost);
            lost.push_back(single_botsort_target);
        }
    }

    // STEP3 : third association with unconfirmed tracks and unmatched detection bboxes in first association
    std::vector<DetectionPtr> unmatched_first_detections;
    for (auto p : unmatched_detection_now) {
        unmatched_first_detections.push_back(detections_high[p]);
    }

    std::vector<std::pair<int, int>> matched_third_pair;
    std::vector<int> unmatched_detection_third;
    std::vector<int> unmatched_track_third;
    _match_maha_distance(unmatched_first_detections, unconfirmed, 0.7, matched_third_pair, unmatched_detection_third, unmatched_track_third);

#if DEBUG
    std::cout << "22. third matched" << std::endl;
    for (auto &match : matched_third_pair) {
        std::cout << match.first << "," << match.second << std::endl;
    }
    std::cout << "23. third u_track" << std::endl;
    for (auto &match : unmatched_track_third) {
        std::cout << match << ",";
    }
    std::cout << std::endl;
    std::cout << "24. third u_detection" << std::endl;
    for (auto &match : unmatched_detection_third) {
        std::cout << match << ",";
    }
    std::cout << std::endl;
#endif

    // update matched track targets
    for (size_t i = 0; i < matched_third_pair.size(); i++) {
        _update_target(unconfirmed[matched_third_pair[i].second], unmatched_first_detections[matched_third_pair[i].first], frame_number, false, activated, refind);
    }

    // update third unmatched track target
    for (size_t i = 0; i < unmatched_track_third.size(); i++) {
        auto botsort_target_ptr = unconfirmed[unmatched_track_third[i]];
        auto single_botsort_target = dynamic_pointer_cast<BotsortTrackTarget>(botsort_target_ptr);
        single_botsort_target->set_path_state(TrackPathStateBitmask::Close);
        single_botsort_target->m_kalman_target->set_path_state(TrackPathStateBitmask::Close);
        removed.push_back(single_botsort_target);
    }

    // STEP4 : init new tracker
    for (auto p : unmatched_detection_third) {
        if (unmatched_first_detections[p]->get_confidence() < p_param->m_new_track_thresh)
            continue;
        // init kalman
        TrackTargetPtr kalman_track_target_ptr = m_kalman_tracker->create_target(unmatched_first_detections[p], frame_number);
        m_kalman_tracker->add_target(kalman_track_target_ptr);
        TrackTargetPtr optical_track_target_ptr = nullptr;
        if (m_optical_flow_tracker) {
            optical_track_target_ptr = m_optical_flow_tracker->create_target(unmatched_first_detections[p],
                                                                            frame_number);
            m_optical_flow_tracker->add_target(optical_track_target_ptr);
        }

        // init botsort
        TrackTargetPtr track_target_ptr = create_target(unmatched_first_detections[p], frame_number,
                                                        kalman_track_target_ptr,
                                                        optical_track_target_ptr);
        add_target(track_target_ptr);
        activated.push_back(track_target_ptr);
    }

    // STEP5 : update state
    for (auto &l : m_lost_targets) {
        if (m_frame_number - l.second->get_end_frame_number() > p_param->m_max_time_lost) {
            l.second->set_path_state(TrackPathStateBitmask::Close);
            dynamic_cast<BotsortTrackTarget *>(l.second.get())->m_kalman_target->set_path_state(TrackPathStateBitmask::Close);
            removed.push_back(l.second);
        }
    }

#if DEBUG
    std::cout << "25. activated_starcks" << std::endl;
    for (auto &p : activated) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        // std::cout << std::endl;
        // std::cout << single_botsort_target->get_feature() << std::endl;
    }
    std::cout << "26. refind_stracks" << std::endl;
    for (auto &p : refind) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        // std::cout << std::endl;
        // std::cout << single_botsort_target->get_feature() << std::endl;
    }
    std::cout << "27. lost_stracks" << std::endl;
    for (auto &p : lost) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        // std::cout << std::endl;
        // std::cout << single_botsort_target->get_feature() << std::endl;
    }
    std::cout << "28. removed_stracks" << std::endl;
    for (auto &p : removed) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        // std::cout << std::endl;
        // std::cout << single_botsort_target->get_feature() << std::endl;
    }
#endif

    // STEP6 : merge
    std::vector<int> del_id;
    for (auto &target : m_tracked_targets) {
        auto tid = target.first;
        auto track_ptr = target.second;
        if (track_ptr->get_path_state() != TrackPathStateBitmask::Open) {
            del_id.push_back(tid);
        }
    }
    for (auto &tid : del_id) {
        m_tracked_targets.erase(tid);
    }

    for (auto &target : activated) {
        if (m_tracked_targets.count(target->get_path_id()) == 0) {
            m_tracked_targets[target->get_path_id()] = target;
        }
    }
    for (auto &target : refind) {
        if (m_tracked_targets.count(target->get_path_id()) == 0) {
            m_tracked_targets[target->get_path_id()] = target;
        }
    }

    for (auto &target : m_tracked_targets) {
        auto tid = target.first;
        auto track_ptr = target.second;
        if (m_lost_targets.count(tid) != 0) {
            m_lost_targets.erase(tid);
        }
    }
    for (auto &target : lost) {
        m_lost_targets[target->get_path_id()] = target;
    }

    for (auto &target : m_removed_targets) {
        auto tid = target.first;
        auto track_ptr = target.second;
        if (m_lost_targets.count(tid) != 0) {
            m_lost_targets.erase(tid);
        }
    }

#if DEBUG
    std::cout << "29. before _remove_duplicate_targets m_tracked_targets" << std::endl;
    for (auto &p : m_tracked_targets) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.second.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
    }
    std::cout << "30. before _remove_duplicate_targets m_lost_targets" << std::endl;
    for (auto &p : m_lost_targets) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.second.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
    }
#endif
    _remove_duplicate_targets();
#if DEBUG
    std::cout << "31. after _remove_duplicate_targets m_lost_targets" << std::endl;
    for (auto &p : m_lost_targets) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.second.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
    }
#endif
    // 在remove targets调用完之后再加到m_removed_targets中
    _remove_targets(removed);

    for (auto &target : removed) {
        m_removed_targets[target->get_path_id()] = target;
    }

#if DEBUG
    std::cout << "32. output" << std::endl;
    for (auto &p : m_tracked_targets) {
        auto single_botsort_target = dyncast_with_check<BotsortTrackTarget>(p.second.get());
        std::cout << single_botsort_target->get_path_id() << "," << single_botsort_target->get_bbox().x << "," << single_botsort_target->get_bbox().y << "," << single_botsort_target->get_bbox().width << "," << single_botsort_target->get_bbox().height << "," << single_botsort_target->m_is_activated << std::endl;
        // for (int i = 0; i < 128; i++) {
        //     std::cout << single_botsort_target->get_feature()(i, 0) << ",";
        // }
        // std::cout << std::endl;
        // std::cout << p.second->get_feature() << std::endl;
    }
#endif
}

void BotsortTracker::track(const cv::Mat &img, int frame_number)
{
    assert_throw(m_frame_number != INIT_TRACKING_FRAME, "m frame number is INIT_TRACKING_FRAME");
    assert_throw(m_frame_number <= frame_number, "frame number less than m frame number");

    if (m_optical_flow_tracker && m_optical_flow_handler) {
        m_optical_flow_handler->clear();
        m_optical_flow_tracker->track(img, frame_number);
    }
    m_kalman_handler->clear();
    m_kalman_tracker->KalmanTracker::track(img, frame_number);

    // update kalman filter and botsort tracker
    for (auto &p : m_id2target) {
        auto &temp_target = p.second;
        auto single_botsort_target = dynamic_pointer_cast<BotsortTrackTarget>(temp_target);
        auto single_kalman_target = single_botsort_target->m_kalman_target;
        auto single_optical_target = single_botsort_target->m_optical_target;

        TrackingEvent::TargetMotionPredict event_data = TrackingEvent::TargetMotionPredict();
        event_data.m_target = single_botsort_target;
        if (m_tracked_targets.count(single_botsort_target->get_path_id()) != 0) {
            for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++) {
                (*iter)->evt_target_motion_predict_before(this, event_data);
            }
        }
        if (single_optical_target) {  // 用光流预测的结果更新kalman, 若没有光流则不更新kalman滤波器
            m_kalman_tracker->KalmanTracker::update_kalman(single_kalman_target, single_optical_target->get_bbox());
        }

        single_botsort_target->set_bbox(single_kalman_target->get_bbox());
        // if (m_id2target[p.first]->get_feature().size() != 0)
        //     _update_features(single_botsort_target, m_id2target[p.first]->get_feature());

        if (m_tracked_targets.count(single_botsort_target->get_path_id()) != 0) {
            for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++) {
                (*iter)->evt_target_motion_predict_after(this, event_data);
            }
        }
    }

    // delete removed tracker
    _update_frame_number(frame_number);
}

void BotsortTracker::_motion_predict(const cv::Mat &img, std::vector<TrackTargetPtr> &target_pool,
                                     int frame_number)
{
    assert_throw(m_frame_number != INIT_TRACKING_FRAME, "m frame number is INIT_TRACKING_FRAME");
    assert_throw(m_frame_number <= frame_number, "frame number less than m frame number");
    assert_throw(!img.empty(), "img is empty");
    assert_throw(m_optical_flow_tracker != nullptr, "m_optical_flow_tracker is nullptr");

    auto p_param = dynamic_cast<BotsortTrackerParam *>(m_param.get());

    m_optical_flow_handler->clear();
    m_optical_flow_tracker->track(img, frame_number);

    m_kalman_handler->clear();
    m_kalman_tracker->KalmanTracker::track(img, frame_number);
    // m_kalman_tracker->track(img, frame_number);
    m_kalman_tracker->push_tracking_state(); // track predict! but not change update state

    // update kalman filter and botsort tracker
    for (auto &p : target_pool) {
        auto single_botsort_target = dynamic_pointer_cast<BotsortTrackTarget>(p);
        auto single_kalman_target = single_botsort_target->m_kalman_target;
        auto single_optical_target = single_botsort_target->m_optical_target;
        // if(single_botsort_target->get_path_state() == TrackPathStateBitmask::Lost){
        //     continue; // LOST continue,
        // }

        m_kalman_tracker->KalmanTracker::update_kalman(single_kalman_target, single_optical_target->get_bbox());

        single_botsort_target->set_bbox(single_kalman_target->get_bbox());
        if (p_param->m_use_reid_feature) {
            if (p->get_feature().size() != 0)
                _update_features(single_botsort_target, p->get_feature());
        }
    }
    m_kalman_tracker->pop_tracking_state();
}

void BotsortTracker::_update_features(BotsortTrackTargetPtr &target, const fVECTOR &features)
{
    auto p = dynamic_cast<BotsortTrackerParam *>(m_param.get());

    fVECTOR new_feature;
    m_feature_traits->linear_combine(&new_feature, target->get_feature(), features,
                                     p->m_alpha_smooth_features, (1 - p->m_alpha_smooth_features));
    target->set_feature(new_feature);
}

void BotsortTracker::_bbox2xcycwh(const BBOX &bbox, cv::Mat &output)
{
    output = cv::Mat::zeros(4, 1, CV_32F);
    output.at<float>(0, 0) = bbox.x + bbox.width / 2.0;
    output.at<float>(1, 0) = bbox.y + bbox.height / 2.0;
    output.at<float>(2, 0) = bbox.width;
    output.at<float>(3, 0) = bbox.height;
}

const std::map<int, TrackTargetPtr> &BotsortTracker::get_all_open_targets() const
{
    return m_tracked_targets;
}

TrackTargetPtr BotsortTracker::get_open_target(int path_id) const
{
    return TrackerBase::get_open_target(path_id);
}

void BotsortTracker::delete_target(int path_id)
{
    // if (m_id2target.count(path_id) != 0) {
    //     auto single_botsort_target = dynamic_pointer_cast<BotsortTrackTarget>(m_id2target[path_id]);
    //     m_kalman_tracker->delete_target(single_botsort_target->m_kalman_target->get_path_id());
    //     m_optical_flow_tracker->delete_target(single_botsort_target->m_kalman_target->get_path_id());
    // }
    m_id2target.erase(path_id);
}

void BotsortTracker::delete_all_targets()
{
}

void BotsortTracker::add_target(const TrackTargetPtr &target)
{
    TrackingEvent::TargetAssociation event_data = TrackingEvent::TargetAssociation();
    event_data.m_detection = target->get_underlying_detection();
    event_data.m_target = target;
    for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++) {
        (*iter)->evt_target_created_before(this, event_data);
    }

    m_id2target[target->get_path_id()] = target;
    m_tracked_targets[target->get_path_id()] = target;

    for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++) {
        (*iter)->evt_target_created_after(this, event_data);
    }
}

TrackTargetPtr BotsortTracker::create_target(const DetectionPtr &det, int frame_number)
{
    BotsortTrackTargetPtr output = std::make_shared<BotsortTrackTarget>();
    output->set_underlying_detection(det, true);
    output->set_start_frame_number(frame_number);
    output->set_end_frame_number(frame_number);
    output->set_path_id(_generate_path_id());
    output->set_path_state(TrackPathStateBitmask::Open);
    return output;
}

TrackTargetPtr BotsortTracker::create_target(const DetectionPtr &det, int frame_number,
                                             const TrackTargetPtr &kalman_target, const TrackTargetPtr &optical_target)
{
    auto output = create_target(det, frame_number);
    auto p = dyncast_with_check<BotsortTrackTarget>(output.get());
    p->m_optical_target = optical_target;  // can be nullptr
    p->m_kalman_target = kalman_target;
    return output;
}

void BotsortTracker::add_event_handler(const TrackingEventHandlerPtr &handler)
{
    m_event_handlers.insert(handler);
}

void BotsortTracker::remove_event_handler(const TrackingEventHandlerPtr &handler)
{
    m_event_handlers.erase(handler);
}

void BotsortTracker::_match_maha_distance(const std::vector<DetectionPtr> &sources,
                                          const std::vector<TrackTargetPtr> &targets,
                                          const float match_thresh,
                                          std::vector<std::pair<int, int>> &output_matched_pair,
                                          std::vector<int> &output_unmatched_source,
                                          std::vector<int> &output_unmatched_target)
{
    // calculate iou distance
    size_t n_det_now = sources.size();
    size_t n_det_predict = targets.size();
    std::vector<std::vector<float>> dist_matrix_iou(n_det_now, std::vector<float>(n_det_predict, 0));
    std::vector<std::vector<int>> dist_matrix_iou_mask(n_det_now, std::vector<int>(n_det_predict, 0));

    auto p_param = dynamic_cast<BotsortTrackerParam *>(m_param.get());

#if DEBUG
    std::cout << "ious_dists" << std::endl;
#endif
    for (size_t i = 0; i < sources.size(); i++) {
        for (size_t j = 0; j < targets.size(); j++) {
            auto bbox_after_predict = targets[j]->get_bbox();
            auto bbox_now = sources[i]->get_bbox();
            auto iou = compute_iou(bbox_now, bbox_after_predict);
            dist_matrix_iou[i][j] = 1 - iou;
#if DEBUG
            std::cout << dist_matrix_iou[i][j] << ",";
#endif
            if (dist_matrix_iou[i][j] > p_param->m_proximity_thresh) {
                dist_matrix_iou_mask[i][j] = 1;
            }
        }
#if DEBUG
        std::cout << std::endl;
#endif
    }

    // fuse confidence and iou
    if (p_param->m_fuse_score)
        _fuse_score(dist_matrix_iou, sources);

    // calculate embedding distance
    std::vector<std::vector<float>> dist_matrix_now2prev(n_det_now, std::vector<float>(n_det_predict, 0));

    bool sources_targets_feature_empty = true;
    // judge targets sources has feature or not
    if (p_param->m_use_reid_feature) {
        for (size_t i = 0; i < sources.size(); i++) {
            auto feature = sources[i]->get_feature();
            if (feature.size() != 0) {
                sources_targets_feature_empty = false;
                break;
            }
        }
        for (size_t i = 0; i < targets.size(); i++) {
            auto feature = targets[i]->get_feature();
            if (feature.size() != 0) {
                sources_targets_feature_empty = false;
                break;
            }
        }
    }


    // no id feature, match by iou distance
    if (sources_targets_feature_empty) {
        dist_matrix_now2prev = dist_matrix_iou;
    } else {
        auto p_param = dynamic_cast<BotsortTrackerParam *>(m_param.get());

#if DEBUG
        std::cout << "cosine dists between strack_pool and HIGH SCORE DETECTIONS" << std::endl;
#endif

        // get distance matrix by combine iou distance and cosine distance
        for (size_t i = 0; i < sources.size(); i++) {
            for (size_t j = 0; j < targets.size(); j++) {
                auto cosine_dis = m_detection_comparision->compute_detection_distance(targets[j].get(), sources[i].get());
                dist_matrix_now2prev[i][j] = cosine_dis > p_param->m_appearance_thresh ? 1.0 : cosine_dis;
                if (dist_matrix_iou_mask[i][j] == 1) {
                    dist_matrix_now2prev[i][j] = 1.0;
                }

                dist_matrix_now2prev[i][j] = min(dist_matrix_iou[i][j], dist_matrix_now2prev[i][j]);
#if DEBUG
                std::cout << cosine_dis << ",";
#endif
            }
#if DEBUG
            std::cout << std::endl;
#endif
        }

        // python版本暂时没用到这个
        // // calculate maha distance
        // auto p_param = dynamic_cast<botsortTrackerParam *>(m_param.get());
        // auto n_kalman_target = m_kalman_tracker->get_all_targets();
        // for (size_t i = 0; i < sources.size(); i++) {
        //     for (size_t j = 0; j < targets.size(); j++) {
        //         auto single_id = targets[j]->get_path_id();
        //         cv::Mat kalman_mean, kalman_covariance;
        //         KalmanTrackTargetPtr n_single_kalman_target = dynamic_pointer_cast<KalmanTrackTarget>(
        //                 n_kalman_target[single_id]);
        //         m_kalman_tracker->get_motion_prediction()->project_state2measurement(
        //                 n_single_kalman_target->get_kf(), kalman_mean, kalman_covariance);
        //         BBOX n_det_bbox = sources[i]->get_bbox();

        //         // maha distance
        //         cv::Mat det_mean;
        //         _bbox2xyah(n_det_bbox, det_mean);
        //         cv::Mat invert_kalman_covariance;
        //         cv::invert(kalman_covariance, invert_kalman_covariance, cv::DECOMP_SVD);
        //         double gating_dist = std::pow(cv::Mahalanobis(det_mean, kalman_mean, invert_kalman_covariance), 2);
        //         if (gating_dist > p_param->m_gating_threshold)
        //             dist_matrix_now2prev[i][j] = MAX_COST_MATRIX_NUM;
        //         dist_matrix_now2prev[i][j] = p_param->m_gating_dist_lambda * dist_matrix_now2prev[i][j] +
        //                                      (1 - p_param->m_gating_dist_lambda) * gating_dist;
        //     }
        // }
    }

    // match
    std::cout << "before lapjv match" << std::endl;
    lapjv_match(dist_matrix_now2prev, n_det_now, n_det_predict, match_thresh,
                output_matched_pair,
                output_unmatched_source, output_unmatched_target);

#if DEBUG
    std::cout << "feature dists" << std::endl;
    for (size_t i = 0; i < sources.size(); i++) {
        for (size_t j = 0; j < targets.size(); j++) {
            std::cout << dist_matrix_now2prev[i][j] << ",";
        }
        std::cout << std::endl;
    }
#endif
}

void BotsortTracker::_match_iou_distance(const std::vector<DetectionPtr> &sources,
                                         const std::vector<TrackTargetPtr> &targets,
                                         const float match_thresh,
                                         std::vector<std::pair<int, int>> &output_matched_pair,
                                         std::vector<int> &output_unmatched_source,
                                         std::vector<int> &output_unmatched_target)
{
    auto n_det_now = sources.size();
    auto n_det_predict = targets.size();
    std::vector<std::vector<float>> dist_matrix_now2prev(n_det_now, std::vector<float>(n_det_predict, 0));

#if DEBUG
    std::cout << "15. ious_dists between u_track and low score detections" << std::endl;
#endif
    for (size_t i = 0; i < sources.size(); i++) {
        for (size_t j = 0; j < targets.size(); j++) {
            auto bbox_after_predict = targets[j]->get_bbox();
            auto bbox_now = sources[i]->get_bbox();
            auto iou = compute_iou(bbox_now, bbox_after_predict);
            dist_matrix_now2prev[i][j] = 1 - iou;
#if DEBUG
            std::cout << dist_matrix_now2prev[i][j] << ",";
#endif
        }
#if DEBUG
        std::cout << std::endl;
#endif
    }
    lapjv_match(dist_matrix_now2prev, n_det_now, n_det_predict, match_thresh, output_matched_pair,
                output_unmatched_source, output_unmatched_target);
}

void BotsortTracker::_fuse_score(std::vector<std::vector<float>> &dist_matrix_iou,
                                 const std::vector<DetectionPtr> &detections)
{
    if (dist_matrix_iou.size() == 0)
        return;
#if DEBUG
    std::cout << "fuse_score ious_dists" << std::endl;
#endif
    for (size_t i = 0; i < dist_matrix_iou.size(); i++) {
        for (size_t j = 0; j < dist_matrix_iou[0].size(); j++) {
            dist_matrix_iou[i][j] = 1 - (1 - dist_matrix_iou[i][j]) * detections[i]->get_confidence();
#if DEBUG
            std::cout << dist_matrix_iou[i][j] << ",";
#endif
        }
#if DEBUG
        std::cout << std::endl;
#endif
    }
}

void BotsortTracker::_remove_duplicate_targets()
{
    std::vector<TrackTargetPtr> targetsa;
    std::vector<TrackTargetPtr> targetsb;
    std::set<int> dupa;
    std::set<int> dupb;

    for (auto &p : m_tracked_targets) {
        auto &single_botsort_target = p.second;
        targetsa.push_back(single_botsort_target);
    }
    for (auto &p : m_lost_targets) {
        auto &single_botsort_target = p.second;
        targetsb.push_back(single_botsort_target);
    }

    // calculate iou distance
    auto n_a = targetsa.size();
    auto n_b = targetsb.size();
    std::vector<std::vector<float>> dist_matrix_iou(n_a, std::vector<float>(n_b, 0));

    for (size_t i = 0; i < n_a; i++) {
        for (size_t j = 0; j < n_b; j++) {
            auto bbox_a = targetsa[i]->get_bbox();
            auto bbox_b = targetsb[j]->get_bbox();
            auto iou = compute_iou(bbox_a, bbox_b);
            dist_matrix_iou[i][j] = 1 - iou;
            if (dist_matrix_iou[i][j] < 0.15) {
                auto timea = targetsa[i]->get_end_frame_number() - targetsa[i]->get_start_frame_number();
                auto timeb = targetsb[j]->get_end_frame_number() - targetsb[j]->get_start_frame_number();

                if (timea > timeb) {
                    dupb.insert(targetsb[j]->get_path_id());
                } else {
                    dupa.insert(targetsa[i]->get_path_id());
                }
            }
        }
    }

    std::vector<int> del_a;
    std::vector<int> del_b;
    for (auto &p : m_tracked_targets) {
        auto &tid = p.first;
        if (dupa.count(tid) == 1)
            del_a.push_back(tid);
    }
    for (auto &p : m_lost_targets) {
        auto &tid = p.first;
        if (dupb.count(tid) == 1)
            del_b.push_back(tid);
    }

    for (auto &tid : del_a) {
        // TrackingEvent::TargetClosed event_data = TrackingEvent::TargetClosed();
        // event_data.m_target = m_tracked_targets[tid];

        m_tracked_targets.erase(tid);
        m_removed_targets[tid] = m_id2target[tid];

        // // 放在closed target中，在外层调用跟踪函数并获取create target和associate target时进行过滤
        // for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
        //     auto res = (*iter)->evt_target_closed_after(this, event_data);
        // }
    }
    for (auto &tid : del_b) {
        m_lost_targets.erase(tid);
        m_removed_targets[tid] = m_id2target[tid];
    }
}

void BotsortTracker::_remove_targets(vector<TrackTargetPtr> &removed)
{
    std::vector<int> delete_id;
    std::vector<int> kalman_delete_id;
    std::vector<int> optical_delete_id;
    // 对所有m_lost_targets存在m_removed_targets ID的对象删除
    for (auto &p : m_removed_targets) {
        if (m_lost_targets.count(p.first) != 0) {
            m_lost_targets.erase(p.first);
        }
        // 删除不存在于m_tracked_targets的m_removed_targets
        auto &single_botsort_target = p.second;
        if (m_tracked_targets.count(p.first) == 0) {
            auto temp = std::find(removed.begin(), removed.end(), p.second);
            if (temp != removed.end())
                removed.erase(temp);
            delete_id.push_back(p.first);
            auto kalman_target = dynamic_cast<BotsortTrackTarget *>(single_botsort_target.get())->m_kalman_target;
            kalman_delete_id.push_back(kalman_target->get_path_id());

            auto optical_target = dynamic_cast<BotsortTrackTarget *>(single_botsort_target.get())->m_optical_target;
            if (optical_target) {  // 如果没用光流则为nullptr
                optical_delete_id.push_back(optical_target->get_path_id());
            }
        }
    }
    for (auto &p : delete_id) {
        TrackingEvent::TargetClosed event_data = TrackingEvent::TargetClosed();
        event_data.m_target = m_id2target[p];
        for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++) {
            (*iter)->evt_target_closed_before(this, event_data);
        }
        m_id2target.erase(p);
        m_removed_targets.erase(p);

        for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++) {
            (*iter)->evt_target_closed_after(this, event_data);
        }
    }

    for (auto &p : kalman_delete_id) {
        m_kalman_tracker->delete_target(p);
    }
    if (m_optical_flow_tracker) {
        for (auto &p : optical_delete_id) {
            m_optical_flow_tracker->delete_target(p);
        }
    }
}

void BotsortTracker::_update_target(TrackTargetPtr &botsort_target_ptr, const DetectionPtr &det, const int &frame_number, bool add_refind,
                                    std::vector<TrackTargetPtr> &activated, std::vector<TrackTargetPtr> &refind)
{
    // type conversion
    auto single_botsort_target = dynamic_pointer_cast<BotsortTrackTarget>(botsort_target_ptr);
    auto single_detection = det;
    auto single_kalman_target = single_botsort_target->m_kalman_target;
    auto single_optical_target = single_botsort_target->m_optical_target;  // can be nullptr

    TrackingEvent::TargetAssociation event_data = TrackingEvent::TargetAssociation();
    event_data.m_detection = single_detection;
    event_data.m_target = single_botsort_target;

    EventHandlerResultType event_handler_res = EventHandlerResultTypes::None;
    for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++) {
        auto res = (*iter)->evt_target_association_before(this, event_data);
        if (res > event_handler_res)
            event_handler_res = res;
    }

    if (event_handler_res == EventHandlerResultTypes::None) {
        if (add_refind) {
            // update kalman filter, include state post update
            m_kalman_tracker->KalmanTracker::update_kalman(single_kalman_target, single_detection->get_bbox());
            if (single_botsort_target->get_path_state() == TrackPathStateBitmask::Open) {
                activated.push_back(single_botsort_target);
            } else {
                single_botsort_target->set_path_state(TrackPathStateBitmask::Open);
                single_botsort_target->m_kalman_target->set_path_state(TrackPathStateBitmask::Open);
                refind.push_back(single_botsort_target);
            }
        } else {
            // update kalman filter, include state post update
            m_kalman_tracker->update_kalman(single_kalman_target, single_detection->get_bbox(), 1);
            activated.push_back(single_botsort_target);
        }

        single_botsort_target->set_bbox(single_kalman_target->get_bbox());
        single_botsort_target->set_end_frame_number(frame_number);
        single_botsort_target->set_path_state(TrackPathStateBitmask::Open);
        single_botsort_target->set_quality(single_detection->get_quality());
        single_botsort_target->m_is_activated = true;
        single_kalman_target->set_end_frame_number(frame_number);

        auto p_param = dynamic_cast<BotsortTrackerParam *>(m_param.get());
        if (p_param->m_use_reid_feature) {
            if (single_detection->get_feature().size() != 0)
                _update_features(single_botsort_target, single_detection->get_feature());
        }

        // optical tracker update
        if (single_optical_target) {
            single_optical_target->set_bbox(single_botsort_target->get_bbox());
            single_optical_target->set_end_frame_number(frame_number);
        }

        for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++) {
            (*iter)->evt_target_association_after(this, event_data);
        }
    } else if (event_handler_res == EventHandlerResultTypes::Association_RejectAndCreateNew) {
        TrackTargetPtr optical_flow_track_target_ptr = nullptr;
        if (m_optical_flow_tracker) {
            optical_flow_track_target_ptr = m_optical_flow_tracker->create_target(single_detection, frame_number);
            m_optical_flow_tracker->add_target(optical_flow_track_target_ptr);
        }
        TrackTargetPtr kalman_track_target_ptr = m_kalman_tracker->create_target(single_detection, frame_number);
        m_kalman_tracker->add_target(kalman_track_target_ptr);
        TrackTargetPtr botsort_track_target_ptr = create_target(single_detection, frame_number,
                                                                kalman_track_target_ptr,
                                                                optical_flow_track_target_ptr);
        add_target(botsort_track_target_ptr);

    } else if (event_handler_res == EventHandlerResultTypes::Association_RejectAndDiscard) {
    }
}

void BotsortTracker::push_tracking_state()
{
    auto x = _tracking_state_create();
    _tracking_state_fill(*x);
    m_state_stack.push_back(x);
    if (m_optical_flow_tracker)
        m_optical_flow_tracker->push_tracking_state();
    m_kalman_tracker->push_tracking_state();
}

void BotsortTracker::pop_tracking_state(bool apply)
{
    assert_throw(m_state_stack.size() > 0, "failed pop tracking state, m_state_stack is empty");
    auto x = m_state_stack.back();
    m_state_stack.pop_back();
    if (apply)
        _tracking_state_recover(*x);
    m_kalman_tracker->pop_tracking_state(apply);
    if (m_optical_flow_tracker)
        m_optical_flow_tracker->pop_tracking_state(apply);
}


int BotsortTracker::OpticalFlowEventHandler::evt_target_association_after(TrackerBase *sender,
                                                                          const TrackingEvent::TargetAssociation &evt_data)
{
    m_det2target_assiciate[evt_data.m_detection] = evt_data.m_target;
    return 0;
}

int BotsortTracker::OpticalFlowEventHandler::evt_target_created_after(TrackerBase *sender,
                                                                      const TrackingEvent::TargetAssociation &evt_data)
{
    m_det2target_create[evt_data.m_detection] = evt_data.m_target;
    return 0;
}

int BotsortTracker::OpticalFlowEventHandler::evt_target_closed_after(TrackerBase *sender,
                                                                     const TrackingEvent::TargetClosed &evt_data)
{
    m_target_close.push_back(evt_data.m_target);
    return 0;
}

int BotsortTracker::OpticalFlowEventHandler::evt_target_motion_predict_after(TrackerBase *sender,
                                                                             const TrackingEvent::TargetMotionPredict &evt_data)
{
    m_target_motion_predict.push_back(evt_data.m_target);
    return 0;
}

int BotsortTracker::KalmanEventHandler::evt_target_association_after(TrackerBase *sender,
                                                                     const TrackingEvent::TargetAssociation &evt_data)
{
    m_det2target_assiciate[evt_data.m_detection] = evt_data.m_target;
    return 0;
}

int BotsortTracker::KalmanEventHandler::evt_target_created_after(TrackerBase *sender,
                                                                 const TrackingEvent::TargetAssociation &evt_data)
{
    m_det2target_create[evt_data.m_detection] = evt_data.m_target;
    return 0;
}

int BotsortTracker::KalmanEventHandler::evt_target_motion_predict_after(TrackerBase *sender,
                                                                        const TrackingEvent::TargetMotionPredict &evt_data)
{
    m_target_motion_predict.push_back(evt_data.m_target);
    return 0;
}

void BotsortTracker::set_feature_traits(const FeatureTraitsPtr &p)
{
    m_feature_traits = p;
}

FeatureTraitsPtr BotsortTracker::get_feature_traits()
{
    return m_feature_traits;
}

FeatureTraitsPtr BotsortTracker::DefaultDetectionTraits::get_feature_traits() const
{
    return m_tracker->get_feature_traits();
}

BotsortTracker::DefaultDetectionTraits::DefaultDetectionTraits(BotsortTracker *p)
    : m_tracker(p)
{
}
} // namespace RedoxiTrack