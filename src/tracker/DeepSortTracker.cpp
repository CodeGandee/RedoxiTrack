//
// Created by wangjing on 1/11/22.
//
#include "RedoxiTrack/tracker/DeepSortTracker.h"
#include "RedoxiTrack/tracker/DeepSortTrackerParam.h"
#include "RedoxiTrack/utils/CosineFeature.h"
#include "RedoxiTrack/utils/utility_functions.h"

#define MAX_COST_MATRIX_NUM 9999

namespace RedoxiTrack
{
void DeepSortTracker::init(const TrackerParam &param)
{
    m_param = param.clone();
    auto p = dynamic_cast<DeepSortTrackerParam *>(m_param.get());
    assert_throw(p, "TrackerParam type is wrong.");

    m_optical_flow_tracker = std::make_shared<OpticalFlowTracker>();
    m_optical_flow_tracker->init(p->get_optical_param());

    m_kalman_tracker = std::make_shared<KalmanTracker>();
    m_kalman_tracker->init(p->get_kalman_param());

    m_optical_flow_handler = std::make_shared<OpticalFlowEventHandler>();
    m_kalman_handler = std::make_shared<KalmanEventHandler>();

    m_optical_flow_tracker->add_event_handler(m_optical_flow_handler);
    m_kalman_tracker->add_event_handler(m_kalman_handler);

    m_feature_traits = std::make_shared<CosineFeature>();
    m_detection_comparision = std::make_shared<DefaultDetectionTraits>(this);
}

const TrackerParam *DeepSortTracker::get_tracker_param() const
{
    return TrackerBase::get_tracker_param();
}

void DeepSortTracker::set_tracker_param(const TrackerParam &param)
{
    assert_throw(false, "not implemented");
}

void DeepSortTracker::set_detection_comparision(
    const DetectionTraitsPtr &hander)
{
    m_detection_comparision = hander;
}

void DeepSortTracker::reset_detection_comparision()
{
    m_detection_comparision = std::make_shared<DefaultDetectionTraits>(this);
}

void DeepSortTracker::begin_track(const cv::Mat &img,
                                  const std::vector<DetectionPtr> &detections,
                                  int frame_number)
{
    m_id2target.clear();

    _update_frame_number(frame_number);

    m_optical_flow_handler->clear();
    m_optical_flow_tracker->begin_track(img, detections, frame_number);
    m_kalman_handler->clear();
    m_kalman_tracker->begin_track(img, detections, frame_number);
    for (auto det : detections) {
        auto optical_target = m_optical_flow_handler->m_det2target_create[det];
        auto kalman_target = m_kalman_handler->m_det2target_create[det];

        TrackTargetPtr deepsort_target =
            create_target(det, frame_number, kalman_target, optical_target);
        add_target(deepsort_target);
    }
}

void DeepSortTracker::finish_track()
{
    m_optical_flow_tracker->finish_track();
    m_kalman_tracker->finish_track();

    std::vector<int> delete_ids;
    for (auto &p : m_id2target) {
        delete_ids.push_back(p.first);
    }

    for (auto del_id : delete_ids) {
        TrackingEvent::TargetClosed event_data = TrackingEvent::TargetClosed();
        event_data.m_target = m_id2target[del_id];
        for (auto iter = m_event_handlers.begin();
             iter != m_event_handlers.end(); iter++) {
            (*iter)->evt_target_closed_before(this, event_data);
        }

        m_id2target.erase(del_id);

        for (auto iter = m_event_handlers.begin();
             iter != m_event_handlers.end(); iter++) {
            (*iter)->evt_target_closed_after(this, event_data);
        }
    }
    m_frame_number = INIT_TRACKING_FRAME;
    m_path_id_for_generate_unique_id = 0;
}

void DeepSortTracker::track(const cv::Mat &img,
                            const std::vector<DetectionPtr> &detections,
                            int frame_number)
{
    assert_throw(m_frame_number != INIT_TRACKING_FRAME,
                 "m frame number is INIT_TRACKING_FRAME");
    assert_throw(m_frame_number <= frame_number,
                 "m frame number less than frame number");

    // delete removed tracker
    _remove_targets(frame_number);

    std::vector<TrackTargetPtr> targets;
    auto p_param = dynamic_cast<DeepSortTrackerParam *>(m_param.get());
    if (p_param->m_use_optical_before_track) {
        _motion_predict(img, m_id2target, frame_number);
        for (auto &p : m_id2target) {
            auto &single_deepsort_target = p.second;
            targets.push_back(single_deepsort_target);
        }
    } else {
        // first kalman predict, set deepsort bbox, delete untracked tracker
        m_kalman_handler->clear();
        m_kalman_tracker->track(img, frame_number);

        for (auto &p : m_id2target) {
            auto &single_deepsort_target = p.second;
            targets.push_back(single_deepsort_target);
            auto kalman_target = dynamic_cast<DeepSortTrackTarget *>(
                                     single_deepsort_target.get())
                                     ->m_kalman_target;
            single_deepsort_target->set_bbox(kalman_target->get_bbox());
        }
    }
    _update_frame_number(frame_number);

    // second embedding and maha matching
    std::vector<std::pair<int, int>> matched_pair;
    std::vector<int> unmatched_detection_now;
    std::vector<int> unmatched_detection_predict;
    _match_maha_distance(detections, targets, matched_pair,
                         unmatched_detection_now, unmatched_detection_predict);

    // update tracker state
    //  update matched detection_now and detection_predict
    for (size_t i = 0; i < matched_pair.size(); i++) {
        _update_target(targets[matched_pair[i].second],
                       detections[matched_pair[i].first], frame_number);
    }

    // third iou matching
    std::vector<TrackTargetPtr> iou_targets; // targets used for iou matching
    std::vector<DetectionPtr>
        iou_detections; // detections used for iou matching
    for (auto p : unmatched_detection_predict) {
        if (targets[p]->get_path_state() == TrackPathStateBitmask::New ||
            targets[p]->get_path_state() == TrackPathStateBitmask::Open)
            iou_targets.push_back(targets[p]);
    }
    for (auto p : unmatched_detection_now) {
        iou_detections.push_back(detections[p]);
    }
    // calculate iou distance
    std::vector<std::pair<int, int>> matched_iou_pair;
    std::vector<int> unmatched_iou_detection_now;
    std::vector<int> unmatched_iou_detection_predict;
    _match_iou_distance(iou_detections, iou_targets, matched_iou_pair,
                        unmatched_iou_detection_now,
                        unmatched_iou_detection_predict);
    // update tracker state
    //  update matched detection_now and detection_predict
    for (size_t i = 0; i < matched_iou_pair.size(); i++) {
        _update_target(iou_targets[matched_iou_pair[i].second],
                       iou_detections[matched_iou_pair[i].first], frame_number);
    }

    // fourth init new tracker
    for (auto p : unmatched_iou_detection_now) {
        // init kalman
        TrackTargetPtr kalman_track_target_ptr =
            m_kalman_tracker->create_target(iou_detections[p], frame_number);
        TrackTargetPtr optical_track_target_ptr =
            m_optical_flow_tracker->create_target(iou_detections[p],
                                                  frame_number);
        m_kalman_tracker->add_target(kalman_track_target_ptr);
        m_optical_flow_tracker->add_target(optical_track_target_ptr);

        // init deepsort
        TrackTargetPtr track_target_ptr =
            create_target(iou_detections[p], frame_number,
                          kalman_track_target_ptr, optical_track_target_ptr);
        add_target(track_target_ptr);
    }
}

void DeepSortTracker::track(const cv::Mat &img, int frame_number)
{
    assert_throw(m_frame_number != INIT_TRACKING_FRAME,
                 "m frame number is INIT_TRACKING_FRAME");
    assert_throw(m_frame_number <= frame_number,
                 "frame number less than m frame number");

    _remove_targets(frame_number);

    m_optical_flow_handler->clear();
    m_optical_flow_tracker->track(img, frame_number);
    m_kalman_handler->clear();
    m_kalman_tracker->track(img, frame_number);

    // update kalman filter and deepsort tracker
    for (auto &p : m_id2target) {
        auto &temp_target = p.second;
        auto single_deepsort_target =
            dynamic_pointer_cast<DeepSortTrackTarget>(temp_target);
        auto single_kalman_target = single_deepsort_target->m_kalman_target;
        auto single_optical_target = single_deepsort_target->m_optical_target;

        TrackingEvent::TargetMotionPredict event_data =
            TrackingEvent::TargetMotionPredict();
        event_data.m_target = single_deepsort_target;
        for (auto iter = m_event_handlers.begin();
             iter != m_event_handlers.end(); iter++) {
            (*iter)->evt_target_motion_predict_before(this, event_data);
        }

        m_kalman_tracker->update_kalman(single_kalman_target,
                                        single_optical_target->get_bbox());

        single_deepsort_target->set_bbox(single_kalman_target->get_bbox());
        _update_features(single_deepsort_target,
                         m_id2target[p.first]->get_feature());

        for (auto iter = m_event_handlers.begin();
             iter != m_event_handlers.end(); iter++) {
            (*iter)->evt_target_motion_predict_after(this, event_data);
        }
    }

    // delete removed tracker
    _update_frame_number(frame_number);
}

void DeepSortTracker::_motion_predict(const cv::Mat &img,
                                      std::map<int, TrackTargetPtr> &id2target,
                                      int frame_number)
{
    assert_throw(m_frame_number != INIT_TRACKING_FRAME,
                 "m frame number is INIT_TRACKING_FRAME");
    assert_throw(m_frame_number <= frame_number,
                 "frame number less than m frame number");

    m_optical_flow_handler->clear();
    m_optical_flow_tracker->track(img, frame_number);

    m_kalman_handler->clear();
    m_kalman_tracker->track(img, frame_number);
    m_kalman_tracker
        ->push_tracking_state(); // track predict! but not change update state

    // update kalman filter and deepsort tracker
    for (auto &p : id2target) {
        auto single_deepsort_target =
            dynamic_pointer_cast<DeepSortTrackTarget>(p.second);
        auto single_kalman_target = single_deepsort_target->m_kalman_target;
        auto single_optical_target = single_deepsort_target->m_optical_target;
        if (single_deepsort_target->get_path_state() ==
            TrackPathStateBitmask::Lost) {
            continue; // LOST continue,
        }

        m_kalman_tracker->update_kalman(single_kalman_target,
                                        single_optical_target->get_bbox());

        single_deepsort_target->set_bbox(single_kalman_target->get_bbox());
        _update_features(single_deepsort_target,
                         (id2target)[p.first]->get_feature());
    }
    m_kalman_tracker->pop_tracking_state();
}

void DeepSortTracker::_update_features(DeepSortTrackTargetPtr &target,
                                       const fVECTOR &features)
{
    auto p = dynamic_cast<DeepSortTrackerParam *>(m_param.get());
    fVECTOR new_feature;
    m_feature_traits->linear_combine(&new_feature, target->get_feature(),
                                     features, p->m_alpha_smooth_features,
                                     (1 - p->m_alpha_smooth_features));
    target->set_feature(new_feature);
}

void DeepSortTracker::_bbox2xyah(const BBOX &bbox, cv::Mat &output)
{
    output = cv::Mat::zeros(4, 1, CV_32F);
    output.at<float>(0, 0) = bbox.x + bbox.width / 2.0;
    output.at<float>(1, 0) = bbox.y + bbox.height / 2.0;
    output.at<float>(2, 0) = bbox.width / bbox.height;
    output.at<float>(3, 0) = bbox.height;
}

const std::map<int, TrackTargetPtr> &
    DeepSortTracker::get_all_open_targets() const
{
    return m_id2target;
}

TrackTargetPtr DeepSortTracker::get_open_target(int path_id) const
{
    return TrackerBase::get_open_target(path_id);
}

void DeepSortTracker::delete_target(int path_id)
{
    m_id2target.erase(path_id);
}

void DeepSortTracker::delete_all_targets()
{
}

void DeepSortTracker::add_target(const TrackTargetPtr &target)
{
    TrackingEvent::TargetAssociation event_data =
        TrackingEvent::TargetAssociation();
    event_data.m_detection = target->get_underlying_detection();
    event_data.m_target = target;
    for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end();
         iter++) {
        (*iter)->evt_target_created_before(this, event_data);
    }

    m_id2target[target->get_path_id()] = target;

    for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end();
         iter++) {
        (*iter)->evt_target_created_after(this, event_data);
    }
}

TrackTargetPtr DeepSortTracker::create_target(const DetectionPtr &det,
                                              int frame_number)
{
    DeepSortTrackTargetPtr output = std::make_shared<DeepSortTrackTarget>();
    output->set_underlying_detection(det, true);
    output->set_start_frame_number(frame_number);
    output->set_end_frame_number(frame_number);
    output->set_path_id(_generate_path_id());
    output->set_path_state(TrackPathStateBitmask::New);
    return output;
}

TrackTargetPtr
    DeepSortTracker::create_target(const DetectionPtr &det, int frame_number,
                                   const TrackTargetPtr &kalman_target,
                                   const TrackTargetPtr &optical_target)
{
    auto output = create_target(det, frame_number);
    auto p = dyncast_with_check<DeepSortTrackTarget>(output.get());
    p->m_optical_target = optical_target;
    p->m_kalman_target = kalman_target;
    return output;
}

void DeepSortTracker::add_event_handler(const TrackingEventHandlerPtr &handler)
{
    m_event_handlers.insert(handler);
}

void DeepSortTracker::remove_event_handler(
    const TrackingEventHandlerPtr &handler)
{
    m_event_handlers.erase(handler);
}

void DeepSortTracker::_match_maha_distance(
    const std::vector<DetectionPtr> &sources,
    const std::vector<TrackTargetPtr> &targets,
    std::vector<std::pair<int, int>> &output_matched_pair,
    std::vector<int> &output_unmatched_source,
    std::vector<int> &output_unmatched_target)
{
    // calculate embedding distance
    auto n_det_now = sources.size();
    auto n_det_predict = targets.size();
    std::vector<std::vector<float>> dist_matrix_now2prev(
        n_det_now, std::vector<float>(n_det_predict, 0));

    bool sources_targets_feature_empty = true;
    // judge targets sources has feature or not
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

    if (sources_targets_feature_empty) {
        for (unsigned int i = 0; i < n_det_now; i++)
            output_unmatched_source.push_back(i);
        for (unsigned int i = 0; i < n_det_predict; i++)
            output_unmatched_target.push_back(i);
    } else {
        for (size_t i = 0; i < sources.size(); i++) {
            for (size_t j = 0; j < targets.size(); j++) {
                dist_matrix_now2prev[i][j] =
                    m_detection_comparision->compute_detection_distance(
                        targets[j].get(), sources[i].get());
            }
        }

        // calculate maha distance
        auto p_param = dynamic_cast<DeepSortTrackerParam *>(m_param.get());
        auto n_kalman_target = m_kalman_tracker->get_all_targets();
        for (size_t i = 0; i < sources.size(); i++) {
            for (size_t j = 0; j < targets.size(); j++) {
                auto single_id = targets[j]->get_path_id();
                cv::Mat kalman_mean, kalman_covariance;
                KalmanTrackTargetPtr n_single_kalman_target =
                    dynamic_pointer_cast<KalmanTrackTarget>(
                        n_kalman_target[single_id]);
                m_kalman_tracker->get_motion_prediction()
                    ->project_state2measurement(
                        n_single_kalman_target->get_kf(), kalman_mean,
                        kalman_covariance);
                BBOX n_det_bbox = sources[i]->get_bbox();

                // maha distance
                cv::Mat det_mean;
                _bbox2xyah(n_det_bbox, det_mean);
                cv::Mat invert_kalman_covariance;
                cv::invert(kalman_covariance, invert_kalman_covariance,
                           cv::DECOMP_SVD);
                double gating_dist =
                    std::pow(cv::Mahalanobis(det_mean, kalman_mean,
                                             invert_kalman_covariance),
                             2);
                if (gating_dist > p_param->get_gating_threshold())
                    dist_matrix_now2prev[i][j] = MAX_COST_MATRIX_NUM;
                dist_matrix_now2prev[i][j] =
                    p_param->m_gating_dist_lambda * dist_matrix_now2prev[i][j] +
                    (1 - p_param->m_gating_dist_lambda) * gating_dist;
            }
        }

        // match
        hungarian_match(dist_matrix_now2prev, n_det_now, n_det_predict,
                        p_param->m_max_gating_distance, output_matched_pair,
                        output_unmatched_source, output_unmatched_target);
    }
}

void DeepSortTracker::_match_iou_distance(
    const std::vector<DetectionPtr> &sources,
    const std::vector<TrackTargetPtr> &targets,
    std::vector<std::pair<int, int>> &output_matched_pair,
    std::vector<int> &output_unmatched_source,
    std::vector<int> &output_unmatched_target)
{
    auto n_det_now = sources.size();
    auto n_det_predict = targets.size();
    std::vector<std::vector<float>> dist_matrix_now2prev(
        n_det_now, std::vector<float>(n_det_predict, 0));

    for (size_t i = 0; i < sources.size(); i++) {
        for (size_t j = 0; j < targets.size(); j++) {
            auto bbox_after_predict = targets[j]->get_bbox();
            auto bbox_now = sources[i]->get_bbox();
            auto iou = compute_iou(bbox_now, bbox_after_predict);
            dist_matrix_now2prev[i][j] = 1 - iou;
        }
    }
    hungarian_match(dist_matrix_now2prev, n_det_now, n_det_predict,
                    m_param->m_max_iou_distance, output_matched_pair,
                    output_unmatched_source, output_unmatched_target);
}

void DeepSortTracker::_remove_targets(const int frame_number)
{
    std::vector<int> delete_id;
    std::vector<int> kalman_delete_id;
    std::vector<int> optical_delete_id;
    for (auto &p : m_id2target) {
        auto &single_deepsort_target = p.second;
        auto time_since_update =
            frame_number - single_deepsort_target->get_end_frame_number();
        if (time_since_update > m_param->m_max_time_since_update) {
            delete_id.push_back(p.first);
            auto kalman_target = dynamic_cast<DeepSortTrackTarget *>(
                                     single_deepsort_target.get())
                                     ->m_kalman_target;
            kalman_delete_id.push_back(kalman_target->get_path_id());

            auto optical_target = dynamic_cast<DeepSortTrackTarget *>(
                                      single_deepsort_target.get())
                                      ->m_optical_target;
            optical_delete_id.push_back(optical_target->get_path_id());
        }
    }
    for (auto &p : delete_id) {
        TrackingEvent::TargetClosed event_data = TrackingEvent::TargetClosed();
        event_data.m_target = m_id2target[p];
        for (auto iter = m_event_handlers.begin();
             iter != m_event_handlers.end(); iter++) {
            (*iter)->evt_target_closed_before(this, event_data);
        }

        m_id2target.erase(p);

        for (auto iter = m_event_handlers.begin();
             iter != m_event_handlers.end(); iter++) {
            (*iter)->evt_target_closed_after(this, event_data);
        }
    }

    for (auto &p : kalman_delete_id) {
        m_kalman_tracker->delete_target(p);
    }
    for (auto &p : optical_delete_id) {
        m_optical_flow_tracker->delete_target(p);
    }
}

void DeepSortTracker::_update_target(TrackTargetPtr &deepsort_target_ptr,
                                     const DetectionPtr &det,
                                     const int &frame_number)
{
    auto single_deepsort_target =
        dynamic_pointer_cast<DeepSortTrackTarget>(deepsort_target_ptr);
    auto single_detection = det;
    auto single_kalman_target = single_deepsort_target->m_kalman_target;
    auto single_optical_target = single_deepsort_target->m_optical_target;

    TrackingEvent::TargetAssociation event_data =
        TrackingEvent::TargetAssociation();
    event_data.m_detection = single_detection;
    event_data.m_target = single_deepsort_target;

    EventHandlerResultType event_handler_res = EventHandlerResultTypes::None;
    for (auto iter = m_event_handlers.begin(); iter != m_event_handlers.end();
         iter++) {
        auto res = (*iter)->evt_target_association_before(this, event_data);
        if (res > event_handler_res)
            event_handler_res = res;
    }

    if (event_handler_res == EventHandlerResultTypes::None) {
        // update kalman filter
        m_kalman_tracker->update_kalman(single_kalman_target,
                                        single_detection->get_bbox());

        single_deepsort_target->set_bbox(single_kalman_target->get_bbox());
        single_deepsort_target->set_end_frame_number(frame_number);
        single_kalman_target->set_end_frame_number(frame_number);
        _update_features(single_deepsort_target,
                         single_detection->get_feature());

        // optical tracker update
        single_optical_target->set_bbox(single_deepsort_target->get_bbox());
        single_optical_target->set_end_frame_number(frame_number);

        for (auto iter = m_event_handlers.begin();
             iter != m_event_handlers.end(); iter++) {
            (*iter)->evt_target_association_after(this, event_data);
        }
    } else if (event_handler_res ==
               EventHandlerResultTypes::Association_RejectAndCreateNew) {
        TrackTargetPtr optical_flow_track_target_ptr =
            m_optical_flow_tracker->create_target(single_detection,
                                                  frame_number);
        m_optical_flow_tracker->add_target(optical_flow_track_target_ptr);
        TrackTargetPtr kalman_track_target_ptr =
            m_kalman_tracker->create_target(single_detection, frame_number);
        m_kalman_tracker->add_target(kalman_track_target_ptr);
        TrackTargetPtr deepsort_track_target_ptr = create_target(
            single_detection, frame_number, kalman_track_target_ptr,
            optical_flow_track_target_ptr);
        add_target(deepsort_track_target_ptr);
    } else if (event_handler_res ==
               EventHandlerResultTypes::Association_RejectAndDiscard) {
    }
}

void DeepSortTracker::push_tracking_state()
{
    auto x = _tracking_state_create();
    _tracking_state_fill(*x);
    m_state_stack.push_back(x);
    m_optical_flow_tracker->push_tracking_state();
    m_kalman_tracker->push_tracking_state();
}

void DeepSortTracker::pop_tracking_state(bool apply)
{
    assert_throw(m_state_stack.size() > 0,
                 "failed pop tracking state, m_state_stack is empty");
    auto x = m_state_stack.back();
    m_state_stack.pop_back();
    if (apply)
        _tracking_state_recover(*x);
    m_kalman_tracker->pop_tracking_state(apply);
    m_optical_flow_tracker->pop_tracking_state(apply);
}

int DeepSortTracker::OpticalFlowEventHandler::evt_target_association_after(
    TrackerBase *sender, const TrackingEvent::TargetAssociation &evt_data)
{
    m_det2target_assiciate[evt_data.m_detection] = evt_data.m_target;
    return 0;
}

int DeepSortTracker::OpticalFlowEventHandler::evt_target_created_after(
    TrackerBase *sender, const TrackingEvent::TargetAssociation &evt_data)
{
    m_det2target_create[evt_data.m_detection] = evt_data.m_target;
    return 0;
}

int DeepSortTracker::OpticalFlowEventHandler::evt_target_closed_after(
    TrackerBase *sender, const TrackingEvent::TargetClosed &evt_data)
{
    m_target_close.push_back(evt_data.m_target);
    return 0;
}

int DeepSortTracker::OpticalFlowEventHandler::evt_target_motion_predict_after(
    TrackerBase *sender, const TrackingEvent::TargetMotionPredict &evt_data)
{
    m_target_motion_predict.push_back(evt_data.m_target);
    return 0;
}

int DeepSortTracker::KalmanEventHandler::evt_target_association_after(
    TrackerBase *sender, const TrackingEvent::TargetAssociation &evt_data)
{
    m_det2target_assiciate[evt_data.m_detection] = evt_data.m_target;
    return 0;
}

int DeepSortTracker::KalmanEventHandler::evt_target_created_after(
    TrackerBase *sender, const TrackingEvent::TargetAssociation &evt_data)
{
    m_det2target_create[evt_data.m_detection] = evt_data.m_target;
    return 0;
}

int DeepSortTracker::KalmanEventHandler::evt_target_motion_predict_after(
    TrackerBase *sender, const TrackingEvent::TargetMotionPredict &evt_data)
{
    m_target_motion_predict.push_back(evt_data.m_target);
    return 0;
}

void DeepSortTracker::set_feature_traits(const FeatureTraitsPtr &p)
{
    m_feature_traits = p;
}

FeatureTraitsPtr DeepSortTracker::get_feature_traits()
{
    return m_feature_traits;
}

FeatureTraitsPtr
    DeepSortTracker::DefaultDetectionTraits::get_feature_traits() const
{
    return m_tracker->get_feature_traits();
}

DeepSortTracker::DefaultDetectionTraits::DefaultDetectionTraits(
    DeepSortTracker *p)
    : m_tracker(p)
{
}
} // namespace RedoxiTrack