#include "RedoxiTrack/tracker/OpticalFlowTracker.h"
#include "RedoxiTrack/utils/utility_functions.h"
#include "RedoxiTrack/tracker/TrackingEventHandler.h"

#define IOU_DISTANCE_MAX 10

namespace RedoxiTrack
{
    void OpticalFlowTracker::init(const TrackerParam& param)
    {
        m_param = param.clone();
    }

    // Track the first frame. should always be called first
//    void
//    OpticalFlowTracker::begin_track(const cv::Mat &img,
//                                    const std::vector<DetectionPtr> &detections,
//                                    int frame_number,
//                                    TrackingDecision *out_decision) {
//        m_id2target.clear();
//        m_prev_img = img.clone();
//        m_motion_predict->set_prev_image(m_prev_img);
//        _update_frame_number(frame_number);
//        for(size_t i = 0; i < detections.size(); i++){
//            TrackTargetPtr track_target_ptr = add_target(detections[i], frame_number);
//            m_id2target[track_target_ptr->get_path_id()] = track_target_ptr;
//            if (out_decision)
//                out_decision->detection_index2path_id[i] = track_target_ptr->get_path_id();
//        }
//    }

    void
    OpticalFlowTracker::begin_track(const cv::Mat &img,
                                    const std::vector<DetectionPtr> &detections,
                                    int frame_number) {
        m_id2target.clear();
        m_motion_predict->set_prev_image(img);
        _update_frame_number(frame_number);
        for(size_t i = 0; i < detections.size(); i++){
            TrackTargetPtr track_target_ptr = create_target(detections[i], frame_number);
            add_target(track_target_ptr);
        }
    }

    void OpticalFlowTracker::finish_track() {
        std::vector<int> delete_ids;
        for (auto &p : m_id2target) {
            delete_ids.push_back(p.first);
        }
        for(auto& del_id : delete_ids){
            TrackingEvent::TargetClosed event_data = TrackingEvent::TargetClosed();
            event_data.m_target = m_id2target[del_id];
            for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                (*iter)->evt_target_closed_before(this, event_data);
            }

            m_id2target.erase(del_id);

            for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                (*iter)->evt_target_closed_after(this, event_data);
            }
        }
        m_frame_number = INIT_TRACKING_FRAME;
        m_path_id_for_generate_unique_id = 0;
    }

    void OpticalFlowTracker::track(const cv::Mat& img,
                                   const std::vector<DetectionPtr>& detections,
                                   int frame_number)
    {
        std::vector<int> delete_id;
        for(auto &p : m_id2target){
            auto time_since_update = frame_number - p.second->get_end_frame_number();
            if (time_since_update > m_param->m_max_time_since_update)
                delete_id.push_back(p.first);
        }
        for(auto &p : delete_id){
            delete_target(p);
        }

        // first motion prediction
        _motion_predict(img, frame_number, m_id2target);

        m_motion_predict->set_prev_image(img);
        _update_frame_number(frame_number);

        if (m_id2target.empty()) {
            for(size_t i = 0; i < detections.size(); i++){
                TrackTargetPtr track_target_ptr = create_target(detections[i], frame_number);
                add_target(track_target_ptr);
            }
            return;
        }


        if (detections.empty()) {
            return;
        }
        // calculate iou
        auto n_det_predict = m_id2target.size();
        auto n_det_now = detections.size();
        std::vector<TrackTargetPtr> targets;
        for(auto& p : m_id2target)
            targets.push_back(p.second);
        std::vector<std::vector<float>> dist_matrix_now2prev(n_det_now, std::vector<float>(n_det_predict, 0));

        for(size_t i=0; i<detections.size(); i++)
        {
            for(size_t j=0; j<targets.size(); j++)
            {
                auto bbox_now = detections[i]->get_bbox();
                auto bbox_after_predict = targets[j]->get_bbox();
                auto iou = compute_iou(bbox_now, bbox_after_predict);
                dist_matrix_now2prev[i][j] = 1-iou;
            }
        }

        // match detection and target after predict
        std::vector<std::pair<int, int>> matched_pair;
        std::vector<int> unmatched_detection_now;
        std::vector<int> unmatched_detection_predict;
        hungarian_match(dist_matrix_now2prev, n_det_now, n_det_predict, m_param->m_max_iou_distance, matched_pair, unmatched_detection_now, unmatched_detection_predict);

        //update tracker state
            // update matched detection_now and detection_predict
            for(size_t i = 0; i < matched_pair.size(); i++){
                auto single_target = targets[matched_pair[i].second];
                auto single_detection = detections[matched_pair[i].first];


                TrackingEvent::TargetAssociation event_data = TrackingEvent::TargetAssociation();
                event_data.m_detection = single_detection;
                event_data.m_target = single_target;
                for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                    (*iter)->evt_target_association_before(this, event_data);
                }

                single_target->set_path_state(TrackPathStateBitmask::Open);
                single_target->set_underlying_detection(single_detection, true);
                single_target->set_end_frame_number(frame_number);

                for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                    (*iter)->evt_target_association_after(this, event_data);
                }
            }

            // create target from detection
            for(size_t i = 0; i < unmatched_detection_now.size(); i++){
                TrackTargetPtr track_target_ptr = create_target(detections[unmatched_detection_now[i]], frame_number);
                add_target(track_target_ptr);
            }

            // delete lost trackers
        return;
    }

    // motion prediction results. will change and delete m_id2target.
    void OpticalFlowTracker::track(const cv::Mat& img,
                                   int frame_number)
    {
        assert_throw(m_frame_number != INIT_TRACKING_FRAME, "m frame number is INIT_TRACKING_FRAME");
        assert_throw(m_frame_number <= frame_number, "frame number less than m frame number");

        auto id2bbox_after_flow = _advance_bbox_with_motion_prediction(img, frame_number, m_id2target);

        vector<int> delete_id;
        // optical flow predict m_id2target,  NOW delete_id is always empty.
        for(auto& p : m_id2target){
            std::map<int, BBOX>::iterator key = id2bbox_after_flow.find(p.first);
            TrackingEvent::TargetMotionPredict event_data = TrackingEvent::TargetMotionPredict();
            event_data.m_target = p.second;
            for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                (*iter)->evt_target_motion_predict_before(this, event_data);
            }

            p.second->set_bbox(key->second);

            for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
                (*iter)->evt_target_motion_predict_after(this, event_data);
            }
        }
        m_motion_predict->set_prev_image(img);
        _update_frame_number(frame_number);
    }


    void OpticalFlowTracker::_motion_predict(const cv::Mat &img, int frame_number,
                                                         const std::map<int, TrackTargetPtr>& id2target){
        auto id2bbox_after_flow = _advance_bbox_with_motion_prediction(img, frame_number, id2target);
        // optical flow predict m_id2target,  if predict bbox out of img then keep old bbox, so NOW delete_id is always empty.
        for(auto& p : id2target){
            std::map<int, BBOX>::iterator key = id2bbox_after_flow.find(p.first);
            p.second->set_bbox(key->second);
        }
    }

    std::map<int, BBOX> OpticalFlowTracker::_advance_bbox_with_motion_prediction(const cv::Mat& img, int frame_number,
                                                                                  const std::map<int, TrackTargetPtr>& id2target){
        assert_throw(m_frame_number < frame_number, "frame number less than m frame number");
        if (id2target.empty())
            return std::map<int, BBOX>();
        // extract id and bbox from m_id2target
        std::vector<int> pre_ids;
        std::vector<BBOX> pre_bbox;
        for(auto p: id2target){
            pre_ids.push_back(p.first);
            pre_bbox.push_back(p.second->get_bbox());
        }

        // generate points based on bboxes
        auto p_param = dynamic_cast<OpticalTrackerParam*>(m_param.get());
        std::vector<POINT> points;
        std::vector<POINT> temp_points;
        for(int i = 0; i < pre_bbox.size(); i++){
            temp_points = generate_uniform_keypoints(pre_bbox[i], p_param->m_pts_per_width, p_param->m_pts_per_height);
            points.insert(points.end(), temp_points.begin(), temp_points.end());
        }

        // using lk flow predict new points
        OpticalFlowMotionPrediction::Result motion_prediction_result;
        m_motion_predict->predict_keypoint_location(img, points, motion_prediction_result);

        // get new bbox based on new points
        std::vector<BBOX> cur_bbox;
        for(int i = 0; i < pre_bbox.size(); i++){
            int number_points = p_param->m_pts_per_height * p_param->m_pts_per_width;
            int point_index = i * number_points;
            cur_bbox.push_back(predict_bbox_by_keypoints(pre_bbox[i],
                                                         &points[0] + point_index,
                                                         &motion_prediction_result.keypoints_predicted[0] + point_index,
                                                         number_points,
                                                         &motion_prediction_result.keypoints_valid[0] + point_index));
        }

        // get output from new bboxes and ids
        std::map<int, BBOX> output;
        for(int i = 0; i < cur_bbox.size(); i++){
            if (is_bbox_inside_image(cur_bbox[i], img.cols, img.rows))
                output[pre_ids[i]] = cur_bbox[i];
            else
                output[pre_ids[i]] = pre_bbox[i];
        }
        return output;
    }


    const TrackerParam *OpticalFlowTracker::get_tracker_param() const {
        return TrackerBase::get_tracker_param();
    }

    void OpticalFlowTracker::set_tracker_param(const TrackerParam &param) {

    }

    const std::map<int, TrackTargetPtr> &OpticalFlowTracker::get_all_open_targets() const {
        return m_id2target;
    }

    void OpticalFlowTracker::delete_target(int path_id) {
        TrackingEvent::TargetClosed event_data = TrackingEvent::TargetClosed();
        event_data.m_target = m_id2target[path_id];
        for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
            (*iter)->evt_target_closed_before(this, event_data);
        }

        m_id2target.erase(path_id);

        for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
            (*iter)->evt_target_closed_after(this, event_data);
        }
    }

    void OpticalFlowTracker::delete_all_targets() {
        m_id2target.clear();
    }

    TrackTargetPtr OpticalFlowTracker::get_open_target(int path_id) const {
        return TrackerBase::get_open_target(path_id);
    }

    void OpticalFlowTracker::add_target(const TrackTargetPtr &target) {
        TrackingEvent::TargetAssociation event_data = TrackingEvent::TargetAssociation();
        event_data.m_detection = target->get_underlying_detection();
        event_data.m_target = target;
        for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
            (*iter)->evt_target_created_before(this, event_data);
        }

        m_id2target[target->get_path_id()] = target;

        for(auto iter = m_event_handlers.begin(); iter != m_event_handlers.end(); iter++){
            (*iter)->evt_target_created_after(this, event_data);
        }

        return;
    }

    TrackTargetPtr OpticalFlowTracker::create_target(const DetectionPtr &det, int frame_number) {
        TrackTargetPtr output = std::make_shared<TrackTarget>();
        output->set_underlying_detection(det, true);
        output->set_start_frame_number(frame_number);
        output->set_end_frame_number(frame_number);
        output->set_path_id(_generate_path_id());
        output->set_path_state(TrackPathStateBitmask::New);
        return output;
    }

    TrackerTrackingStatePtr OpticalFlowTracker::_tracking_state_create() {
        auto output = std::make_shared<OpticalFlowTrackerTrackingSate>();
        return output;
    }

    void OpticalFlowTracker::_tracking_state_fill(TrackerTrackingState& state) {
        auto optical_state = dyncast_with_check<OpticalFlowTrackerTrackingSate>(&state);
        TrackerBase::_tracking_state_fill(state);
        optical_state->m_prev_img = m_motion_predict->get_prev_image();
    }

    void OpticalFlowTracker::_tracking_state_recover(const TrackerTrackingState& state) {
        auto optical_state = dyncast_with_check<OpticalFlowTrackerTrackingSate>(&state);
        TrackerBase::_tracking_state_recover(state);
        m_motion_predict->set_prev_image(optical_state->m_prev_img);
    }

    void OpticalFlowTracker::add_event_handler(const TrackingEventHandlerPtr& handler) {
        m_event_handlers.insert(handler);
    }

    void OpticalFlowTracker::remove_event_handler(const TrackingEventHandlerPtr& handler) {
        m_event_handlers.erase(handler);
    }

}


