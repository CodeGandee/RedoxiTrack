#include "RedoxiTrack/utils/softnms.h"
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
typedef bg::model::d2::point_xy<double, boost::geometry::cs::cartesian> DPoint;
typedef bg::model::box<DPoint> DBox;
typedef std::pair<DBox, unsigned> Value;

namespace RedoxiTrack {
    void SoftNMS::set_method(enum softnms_method method) {
        m_method = method;
    }

    void SoftNMS::set_iou_thres(float iou_thres) {
        m_iou_thres = iou_thres;
    }

    void SoftNMS::set_iou_scale(float iou_scale) {
        m_iou_scale = iou_scale;
    }

    void SoftNMS::set_sigma(float sigma) {
        m_sigma = sigma;
    }

    void SoftNMS::set_use_IDfeature(bool use_IDfeature) {
        m_use_IDfeature = use_IDfeature;
    }

    void SoftNMS::set_compute_IDdistance(const RedoxiTrack::DetectionComparisionPtr& compute_IDdistance) {
        m_compute_IDdistance = compute_IDdistance;
    }

    std::vector<float> SoftNMS::apply(std::vector<RedoxiTrack::DetectionPtr>& detections) {
        // detections are empty
        if (detections.empty()) {
            return std::vector<float>();
        }

        m_detections = detections;
        int N = m_detections.size();
        float max_score, max_pos, pos, weight;
        RedoxiTrack::DetectionPtr tmp_box;

        // copy scores and create m_map_influenced_box
        m_scores.resize(N);
        for (int i = 0; i < N; i++) {
            m_scores[i] = m_detections[i]->get_confidence();
            m_map_influenced_box[m_detections[i]] = std::vector<RedoxiTrack::DetectionPtr>();
        }

        for (int i = 0; i < N; i++) {
            max_score = m_detections[i]->get_confidence();
            max_pos = i;
            pos = i + 1;

            // get max score box
            while (pos < N) {
                if (max_score < m_detections[pos]->get_confidence()) {
                    max_score = m_detections[pos]->get_confidence();
                    max_pos = pos;
                }
                pos += 1;
            }

            // swap box and score
            tmp_box = m_detections[max_pos];
            std::swap(m_detections[i], m_detections[max_pos]);
            std::swap(m_scores[i], m_scores[max_pos]);

            // create rtree
            bgi::rtree<Value, bgi::quadratic<100>> rtree;
            for (unsigned t = i + 1; t < N; t++) {
                DBox b(DPoint(m_detections[t]->get_bbox().tl().x, m_detections[t]->get_bbox().tl().y),
                        DPoint(m_detections[t]->get_bbox().br().x, m_detections[t]->get_bbox().br().y));
                rtree.insert(std::make_pair(b, t));
            }

            // query box intersects with m_detection[i]
            DBox query_box(DPoint(tmp_box->get_bbox().tl().x, tmp_box->get_bbox().tl().y),
                        DPoint(tmp_box->get_bbox().br().x, tmp_box->get_bbox().br().y));
            std::vector<Value> intersects_boxes;
            rtree.query(bgi::intersects(query_box), std::back_inserter(intersects_boxes));

            // calculate iou and weight
            for (auto it = intersects_boxes.rbegin(); it != intersects_boxes.rend(); it++) {
                // iou
                auto maxX = std::max(tmp_box->get_bbox().tl().x, m_detections[it->second]->get_bbox().tl().x);
                auto maxY = std::max(tmp_box->get_bbox().tl().y, m_detections[it->second]->get_bbox().tl().y);
                auto minX = std::min(tmp_box->get_bbox().br().x, m_detections[it->second]->get_bbox().br().x);
                auto minY = std::min(tmp_box->get_bbox().br().y, m_detections[it->second]->get_bbox().br().y);

                float width = ((minX - maxX + 1) > 0)? (minX - maxX + 1) : 0;
                float height = ((minY - maxY + 1) > 0)? (minY - maxY + 1) : 0;

                auto IOU = (width * height) / (tmp_box->get_bbox().area() + m_detections[it->second]->get_bbox().area() - width * height);

                float similarity = 0;
                // if without id feature, back to navie softnms
                if (!m_use_IDfeature || m_detections[it->second]->get_feature().size() == 0) {
                    similarity = IOU;
                }
                else {
                    auto id_similarity = 1 - m_compute_IDdistance->compute_detection_distance(tmp_box.get(), m_detections[it->second].get());
                    similarity = m_iou_scale * IOU + (1 - m_iou_scale) * id_similarity;
                }

                // weight
                weight = 1.0;
                if (m_method == NMS) {
                    if (IOU >= m_iou_thres) weight = 0;
                }
                else if (m_method == SOFTNMS) {
                    if (IOU >= m_iou_thres) weight = 1 - similarity;
                }
                else if (m_method == SIGMA_SOFTNMS) {
                    weight = std::exp(-(similarity * similarity) / m_sigma);
                }
                // update
                m_scores[it->second] = m_scores[it->second] * weight;
                m_map_influenced_box[tmp_box].push_back(m_detections[it->second]);
            }
        }

        return m_scores;
    }

    std::map<RedoxiTrack::DetectionPtr, std::vector<RedoxiTrack::DetectionPtr>>
    SoftNMS::get_influenced_boxes() {
        return m_map_influenced_box;
    }

    std::vector<RedoxiTrack::DetectionPtr> SoftNMS::filter(const float thres, std::vector<int>& del_inds) {
        for(int i = 0; i < m_detections.size(); i++) {
            if (m_scores[i] < thres) {
                del_inds.push_back(i + del_inds.size());
                m_detections.erase(m_detections.begin() + i);
                m_scores.erase(m_scores.begin() + i);
                i--;
            }
        }
        return m_detections;
    }
}