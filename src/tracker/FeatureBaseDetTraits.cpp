//
// Created by sfj on 2022/5/23.
//

#include "RedoxiTrack/tracker/DetectionTraits.h"


namespace RedoxiTrack{

    double FeatureBasedDetTraits::compute_detection_distance(const Detection *a, const Detection *b) {
        auto feature_a = a->get_feature();
        auto feature_b = b->get_feature();
        auto feature_traits = get_feature_traits();
        double dist;
        double max_dist = feature_traits->max_distance();
        if(feature_a.size() == 0 || feature_b.size() == 0){
            dist = max_dist;
        } else {
            dist = feature_traits->distance(feature_a, feature_b);
        }
        return dist;
    }
}