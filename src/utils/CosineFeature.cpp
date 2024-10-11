//
// Created by sfj on 2022/7/29.
//

#include "RedoxiTrack/utils/CosineFeature.h"

namespace RedoxiTrack{
    double CosineFeature::distance(const fVECTOR &input1, const fVECTOR &input2) const {
        auto input1_normal = input1.normalized();
        auto input2_normal = input2.normalized();
        auto dist = (1-input1_normal.dot(input2_normal))/2.0;
        return dist;
    }

    void
    CosineFeature::linear_combine(fVECTOR *output, const fVECTOR &fa, const fVECTOR &fb, double wa, double wb) const {
        auto fa_normal = fa.normalized();
        auto fb_normal = fb.normalized();
        *output = (wa*fa_normal+wb*fb_normal).normalized();
    }

    double CosineFeature::max_distance() const {
        return 1;
    }

    double CosineFeature::min_distance() const {
        return 0;
    }
}