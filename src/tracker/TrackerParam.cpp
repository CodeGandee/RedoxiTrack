//
// Created by 18200 on 2022/2/9.
//

#include "RedoxiTrack/tracker/TrackerParam.h"

namespace RedoxiTrack
{
void TrackerParam::copy_to(TrackerParam &p) const
{
    p = *this;
}

std::shared_ptr<TrackerParam> TrackerParam::clone() const
{
    std::shared_ptr<TrackerParam> m = std::make_shared<TrackerParam>();
    copy_to(*m);
    return m;
}

void TrackerParam::set_preferred_image_size(const cv::Size &size)
{
    m_preferred_image_size = size;
}
} // namespace RedoxiTrack
