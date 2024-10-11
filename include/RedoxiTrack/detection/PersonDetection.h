#pragma once
#include "RedoxiTrack/detection/Detection.h"
#include "RedoxiTrack/detection/SingleDetection.h"

namespace RedoxiTrack
{

class REDOXI_TRACK_API PersonDetection : public Detection
{
  protected:
    std::map<int, DetectionPtr> m_detections;

  public:
    PersonDetection() = default;
    virtual ~PersonDetection() = default;

    /**
     * create person detection with body parts, if not exist, input NULL
     * @param head
     * @param face
     * @param body
     */
    void init(
        const SingleDetection *head,
        const SingleDetection *face,
        const SingleDetection *body);

    DetectionPtr get_head() const;
    DetectionPtr get_face() const;
    DetectionPtr get_body() const;
};

} // namespace RedoxiTrack
