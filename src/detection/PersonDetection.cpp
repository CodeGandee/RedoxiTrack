#include "RedoxiTrack/detection/PersonDetection.h"

namespace RedoxiTrack{

DetectionPtr PersonDetection::get_head() const
{
    auto it = m_detections.find(DetectionTypes::PersonHead);
    if(it != m_detections.end())
        return it->second;
    return NULL;
}

DetectionPtr PersonDetection::get_face() const
{
    auto it = m_detections.find(DetectionTypes::PersonFace);
    if(it != m_detections.end())
        return it->second;
    return NULL;
}

DetectionPtr PersonDetection::get_body() const
{
    auto it = m_detections.find(DetectionTypes::PersonBody);
    if(it != m_detections.end())
        return it->second;
    return NULL;
}

void PersonDetection::init(
       const SingleDetection* head,
       const SingleDetection* face,
       const SingleDetection* body)
{
    m_detections.clear();

    if(head){
        auto p = std::make_shared<SingleDetection>();
        *p = *head;
        p->set_type(DetectionTypes::PersonHead);
        m_detections[DetectionTypes::PersonHead] = p;
    }

    if(face){
        auto p = std::make_shared<SingleDetection>();
        *p = *face;
        p->set_type(DetectionTypes::PersonFace);
        m_detections[DetectionTypes::PersonFace] = p;
    }

    if(body){
        auto p = std::make_shared<SingleDetection>();
        *p = *body;
        p->set_type(DetectionTypes::PersonBody);
        m_detections[DetectionTypes::PersonBody] = p;
    }
}

}


