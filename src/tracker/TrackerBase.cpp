#include "RedoxiTrack/tracker/TrackerBase.h"
#include "RedoxiTrack/external/Hungarian.h"

namespace RedoxiTrack
{

TrackTargetPtr TrackerBase::get_open_target(int path_id) const
{
    auto id2target = get_all_open_targets();
    auto p = id2target.find(path_id);
    if (p == id2target.end())
        return NULL;
    else
        return p->second;
}


void TrackerBase::push_tracking_state()
{
    auto x = _tracking_state_create();
    _tracking_state_fill(*x);
    m_state_stack.push_back(x);
}

void TrackerBase::pop_tracking_state(bool apply)
{
    assert_throw(m_state_stack.size() > 0, "failed pop tracking state, m_state_stack is empty");
    auto x = m_state_stack.back();
    m_state_stack.pop_back();
    if (apply)
        _tracking_state_recover(*x);
}

TrackerTrackingStatePtr TrackerBase::_tracking_state_create()
{
    auto output = std::make_shared<TrackerTrackingState>();
    return output;
}

void TrackerBase::_tracking_state_fill(TrackerTrackingState &state)
{
    state.m_id2target = m_id2target;
    state.m_id2target_clone.clear();
    for (auto &p : m_id2target) {
        state.m_id2target_clone[p.first] = dynamic_pointer_cast<TrackTarget>(p.second->clone());
    }
    state.m_frame_number = m_frame_number;
}

void TrackerBase::_tracking_state_recover(const TrackerTrackingState &state)
{
    m_id2target = state.m_id2target;
    for (auto &p : state.m_id2target_clone) {
        p.second->copy_to(*m_id2target[p.first]);
    }
    m_frame_number = state.m_frame_number;
}

void TrackerBase::reset_tracking_state()
{
    init(*m_param);
}

TrackerBase::TrackerBase()
{
    // ctor
}

TrackerBase::~TrackerBase()
{
    // dtor
}

} // namespace RedoxiTrack
