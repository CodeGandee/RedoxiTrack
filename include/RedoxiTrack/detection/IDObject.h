#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"

namespace RedoxiTrack
{

// object with globally unique id
class REDOXI_TRACK_API IDObject
{
  private:
    int m_id = 0;
    static int generate_id()
    {
        static int _id = 0;
        return _id++;
    }

  public:
    IDObject()
    {
        m_id = generate_id();
    }
    virtual ~IDObject()
    {
    }

    int get_id() const
    {
        return m_id;
    }
};
} // namespace RedoxiTrack
