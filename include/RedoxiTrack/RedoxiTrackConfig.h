#pragma once

// Define REDOXI_TRACK_API for any platform
#if defined _WIN32 || defined __CYGWIN__
#    ifdef REDOXI_TRACK_EXPORT
// Exporting...
#        ifdef __GNUC__
#            define REDOXI_TRACK_API __attribute__((dllexport))
#        else
#            define REDOXI_TRACK_API __declspec(dllexport)
#        endif
#    else
#        ifdef REDOXI_TRACK_STATIC_LIBS
#            define REDOXI_TRACK_API
#        else
#            ifdef __GNUC__
#                define REDOXI_TRACK_API __attribute__((dllimport))
#            else
#                define REDOXI_TRACK_API __declspec(dllimport)
#            endif
#        endif
#    endif
#    define REDOXI_TRACK_PRIVATE_API
#else
#    if __GNUC__ >= 4
#        define REDOXI_TRACK_API __attribute__((visibility("default")))
#        define REDOXI_TRACK_PRIVATE_API __attribute__((visibility("hidden")))
#    else
#        define REDOXI_TRACK_API
#        define REDOXI_TRACK_PRIVATE_API
#    endif
#endif

// for igcpp floating point type selection

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

namespace RedoxiTrack
{

using BBOX = cv::Rect2f;
using POINT = cv::Point2f;
using fVECTOR = Eigen::Matrix<float, Eigen::Dynamic, 1>;
using fMATRIX =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// value representing invalid index in linear assignment match
const int INVALID_INDEX = -1;

// value representing init tracking frame
const int INIT_TRACKING_FRAME = -1;

using DetectionType = int;
namespace DetectionTypes
{
enum {
    None,

    PersonHead,
    PersonFace,
    PersonBody
};
};

using EventHandlerResultType = int;
// if has multi evenHandlerResult, then using the max value type
namespace EventHandlerResultTypes
{
enum {
    None = 0,
    Association_RejectAndCreateNew,
    Association_RejectAndDiscard
};
}

// state of a path being tracked
namespace TrackPathStateBitmask
{
const int None = 0;       // unspecified state
const int New = 1;        // the first frame being create, 目标刚被创建时

// open/lost/close状态互斥
// 举个例子，类似于射击游戏，
// 瞄准一个目标时，视野中存在这个目标且我们跟踪瞄准这个目标，此时目标处于Open状态
// 如果目标躲在障碍物后面，我们没法瞄准，但我们认为目标只是短期看不到，还是会露头，于是我们依旧尝试再瞄准他，则目标处于Lost状态，
// 如果目标进了一个门，我们没法瞄准（进入lost状态），等待一段时间后我们认为目标已经离开，不再打算瞄准他，则目标处于Close状态
const int Open = 1 << 1;  // 目标短期没丢失，仍可能被检测框匹配，运动预测依然有效
const int Lost = 1 << 2;  // 目标短期丢失，运动预测不可靠（也可能不做运动预测），仍可能被检测框/reid匹配，运动预测依然有效
const int Close = 1 << 3; // 目标跟踪结束，不会再被更新

const int NoMatch = 1 << 4; // 目标在当前帧中未被检测框匹配
} // namespace TrackPathStateBitmask

inline void assert_throw(bool expr, std::string msg, bool warn_only = false)
{
    if (!expr) {
        std::cout << "== ASSERT FAILED ==" << std::endl;
        std::cout << msg << std::endl;
        std::cout.flush();
        if (!warn_only)
            throw std::logic_error(msg.c_str());
    }
}

template <typename T_TO, typename T_FROM>
T_TO *dyncast_with_check(T_FROM *p, const char *msg_if_failed = nullptr)
{
    auto x = dynamic_cast<T_TO *>(p);
    if (!x) {
        std::ostringstream os;
        os << "failed to cast " << typeid(p).name() << " to "
           << typeid(T_TO).name();
        std::cout << os.str() << std::endl;
        throw std::logic_error(os.str());
    }
    return x;
}

template <typename T_TO, typename T_FROM>
const T_TO *dyncast_with_check(const T_FROM *p,
                               const char *msg_if_failed = nullptr)
{
    return dyncast_with_check<T_TO, T_FROM>(const_cast<T_FROM *>(p),
                                            msg_if_failed);
}

} // namespace RedoxiTrack
