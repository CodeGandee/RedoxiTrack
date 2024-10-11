//
// Created by 18200 on 2022/2/21.
//

#pragma once

#include "OpticalFlowMotionPrediction.h"
#include "OpticalTrackerParam.h"
#include "RedoxiTrack/RedoxiTrackConfig.h"


#include "hi_comm_isp.h"
#include "hi_comm_ive.h"
#include "hi_comm_sys.h"
#include "hi_comm_vpss.h"
#include "hi_common.h"
#include "hi_errno.h"
#include "hi_nnie.h"
#include "mpi_ive.h"
#include "mpi_nnie.h"

// #include "sample_comm.h"
// #include "sample_comm_ive.h"
// #include "sample_comm_svp.h"
// #include "sample_comm_nnie.h"

// extern "C" {
//     #include <if_algo/if_algo_api.h>
// }

#define IVE_MASK 0b00000000000000000000000001111111
namespace RedoxiTrack
{

// typedef struct hiSAMPLE_IVE_ST_LK_S
// {
//     IVE_SRC_IMAGE_S                       astPrevPyr[4];//图片存储
//     IVE_SRC_IMAGE_S                       astNextPyr[4];//图片存储
//     IVE_SRC_MEM_INFO_S                    stPrevPts;//光流点坐标
//     IVE_MEM_INFO_S                        stNextPts;//光流点坐标
//     IVE_DST_MEM_INFO_S                    stStatus;//光流点计算结果状态标志
//     IVE_DST_MEM_INFO_S                    stErr;//光流点计算结果误差
//     IVE_LK_OPTICAL_FLOW_PYR_CTRL_S        stLkPyrCtrl;//光流技术控制器
//     IVE_IMAGE_S                           stTmp;//
// } IVE_LK_TRACK;

struct hiSAMPLE_IVE_ST_LK_S {
    IVE_SRC_IMAGE_S astPrevPyr[4];              // 图片存储
    IVE_SRC_IMAGE_S astNextPyr[4];              // 图片存储
    IVE_SRC_MEM_INFO_S stPrevPts;               // 光流点坐标
    IVE_MEM_INFO_S stNextPts;                   // 光流点坐标
    IVE_DST_MEM_INFO_S stStatus;                // 光流点计算结果状态标志
    IVE_DST_MEM_INFO_S stErr;                   // 光流点计算结果误差
    IVE_LK_OPTICAL_FLOW_PYR_CTRL_S stLkPyrCtrl; // 光流技术控制器
    IVE_IMAGE_S stTmp;                          //

    hiSAMPLE_IVE_ST_LK_S() = default;
    hiSAMPLE_IVE_ST_LK_S(const hiSAMPLE_IVE_ST_LK_S &) = delete;
    hiSAMPLE_IVE_ST_LK_S &operator=(const hiSAMPLE_IVE_ST_LK_S &x) const = delete;
};
using IVE_LK_TRACK = hiSAMPLE_IVE_ST_LK_S;

using IVE_LK_TRACK_PTR = std::shared_ptr<IVE_LK_TRACK>;

REDOXI_TRACK_API void ive_lk_track_deleter(IVE_LK_TRACK *pstStLk);

class REDOXI_TRACK_API NNIEOpticalFlow : public OpticalFlowMotionPrediction
{
  public:
    NNIEOpticalFlow();

    void set_prev_image(const cv::Mat &img) override;

    void set_current_image(const cv::Mat &img) override;

    void set_prev_image_by_current() override;

    void predict_keypoint_location(const std::vector<POINT> &points,
                                   MotionPredictionByImageKeypoint::Result &output) override;

  private:
    void _trans_mat2hi(const cv::Mat &img, IVE_LK_TRACK_PTR pstStLk);
    HI_S32 _trans_hi2pyr(IVE_LK_TRACK_PTR pstStLk, IVE_SRC_IMAGE_S *pyr);
    HI_S32 _trans_hi2pyr_helper(IVE_LK_TRACK_PTR pstStLk, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst);
    void _trans_points2hi(const std::vector<POINT> &points, IVE_LK_TRACK_PTR pstStLk);
    void _trans_hi2points(IVE_LK_TRACK_PTR pstStLk, std::vector<POINT> &points, std::vector<uint8_t> &valid);
    HI_S32 _copy_single_pyr(IVE_HANDLE *pIveHandle, IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst, IVE_DMA_CTRL_S *pstDmaCtrl, HI_BOOL bInstant);

  private:
    IVE_LK_TRACK_PTR m_ive_lk_tracker;

    static constexpr int _get_hi_max_level()
    {
        return 2;
    }
    static constexpr int _get_hi_max_pts_num()
    {
        return 500;
    }
    static constexpr int _get_hi_min_eig_thr()
    {
        return 2;
    }
    static constexpr int _get_hi_iter_cnt()
    {
        return 20;
    }
    static constexpr int _get_hi_q8_eps()
    {
        return 230;
    }
    static constexpr int _get_hi_width()
    {
        return 0;
    }
};
} // namespace RedoxiTrack
