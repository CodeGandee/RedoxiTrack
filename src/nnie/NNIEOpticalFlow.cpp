//
// Created by 18200 on 2022/2/21.
//
#include "RedoxiTrack/nnie/NNIEOpticalFlow.h"
#include "unistd.h"
#include "RedoxiTrack/utils/utility_functions.h"
#include "sample_comm.h"
#include "sample_comm_ive.h"
#include "sample_comm_svp.h"
#include "sample_comm_nnie.h"

#undef SAMPLE_CHECK_EXPR_RET

namespace RedoxiTrack{
    HI_S32 SAMPLE_COMM_IVE_FreeImage(IVE_IMAGE_S *pstImg){
        switch (pstImg->enType){
            case IVE_IMAGE_TYPE_U8C1:
                if (pstImg->au64PhyAddr[0] != 0 && pstImg->au64VirAddr[0] != 0){
                    HI_MPI_SYS_MmzFree(pstImg->au64PhyAddr[0], (void *)pstImg->au64VirAddr[0]);
                    pstImg->au64PhyAddr[0] = 0;
                    pstImg->au64VirAddr[0] = 0;
                }
                break;
            case IVE_IMAGE_TYPE_U8C3_PACKAGE:
                if (pstImg->au64PhyAddr[0] != 0 && pstImg->au64VirAddr[0] != 0){
                    HI_MPI_SYS_MmzFree(pstImg->au64PhyAddr[0], (void *)pstImg->au64VirAddr[0]);
                    pstImg->au64PhyAddr[0] = 0;
                    pstImg->au64VirAddr[0] = 0;
                }
                break;
            case IVE_IMAGE_TYPE_U8C3_PLANAR:
                for(int i = 0; i < 3; ++i){
                    if(pstImg->au64PhyAddr[i] != 0 && pstImg->au64VirAddr[i] != 0){
                        HI_MPI_SYS_MmzFree(pstImg->au64PhyAddr[i], (void *)pstImg->au64VirAddr[i]);
                        pstImg->au64PhyAddr[i] = 0;
                        pstImg->au64VirAddr[i] = 0;
                    }
                }
                break;
            case IVE_IMAGE_TYPE_YUV420SP:
                if (pstImg->au64PhyAddr[0] != 0 && pstImg->au64VirAddr[0] != 0){
                    HI_MPI_SYS_MmzFree(pstImg->au64PhyAddr[0], (void *)pstImg->au64VirAddr[0]);
                    pstImg->au64PhyAddr[0] = 0;
                    pstImg->au64VirAddr[0] = 0;
                }
                break;
        }
        return HI_SUCCESS;
    }

    HI_S32 SAMPLE_COMM_IVE_FreeMemInfo(IVE_MEM_INFO_S *pstMemInfo){
        if (pstMemInfo->u64VirAddr != 0 && pstMemInfo->u64VirAddr != 0){
            HI_MPI_SYS_MmzFree(pstMemInfo->u64PhyAddr, (void *)pstMemInfo->u64VirAddr);
            pstMemInfo->u64PhyAddr = 0;
            pstMemInfo->u64VirAddr = 0;
        }
        return HI_SUCCESS;
    }

    void ive_lk_track_deleter(IVE_LK_TRACK* pstStLk){
        HI_S32 s32Ret = HI_SUCCESS;
        // free prev pts
        s32Ret = SAMPLE_COMM_IVE_FreeMemInfo(&(pstStLk->stPrevPts));
        // free next pts
        s32Ret = SAMPLE_COMM_IVE_FreeMemInfo(&(pstStLk->stNextPts));
        // free status
        s32Ret = SAMPLE_COMM_IVE_FreeMemInfo(&(pstStLk->stStatus));
        // free err
        s32Ret = SAMPLE_COMM_IVE_FreeMemInfo(&(pstStLk->stErr));
        // free stTmp
        s32Ret = SAMPLE_COMM_IVE_FreeImage(&(pstStLk->stTmp));
        // free prev img next img
        if(pstStLk->stTmp.u32Width != 0){
            for(HI_U16 i = 0; i <= pstStLk->stLkPyrCtrl.u8MaxLevel; i++) {
                s32Ret = SAMPLE_COMM_IVE_FreeImage(&(pstStLk->astPrevPyr[i]));
                s32Ret = SAMPLE_COMM_IVE_FreeImage(&(pstStLk->astPrevPyr[i]));
            }
        }
    }

    NNIEOpticalFlow::NNIEOpticalFlow(){
        IVE_LK_TRACK_PTR n_ive_tracker(new IVE_LK_TRACK(), ive_lk_track_deleter);
        m_ive_lk_tracker = n_ive_tracker;
        // m_ive_lk_tracker = std::make_shared<IVE_LK_TRACK>();
        m_ive_lk_tracker->stLkPyrCtrl.enOutMode = IVE_LK_OPTICAL_FLOW_PYR_OUT_MODE_BOTH;//output mode of OF
        m_ive_lk_tracker->stLkPyrCtrl.bUseInitFlow = HI_FALSE;
        m_ive_lk_tracker->stLkPyrCtrl.u16PtsNum = _get_hi_max_pts_num();//芯片api参数，设置为最大值500即可
        m_ive_lk_tracker->stLkPyrCtrl.u8MaxLevel = _get_hi_max_level();
        m_ive_lk_tracker->stLkPyrCtrl.u0q8MinEigThr = _get_hi_min_eig_thr();//最小特征阈值，设置小值，过大导致计算光流失败
        m_ive_lk_tracker->stLkPyrCtrl.u8IterCnt = _get_hi_iter_cnt(); // 迭代次数 20
        m_ive_lk_tracker->stLkPyrCtrl.u0q8Eps = _get_hi_q8_eps();//迭代收敛阈值，推荐值2，设置大阈值利于计算得到光流值 230
        m_ive_lk_tracker->stTmp.u32Width = _get_hi_width(); //用来判断是否分配空间

        HI_S32 s32Ret = HI_SUCCESS;
        HI_U32 u32Size = 0;
        //Init prev pts
        u32Size = sizeof(IVE_POINT_S25Q7_S) * m_ive_lk_tracker->stLkPyrCtrl.u16PtsNum;
        u32Size = SAMPLE_COMM_IVE_CalcStride(u32Size, IVE_ALIGN);
        s32Ret = SAMPLE_COMM_IVE_CreateMemInfo(&(m_ive_lk_tracker->stPrevPts), u32Size);
        assert_throw(HI_SUCCESS == s32Ret, "Error, prev pts SAMPLE_COMM_IVE_CreateMemInfo failed!");

        //Init next pts
        s32Ret = SAMPLE_COMM_IVE_CreateMemInfo(&(m_ive_lk_tracker->stNextPts), u32Size);
        assert_throw(HI_SUCCESS == s32Ret, "Error, next pts SAMPLE_COMM_IVE_CreateMemInfo failed!");

        //Init status
        u32Size = sizeof(HI_U8) *m_ive_lk_tracker->stLkPyrCtrl.u16PtsNum;
        u32Size = SAMPLE_COMM_IVE_CalcStride(u32Size, IVE_ALIGN);
        s32Ret = SAMPLE_COMM_IVE_CreateMemInfo(&(m_ive_lk_tracker->stStatus), u32Size);
        assert_throw(HI_SUCCESS == s32Ret, "Error, stStatus SAMPLE_COMM_IVE_CreateMemInfo failed!");

        //Init err
        u32Size = sizeof(HI_U9Q7) * m_ive_lk_tracker->stLkPyrCtrl.u16PtsNum;
        u32Size = SAMPLE_COMM_IVE_CalcStride(u32Size, IVE_ALIGN);
        s32Ret = SAMPLE_COMM_IVE_CreateMemInfo(&(m_ive_lk_tracker->stErr), u32Size);
        assert_throw(HI_SUCCESS == s32Ret, "Error, stErr SAMPLE_COMM_IVE_CreateMemInfo failed!");
    }

    void NNIEOpticalFlow::set_prev_image(const cv::Mat &img) {
        assert_throw(img.rows % 8 == 0, "Error, ive_input_img_height error!");
        assert_throw(img.cols % 8 == 0, "Error, ive_input_img_width error!");
        assert_throw(img.rows <= 720, "Error, ive_input_img_height error!");
        assert_throw(img.cols <= 1280, "Error, ive_input_img_width error!");

        //Init Pyr
        if(m_ive_lk_tracker->stTmp.u32Width == 0){
            HI_S32 s32Ret = HI_SUCCESS;
            HI_U32 img_width = img.cols;
            HI_U32 img_height = img.rows;
            for(HI_U16 i = 0; i <= m_ive_lk_tracker->stLkPyrCtrl.u8MaxLevel; i++) {
                s32Ret = SAMPLE_COMM_IVE_CreateImage(&m_ive_lk_tracker->astPrevPyr[i], IVE_IMAGE_TYPE_U8C1, img_width >> i, img_height >> i);
                assert_throw(HI_SUCCESS == s32Ret, "Error, astPrevPyr SAMPLE_COMM_IVE_CreateImage failed!");

                s32Ret = SAMPLE_COMM_IVE_CreateImage(&m_ive_lk_tracker->astNextPyr[i], IVE_IMAGE_TYPE_U8C1,
                    m_ive_lk_tracker->astPrevPyr[i].u32Width, m_ive_lk_tracker->astPrevPyr[i].u32Height);
                assert_throw(HI_SUCCESS == s32Ret, "Error, astNextPyr SAMPLE_COMM_IVE_CreateImage failed!");
            }

            s32Ret = SAMPLE_COMM_IVE_CreateImage(&m_ive_lk_tracker->stTmp, IVE_IMAGE_TYPE_U8C1, img_width, img_height);
            assert_throw(HI_SUCCESS == s32Ret, "Error, stTmp SAMPLE_COMM_IVE_CreateImage failed!");
        }

        _trans_mat2hi(img, m_ive_lk_tracker);
        _trans_hi2pyr(m_ive_lk_tracker, m_ive_lk_tracker->astPrevPyr);
    }

    void NNIEOpticalFlow::set_current_image(const cv::Mat &img) {
        assert_throw(img.rows % 8 == 0, "Error, ive_input_img_height error!");
        assert_throw(img.cols % 8 == 0, "Error, ive_input_img_width error!");
        assert_throw(img.rows <= 720, "Error, ive_input_img_height error!");
        assert_throw(img.cols <= 1280, "Error, ive_input_img_width error!");
        assert_throw(m_ive_lk_tracker->stTmp.u32Width != 0, "Error, use set_prev_image before set_current_imager!");

        _trans_mat2hi(img, m_ive_lk_tracker);
        _trans_hi2pyr(m_ive_lk_tracker, m_ive_lk_tracker->astNextPyr);
    }

    void NNIEOpticalFlow::set_prev_image_by_current() {
        HI_U8 i;
        HI_S32 s32Ret = HI_FAILURE;
        IVE_HANDLE hIveHandle;

        IVE_DMA_CTRL_S stCtrl;
        memset(&stCtrl,0,sizeof(stCtrl));
        stCtrl.enMode = IVE_DMA_MODE_DIRECT_COPY;

        for (i = 0; i <= m_ive_lk_tracker->stLkPyrCtrl.u8MaxLevel; i++) {
            s32Ret = _copy_single_pyr(&hIveHandle,&m_ive_lk_tracker->astNextPyr[i],&m_ive_lk_tracker->astPrevPyr[i],&stCtrl,HI_TRUE);
            assert_throw(HI_SUCCESS == s32Ret, "Error, _copy_single_pyr failed!");
        }
    }

    void NNIEOpticalFlow::predict_keypoint_location(const std::vector<POINT> &points,
                                                    MotionPredictionByImageKeypoint::Result &output) {
        IVE_HANDLE hIveHandle;
        _trans_points2hi(points, m_ive_lk_tracker);
        HI_S32 s32Ret = HI_MPI_IVE_LKOpticalFlowPyr(&hIveHandle,
                m_ive_lk_tracker->astPrevPyr, m_ive_lk_tracker->astNextPyr,
                &m_ive_lk_tracker->stPrevPts, &m_ive_lk_tracker->stNextPts,
                &m_ive_lk_tracker->stStatus, &m_ive_lk_tracker->stErr,
                &m_ive_lk_tracker->stLkPyrCtrl, HI_TRUE);

        HI_BOOL bFinish = HI_FALSE;
        HI_BOOL bBlock = HI_TRUE;
        assert_throw(HI_SUCCESS == s32Ret, "Error, HI_MPI_IVE_LKOpticalFlowPyr failed!");
        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        while(HI_ERR_IVE_QUERY_TIMEOUT == s32Ret){
            usleep(100);
            s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        }
        assert_throw(HI_SUCCESS == s32Ret, "Error, HI_MPI_IVE_LKOpticalFlowPyr HI_MPI_IVE_Query failed!");

        auto p = dynamic_cast<OpticalFlowMotionPrediction::Result *>(&output);
        _trans_hi2points(m_ive_lk_tracker, p->keypoints_predicted, p->keypoints_valid);
    }

    HI_S32 NNIEOpticalFlow::_copy_single_pyr(IVE_HANDLE *pIveHandle, IVE_SRC_IMAGE_S *pstSrc,
                                    IVE_DST_IMAGE_S *pstDst, IVE_DMA_CTRL_S *pstDmaCtrl, HI_BOOL bInstant) {
        HI_S32 s32Ret = HI_FAILURE;
        IVE_SRC_DATA_S stDataSrc;
        IVE_DST_DATA_S stDataDst;

        stDataSrc.u64VirAddr    = pstSrc->au64VirAddr[0];
        stDataSrc.u64PhyAddr    = pstSrc->au64PhyAddr[0];
        stDataSrc.u32Width      = pstSrc->u32Width;
        stDataSrc.u32Height     = pstSrc->u32Height;
        stDataSrc.u32Stride     = pstSrc->au32Stride[0];

        stDataDst.u64VirAddr    = pstDst->au64VirAddr[0];
        stDataDst.u64PhyAddr    = pstDst->au64PhyAddr[0];
        stDataDst.u32Width      = pstDst->u32Width;
        stDataDst.u32Height     = pstDst->u32Height;
        stDataDst.u32Stride     = pstDst->au32Stride[0];
        s32Ret = HI_MPI_IVE_DMA(pIveHandle, &stDataSrc, &stDataDst,pstDmaCtrl,bInstant);

        HI_BOOL bFinish = HI_FALSE;
        HI_BOOL bBlock = HI_TRUE;
        assert_throw(HI_SUCCESS == s32Ret, "Error, HI_MPI_IVE_DMA failed!");
        s32Ret = HI_MPI_IVE_Query(*pIveHandle, &bFinish, bBlock);
        while(HI_ERR_IVE_QUERY_TIMEOUT == s32Ret){
            usleep(100);
            s32Ret = HI_MPI_IVE_Query(*pIveHandle, &bFinish, bBlock);
        }
        assert_throw(HI_SUCCESS == s32Ret, "Error, HI_MPI_IVE_DMA HI_MPI_IVE_Query failed!");

        return s32Ret;
    }

    void NNIEOpticalFlow::_trans_mat2hi(const cv::Mat &img, IVE_LK_TRACK_PTR pstStLk){
        HI_U8* dst_pointer = (HI_U8 *)(HI_UL)pstStLk->stTmp.au64VirAddr[0];
        HI_U8* src_pointer = (HI_U8*)img.data;

        HI_U16 height = img.rows;
        pstStLk->stTmp.u32Height = height;
        HI_U16 width = img.cols;
        pstStLk->stTmp.u32Width = width;

        for (HI_U32 y = 0; y < height; y++) {
            memcpy((HI_VOID*)dst_pointer, (HI_VOID*)src_pointer, sizeof(HI_U8) * width);
            dst_pointer += pstStLk->stTmp.au32Stride[0];
            src_pointer += img.step[0];
        }
    }

    HI_S32 NNIEOpticalFlow::_trans_hi2pyr(IVE_LK_TRACK_PTR pstStLk, IVE_SRC_IMAGE_S* pyr){
        IVE_HANDLE hIveHandle;
        HI_S32 s32Ret = HI_SUCCESS;
        IVE_DMA_CTRL_S stDmaCtrl;
        memset(&stDmaCtrl, 0, sizeof(stDmaCtrl));
        stDmaCtrl.enMode = IVE_DMA_MODE_DIRECT_COPY;

        IVE_FILTER_CTRL_S stFltCtrl = {{
                1, 2, 3, 2, 1,
                2, 5, 6, 5, 2,
                3, 6, 8, 6, 3,
                2, 5, 6, 5, 2,
                1, 2, 3, 2, 1
            }, 8
        };

        for(int i = 0 ; i < 25; i++) {
            stFltCtrl.as8Mask[i] *= 3;
        }

        s32Ret = HI_MPI_IVE_Filter(&hIveHandle, &pstStLk->stTmp, &pyr[0], &stFltCtrl,  HI_TRUE);

        HI_BOOL bFinish = HI_FALSE;
        HI_BOOL bBlock = HI_TRUE;
        assert_throw(HI_SUCCESS == s32Ret, "Error, HI_MPI_IVE_Filter failed!");
        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        while(HI_ERR_IVE_QUERY_TIMEOUT == s32Ret){
            usleep(100);
            s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        }
        assert_throw(HI_SUCCESS == s32Ret, "Error, HI_MPI_IVE_Filter HI_MPI_IVE_Query failed!");

        for (HI_U32 k = 1; k < pstStLk->stLkPyrCtrl.u8MaxLevel; k++) {
            s32Ret = _trans_hi2pyr_helper(pstStLk, &pyr[k - 1], &pyr[k]);
            assert_throw(HI_SUCCESS == s32Ret, "Error, _trans_hi2pyr_helper failed!");
        }
        return s32Ret;
    }

    HI_S32 NNIEOpticalFlow::_trans_hi2pyr_helper(IVE_LK_TRACK_PTR pstStLk, IVE_SRC_IMAGE_S* pstSrc,
                                            IVE_DST_IMAGE_S* pstDst) {
        HI_S32 s32Ret = HI_SUCCESS;
        IVE_HANDLE hIveHandle;
        IVE_DMA_CTRL_S  stDmaCtrl = {IVE_DMA_MODE_INTERVAL_COPY,
                                    0, 2, 1, 2
                                    };
        IVE_FILTER_CTRL_S stFltCtrl = {{
                1, 2, 3, 2, 1,
                2, 5, 6, 5, 2,
                3, 6, 8, 6, 3,
                2, 5, 6, 5, 2,
                1, 2, 3, 2, 1
            }, 3
        };

        IVE_RESIZE_CTRL_S stResizeCtrl = {
            hiIVE_RESIZE_MODE_E::IVE_RESIZE_MODE_LINEAR,
            {
                pstDst->au64PhyAddr[0],
                pstDst->au64VirAddr[0],
                pstDst->au32Stride[0]*pstDst->u32Height
            },
            1
        };

        pstStLk->stTmp.u32Width = pstSrc->u32Width;
        pstStLk->stTmp.u32Height = pstSrc->u32Height;

        s32Ret = HI_MPI_IVE_Resize(&hIveHandle, pstSrc, pstDst, &stResizeCtrl, HI_TRUE);

        HI_BOOL bFinish = HI_FALSE;
        HI_BOOL bBlock = HI_TRUE;
        assert_throw(HI_SUCCESS == s32Ret, "Error, HI_MPI_IVE_Resize failed!");
        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        while(HI_ERR_IVE_QUERY_TIMEOUT == s32Ret){
            usleep(100);
            s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        }
        assert_throw(HI_SUCCESS == s32Ret, "Error, HI_MPI_IVE_Resize HI_MPI_IVE_Query failed!");

        return s32Ret;
    }

    void NNIEOpticalFlow::_trans_points2hi(const std::vector<POINT> &points, IVE_LK_TRACK_PTR pstStLk){
        IVE_POINT_S25Q7_S *pre_pts = SAMPLE_COMM_IVE_CONVERT_64BIT_ADDR(IVE_POINT_S25Q7_S, pstStLk->stPrevPts.u64VirAddr);
        int n_points_size = points.size();
        n_points_size = std::min(n_points_size, _get_hi_max_pts_num());

        HI_FLOAT tmp_x, tmp_y;
        for(HI_U16 cnt = 0; cnt < n_points_size; cnt++){
            tmp_x = (HI_FLOAT)(points[cnt].x);
            tmp_y = (HI_FLOAT)(points[cnt].y);
            pre_pts[cnt].s25q7X = (HI_S32)((HI_S32)(tmp_x + 0.5f) << 7);
            pre_pts[cnt].s25q7Y = (HI_S32)((HI_S32)(tmp_y + 0.5f) << 7);
        }
        pstStLk->stLkPyrCtrl.u16PtsNum = n_points_size;
    }

    void NNIEOpticalFlow::_trans_hi2points(IVE_LK_TRACK_PTR pstStLk, std::vector<POINT> &points, std::vector<uint8_t> &valid){
        points.clear();
        valid.clear();
        IVE_POINT_S25Q7_S *next_pts = SAMPLE_COMM_IVE_CONVERT_64BIT_ADDR(IVE_POINT_S25Q7_S, pstStLk->stNextPts.u64VirAddr);
        for(HI_U32 k = 0; k < pstStLk->stLkPyrCtrl.u16PtsNum; k++) {
            HI_FLOAT next_x = (HI_FLOAT)(next_pts[k].s25q7X >> 7) + (HI_FLOAT)(IVE_MASK & next_pts[k].s25q7X) / 128.;
            HI_FLOAT next_y = (HI_FLOAT)(next_pts[k].s25q7Y >> 7) + (HI_FLOAT)(IVE_MASK & next_pts[k].s25q7Y) / 128.;
            points.push_back(POINT(next_x, next_y));
            if(!(SAMPLE_COMM_IVE_CONVERT_64BIT_ADDR(HI_U8,pstStLk->stStatus.u64VirAddr))[k])
                valid.push_back(0);
            else
                valid.push_back(1);
        }
    }
}
