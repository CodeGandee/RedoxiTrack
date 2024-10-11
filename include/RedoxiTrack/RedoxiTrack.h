//
// Created by 18200 on 2022/3/7.
//
#pragma once

#include "RedoxiTrack/RedoxiTrackConfig.h"

#include "RedoxiTrack/detection/Detection.h"
#include "RedoxiTrack/detection/SingleDetection.h"
#include "RedoxiTrack/detection/TrackTarget.h"
#include "RedoxiTrack/detection/KalmanTrackTarget.h"
#include "RedoxiTrack/detection/DeepSortTrackTarget.h"
#include "RedoxiTrack/detection/BotsortTrackTarget.h"

#include "RedoxiTrack/tracker/TrackerBase.h"
#include "RedoxiTrack/tracker/OpticalFlowTracker.h"
#include "RedoxiTrack/tracker/KalmanTracker.h"
#include "RedoxiTrack/tracker/BotsortKalmanTracker.h"
#include "RedoxiTrack/tracker/DeepSortTracker.h"
#include "RedoxiTrack/tracker/BotsortTracker.h"

#include "RedoxiTrack/tracker/DeepSortMotionPrediction.h"
#include "RedoxiTrack/tracker/BotsortMotionPrediction.h"
#include "RedoxiTrack/tracker/SortMotionPrediction.h"
#include "RedoxiTrack/tracker/OpticalFlowMotionPrediction.h"
//#include "RedoxiTrack/nnie/NNIEOpticalFlow.h"
#include "RedoxiTrack/tracker/OpencvOpticalFlow.h"

#include "RedoxiTrack/tracker/DeepSortTrackerParam.h"
#include "RedoxiTrack/tracker/BotsortTrackerParam.h"
#include "RedoxiTrack/tracker/TrackerParam.h"
#include "RedoxiTrack/tracker/OpticalTrackerParam.h"
#include "RedoxiTrack/tracker/TrackingDecision.h"