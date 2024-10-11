#pragma once
// https://github.com/opencv/opencv_zoo/blob/main/models/object_detection_yolox/demo.cpp

#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace cv_yolox
{

using namespace std;
using namespace cv;
using namespace dnn;

const vector<string> labelYolox = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

class YoloX
{
  private:
    Net net;
    string modelPath;
    Size inputSize;
    float confThreshold;
    float nmsThreshold;
    float objThreshold;
    dnn::Backend backendId;
    dnn::Target targetId;
    int num_classes;
    vector<int> strides;
    Mat expandedStrides;
    Mat grids;

  public:
    YoloX(string modPath, float confThresh = 0.35, float nmsThresh = 0.5, float objThresh = 0.5, dnn::Backend bId = DNN_BACKEND_DEFAULT, dnn::Target tId = DNN_TARGET_CPU)
        : modelPath(modPath), confThreshold(confThresh),
          nmsThreshold(nmsThresh), objThreshold(objThresh),
          backendId(bId), targetId(tId)
    {
        this->num_classes = int(labelYolox.size());
        this->net = readNet(modelPath);
        this->inputSize = Size(640, 640);
        this->strides = vector<int>{8, 16, 32};
        this->net.setPreferableBackend(this->backendId);
        this->net.setPreferableTarget(this->targetId);
        this->generateAnchors();
    }

    Mat preprocess(Mat img)
    {
        Mat blob;
        Image2BlobParams paramYolox;
        paramYolox.datalayout = DNN_LAYOUT_NCHW;
        paramYolox.ddepth = CV_32F;
        paramYolox.mean = Scalar::all(0);
        paramYolox.scalefactor = Scalar::all(1);
        paramYolox.size = Size(img.cols, img.rows);
        paramYolox.swapRB = true;

        blob = blobFromImageWithParams(img, paramYolox);
        return blob;
    }

    Mat infer(Mat srcimg)
    {
        Mat inputBlob = this->preprocess(srcimg);

        this->net.setInput(inputBlob);
        vector<Mat> outs;
        this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

        Mat predictions = this->postprocess(outs[0]);
        return predictions;
    }

    Mat postprocess(Mat outputs)
    {
        Mat dets = outputs.reshape(0, outputs.size[1]);
        Mat col01;
        add(dets.colRange(0, 2), this->grids, col01);
        Mat col23;
        exp(dets.colRange(2, 4), col23);
        vector<Mat> col = {col01, col23};
        Mat boxes;
        hconcat(col, boxes);
        float *ptr = this->expandedStrides.ptr<float>(0);
        for (int r = 0; r < boxes.rows; r++, ptr++) {
            boxes.rowRange(r, r + 1) = *ptr * boxes.rowRange(r, r + 1);
        }
        // get boxes
        Mat boxes_xyxy(boxes.rows, boxes.cols, CV_32FC1, Scalar(1));
        Mat scores = dets.colRange(5, dets.cols).clone();
        vector<float> maxScores(dets.rows);
        vector<int> maxScoreIdx(dets.rows);
        vector<Rect2d> boxesXYXY(dets.rows);

        for (int r = 0; r < boxes_xyxy.rows; r++, ptr++) {
            boxes_xyxy.at<float>(r, 0) = boxes.at<float>(r, 0) - boxes.at<float>(r, 2) / 2.f;
            boxes_xyxy.at<float>(r, 1) = boxes.at<float>(r, 1) - boxes.at<float>(r, 3) / 2.f;
            boxes_xyxy.at<float>(r, 2) = boxes.at<float>(r, 0) + boxes.at<float>(r, 2) / 2.f;
            boxes_xyxy.at<float>(r, 3) = boxes.at<float>(r, 1) + boxes.at<float>(r, 3) / 2.f;
            // get scores and class indices
            scores.rowRange(r, r + 1) = scores.rowRange(r, r + 1) * dets.at<float>(r, 4);
            double minVal, maxVal;
            Point maxIdx;
            minMaxLoc(scores.rowRange(r, r + 1), &minVal, &maxVal, nullptr, &maxIdx);
            maxScoreIdx[r] = maxIdx.x;
            maxScores[r] = float(maxVal);
            boxesXYXY[r].x = boxes_xyxy.at<float>(r, 0);
            boxesXYXY[r].y = boxes_xyxy.at<float>(r, 1);
            boxesXYXY[r].width = boxes_xyxy.at<float>(r, 2);
            boxesXYXY[r].height = boxes_xyxy.at<float>(r, 3);
        }

        vector<int> keep;
        NMSBoxesBatched(boxesXYXY, maxScores, maxScoreIdx, this->confThreshold, this->nmsThreshold, keep);
        Mat candidates(int(keep.size()), 6, CV_32FC1);
        int row = 0;
        for (auto idx : keep) {
            boxes_xyxy.rowRange(idx, idx + 1).copyTo(candidates(Rect(0, row, 4, 1)));
            candidates.at<float>(row, 4) = maxScores[idx];
            candidates.at<float>(row, 5) = float(maxScoreIdx[idx]);
            row++;
        }
        if (keep.size() == 0)
            return Mat();
        return candidates;
    }

    void generateAnchors()
    {
        vector<tuple<int, int, int>> nb;
        int total = 0;

        for (auto v : this->strides) {
            int w = this->inputSize.width / v;
            int h = this->inputSize.height / v;
            nb.push_back(tuple<int, int, int>(w * h, w, v));
            total += w * h;
        }
        this->grids = Mat(total, 2, CV_32FC1);
        this->expandedStrides = Mat(total, 1, CV_32FC1);
        float *ptrGrids = this->grids.ptr<float>(0);
        float *ptrStrides = this->expandedStrides.ptr<float>(0);
        int pos = 0;
        for (auto le : nb) {
            int r = get<1>(le);
            for (int i = 0; i < get<0>(le); i++, pos++) {
                *ptrGrids++ = float(i % r);
                *ptrGrids++ = float(i / r);
                *ptrStrides++ = float((get<2>(le)));
            }
        }
    }
};

pair<Mat, double> letterBox(Mat srcimg, Size targetSize = Size(640, 640));
Mat unLetterBox(Mat bbox, double letterboxScale);
Mat visualize(Mat dets, Mat srcimg, double letterbox_scale, double fps = -1);

}; // namespace cv_yolox