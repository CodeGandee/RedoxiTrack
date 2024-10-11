#pragma once

// from:
// https://github.com/opencv/opencv_zoo/blob/main/models/person_detection_mediapipe/demo.cpp

#include <vector>
#include <string>
#include <utility>

#include <opencv2/opencv.hpp>

namespace cv_person_det{

using namespace std;
using namespace cv;
using namespace dnn;

Mat getMediapipeAnchor();

class MPPersonDet {
private:
    Net net;
    string modelPath;
    Size inputSize;
    float scoreThreshold;
    float nmsThreshold;
    dnn::Backend backendId;
    dnn::Target targetId;
    int topK;
    Mat anchors;

public:
    MPPersonDet(string modPath, float nmsThresh = 0.5, float scoreThresh = 0.3, int tok = 1, dnn::Backend bId = DNN_BACKEND_DEFAULT, dnn::Target tId = DNN_TARGET_CPU) :
        modelPath(modPath), nmsThreshold(nmsThresh),
        scoreThreshold(scoreThresh), topK(tok),
        backendId(bId), targetId(tId)
    {
        this->inputSize = Size(224, 224);
        this->net = readNet(this->modelPath);
        this->net.setPreferableBackend(this->backendId);
        this->net.setPreferableTarget(this->targetId);
        this->anchors = getMediapipeAnchor();
    }

    pair<Mat, Size> preprocess(Mat img)
    {
        Mat blob;
        Image2BlobParams paramMediapipe;
        paramMediapipe.datalayout = DNN_LAYOUT_NCHW;
        paramMediapipe.ddepth = CV_32F;
        paramMediapipe.mean = Scalar::all(127.5);
        paramMediapipe.scalefactor = Scalar::all(1 / 127.5);
        paramMediapipe.size = this->inputSize;
        paramMediapipe.swapRB = true;
        paramMediapipe.paddingmode = DNN_PMODE_LETTERBOX;

        double ratio = min(this->inputSize.height / double(img.rows), this->inputSize.width / double(img.cols));
        Size padBias(0, 0);
        if (img.rows != this->inputSize.height || img.cols != this->inputSize.width)
        {
            // keep aspect ratio when resize
            Size ratioSize(int(img.cols * ratio), int(img.rows * ratio));
            int padH = this->inputSize.height - ratioSize.height;
            int padW = this->inputSize.width - ratioSize.width;
            padBias.width = padW / 2;
            padBias.height = padH / 2;
        }
        blob = blobFromImageWithParams(img, paramMediapipe);
        padBias = Size(int(padBias.width / ratio), int(padBias.height / ratio));
        return pair<Mat, Size>(blob, padBias);
    }

    Mat infer(Mat srcimg)
    {
        pair<Mat, Size> w = this->preprocess(srcimg);
        Mat inputBlob = get<0>(w);
        Size padBias = get<1>(w);
        this->net.setInput(inputBlob);
        vector<Mat> outs;
        this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
        Mat predictions = this->postprocess(outs, Size(srcimg.cols, srcimg.rows), padBias);
        return predictions;
    }

    Mat postprocess(vector<Mat> outputs, Size orgSize, Size padBias)
    {
        Mat score = outputs[1].reshape(0, outputs[1].size[0]);
        Mat boxLandDelta = outputs[0].reshape(outputs[0].size[0], outputs[0].size[1]);
        Mat boxDelta = boxLandDelta.colRange(0, 4);
        Mat landmarkDelta = boxLandDelta.colRange(4, boxLandDelta.cols);
        double scale = max(orgSize.height, orgSize.width);
        Mat mask = score < -100;
        score.setTo(-100, mask);
        mask = score > 100;
        score.setTo(100, mask);
        Mat deno;
        exp(-score, deno);
        divide(1.0, 1 + deno, score);
        boxDelta.colRange(0, 1) = boxDelta.colRange(0, 1) / this->inputSize.width;
        boxDelta.colRange(1, 2) = boxDelta.colRange(1, 2) / this->inputSize.height;
        boxDelta.colRange(2, 3) = boxDelta.colRange(2, 3) / this->inputSize.width;
        boxDelta.colRange(3, 4) = boxDelta.colRange(3, 4) / this->inputSize.height;
        Mat xy1 = (boxDelta.colRange(0, 2) - boxDelta.colRange(2, 4) / 2 + this->anchors) * scale;
        Mat xy2 = (boxDelta.colRange(0, 2) + boxDelta.colRange(2, 4) / 2 + this->anchors) * scale;
        Mat boxes;
        hconcat(xy1, xy2, boxes);
        vector< Rect2d > rotBoxes(boxes.rows);
        boxes.colRange(0, 1) = boxes.colRange(0, 1) - padBias.width;
        boxes.colRange(1, 2) = boxes.colRange(1, 2) - padBias.height;
        boxes.colRange(2, 3) = boxes.colRange(2, 3) - padBias.width;
        boxes.colRange(3, 4) = boxes.colRange(3, 4) - padBias.height;
        for (int i = 0; i < boxes.rows; i++)
        {
            rotBoxes[i] = Rect2d(Point2d(boxes.at<float>(i, 0), boxes.at<float>(i, 1)), Point2d(boxes.at<float>(i, 2), boxes.at<float>(i, 3)));
        }
        vector< int > keep;
        NMSBoxes(rotBoxes, score, this->scoreThreshold, this->nmsThreshold, keep, this->topK);
        if (keep.size() == 0)
            return Mat();
        int nbCols = landmarkDelta.cols + boxes.cols + 1;
        Mat candidates(int(keep.size()), nbCols, CV_32FC1);
        int row = 0;
        for (auto idx : keep)
        {
            candidates.at<float>(row, nbCols - 1) = score.at<float>(idx);
            boxes.row(idx).copyTo(candidates.row(row).colRange(0, 4));
            candidates.at<float>(row, 4) = (landmarkDelta.at<float>(idx, 0) / this->inputSize.width + this->anchors.at<float>(idx, 0)) * scale - padBias.width;
            candidates.at<float>(row, 5) = (landmarkDelta.at<float>(idx, 1) / this->inputSize.height + this->anchors.at<float>(idx, 1)) * scale - padBias.height;
            candidates.at<float>(row, 6) = (landmarkDelta.at<float>(idx, 2) / this->inputSize.width + this->anchors.at<float>(idx, 0)) * scale - padBias.width;
            candidates.at<float>(row, 7) = (landmarkDelta.at<float>(idx, 3) / this->inputSize.height + this->anchors.at<float>(idx, 1)) * scale - padBias.height;
            candidates.at<float>(row, 8) = (landmarkDelta.at<float>(idx, 4) / this->inputSize.width + this->anchors.at<float>(idx, 0)) * scale - padBias.width;
            candidates.at<float>(row, 9) = (landmarkDelta.at<float>(idx, 5) / this->inputSize.height + this->anchors.at<float>(idx, 1)) * scale - padBias.height;
            candidates.at<float>(row, 10) = (landmarkDelta.at<float>(idx, 6) / this->inputSize.width + this->anchors.at<float>(idx, 0)) * scale - padBias.width;
            candidates.at<float>(row, 11) = (landmarkDelta.at<float>(idx, 7) / this->inputSize.height + this->anchors.at<float>(idx, 1)) * scale - padBias.height;
            row++;
        }
        return candidates;

    }
};

}