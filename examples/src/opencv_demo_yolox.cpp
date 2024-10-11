#include "opencv_demo_yolox.h"

namespace cv_yolox
{

static vector<pair<dnn::Backend, dnn::Target>> backendTargetPairs = {
    std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_OPENCV,
                                              dnn::DNN_TARGET_CPU),
    std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA,
                                              dnn::DNN_TARGET_CUDA),
    std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA,
                                              dnn::DNN_TARGET_CUDA_FP16),
    std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_TIMVX,
                                              dnn::DNN_TARGET_NPU),
    std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CANN,
                                              dnn::DNN_TARGET_NPU)};

static std::string keys =
    "{ help  h          |                                               | "
    "Print help message. }"
    "{ model m          | object_detection_yolox_2022nov.onnx           | "
    "Usage: Path to the model, defaults to object_detection_yolox_2022nov.onnx "
    " }"
    "{ input i          |                                               | Path "
    "to input image or video file. Skip this argument to capture frames from a "
    "camera.}"
    "{ confidence       | 0.5                                           | "
    "Class confidence }"
    "{ obj              | 0.5                                           | "
    "Enter object threshold }"
    "{ nms              | 0.5                                           | "
    "Enter nms IOU threshold }"
    "{ save s           | true                                          | "
    "Specify to save results. This flag is invalid when using camera. }"
    "{ vis v            | 1                                             | "
    "Specify to open a window for result visualization. This flag is invalid "
    "when using camera. }"
    "{ backend bt       | 0                                             | "
    "Choose one of computation backends: "
    "0: (default) OpenCV implementation + CPU, "
    "1: CUDA + GPU (CUDA), "
    "2: CUDA + GPU (CUDA FP16), "
    "3: TIM-VX + NPU, "
    "4: CANN + NPU}";

pair<Mat, double> letterBox(Mat srcimg, Size targetSize)
{
    Mat paddedImg(targetSize.height, targetSize.width, CV_32FC3,
                  Scalar::all(114.0));
    Mat resizeImg;

    double ratio = min(targetSize.height / double(srcimg.rows),
                       targetSize.width / double(srcimg.cols));
    resize(srcimg, resizeImg,
           Size(int(srcimg.cols * ratio), int(srcimg.rows * ratio)),
           INTER_LINEAR);
    resizeImg.copyTo(paddedImg(
        Rect(0, 0, int(srcimg.cols * ratio), int(srcimg.rows * ratio))));
    return pair<Mat, double>(paddedImg, ratio);
}

Mat unLetterBox(Mat bbox, double letterboxScale)
{
    return bbox / letterboxScale;
}

Mat visualize(Mat dets, Mat srcimg, double letterbox_scale, double fps)
{
    Mat resImg = srcimg.clone();

    if (fps > 0)
        putText(resImg, format("FPS: %.2f", fps), Size(10, 25),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

    for (int row = 0; row < dets.rows; row++)
    {
        Mat boxF = unLetterBox(dets(Rect(0, row, 4, 1)), letterbox_scale);
        Mat box;
        boxF.convertTo(box, CV_32S);
        float score = dets.at<float>(row, 4);
        int clsId = int(dets.at<float>(row, 5));

        int x0 = box.at<int>(0, 0);
        int y0 = box.at<int>(0, 1);
        int x1 = box.at<int>(0, 2);
        int y1 = box.at<int>(0, 3);

        string text = format("%s : %f", labelYolox[clsId].c_str(), score * 100);
        int font = FONT_HERSHEY_SIMPLEX;
        int baseLine = 0;
        Size txtSize = getTextSize(text, font, 0.4, 1, &baseLine);
        rectangle(resImg, Point(x0, y0), Point(x1, y1), Scalar(0, 255, 0), 2);
        rectangle(resImg, Point(x0, y0 + 1),
                  Point(x0 + txtSize.width + 1, y0 + int(1.5 * txtSize.height)),
                  Scalar(255, 255, 255), -1);
        putText(resImg, text, Point(x0, y0 + txtSize.height), font, 0.4,
                Scalar(0, 0, 0), 1);
    }

    return resImg;
}

#ifdef _YOU_DO_NOT_NEED_THIS_
int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    parser.about("Use this script to run Yolox deep learning networks in "
                 "opencv_zoo using OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string model = parser.get<String>("model");
    float confThreshold = parser.get<float>("confidence");
    float objThreshold = parser.get<float>("obj");
    float nmsThreshold = parser.get<float>("nms");
    bool vis = parser.get<bool>("vis");
    bool save = parser.get<bool>("save");
    int backendTargetid = parser.get<int>("backend");

    if (model.empty())
    {
        CV_Error(Error::StsError, "Model file " + model + " not found");
    }

    YoloX modelNet(model, confThreshold, nmsThreshold, objThreshold,
                   backendTargetPairs[backendTargetid].first,
                   backendTargetPairs[backendTargetid].second);
    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(samples::findFile(parser.get<String>("input")));
    else
        cap.open(0);
    if (!cap.isOpened())
        CV_Error(Error::StsError, "Cannot open video or file");
    Mat frame, inputBlob;
    double letterboxScale;

    static const std::string kWinName = model;
    int nbInference = 0;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "Frame is empty" << endl;
            waitKey();
            break;
        }
        pair<Mat, double> w = letterBox(frame);
        inputBlob = get<0>(w);
        letterboxScale = get<1>(w);
        TickMeter tm;
        tm.start();
        Mat predictions = modelNet.infer(inputBlob);
        tm.stop();
        cout << "Inference time: " << tm.getTimeMilli() << " ms\n";
        Mat img = visualize(predictions, frame, letterboxScale, tm.getFPS());
        if (save && parser.has("input"))
        {
            imwrite("result.jpg", img);
        }
        if (vis)
        {
            imshow(kWinName, img);
        }
    }
    return 0;
}
#endif // _YOU_DO_NOT_NEED_THIS_

}; // namespace cv_yolox
