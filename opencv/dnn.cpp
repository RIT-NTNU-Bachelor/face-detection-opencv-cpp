#include <algorithm>
#include <cctype>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const Scalar meanVal(104.0, 177.0, 123.0);

const string caffeConfigFile = "../models/deploy.prototxt";
const string caffeWeightFile = "../models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

const string tensorflowConfigFile = "../models/opencv_face_detector.pbtxt";
const string tensorflowWeightFile = "../models/opencv_face_detector_uint8.pb";

void detectFaceOpenCVDNN(Net net, Mat& frameOpenCVDNN, string framework) {
    if (frameOpenCVDNN.empty()) {
        throw std::runtime_error("Empty frame received for DNN processing.");
    }

    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;

    cv::Mat inputBlob;
    if (framework == "caffe") {
        inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
    }
    else {
        inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
    }

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > confidenceThreshold) {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
        }
    }
}

int main(int argc, const char** argv) {
    string videoFileName;
    string device;
    string framework;

    // Take arguments from command line
    if (argc == 4) {
        videoFileName = argv[1];
        device = argv[2];
        framework = argv[3];
    }
    else if (argc == 3) {
        videoFileName = argv[1];
        device = argv[2];
        framework = "caffe";
    }
    else if (argc == 2) {
        videoFileName = argv[1];
        device = "cpu";
        framework = "caffe";
    }
    else {
        videoFileName = "";
        device = "cpu";
        framework = "caffe";
    }

    // Convert device to uppercase
    std::transform(device.begin(), device.end(), device.begin(),
        [](unsigned char c) { return std::toupper(c); });

    cout << "Configuration:" << endl;
    cout << "Device - " << device << endl;
    cout << "Network type - " << (framework == "caffe" ? "Caffe" : "TensorFlow") << endl;
    cout << "Video file - " << (videoFileName.empty() ? "Camera Stream" : videoFileName) << endl;

    Net net;
    try {
        if (framework == "caffe") {
            net = readNetFromCaffe(caffeConfigFile, caffeWeightFile);
        }
        else {
            net = readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
        }

        if (net.empty()) {
            cerr << "Could not load the neural network." << endl;
            return -1;
        }

        // In OpenCV 4, DNN_BACKEND_CUDA and DNN_TARGET_CUDA are available
        if (device == "CPU") {
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
        }
        else {
            net.setPreferableBackend(DNN_BACKEND_CUDA);
            net.setPreferableTarget(DNN_TARGET_CUDA);
        }

        VideoCapture source;
        if (!videoFileName.empty()) {
            source.open(videoFileName);
        }
        else {
            // Try to open the default camera
            source.open(0);
        }

        if (!source.isOpened()) {
            cerr << "Failed to open video capture source." << endl;
            return -1;
        }

        Mat frame;
        double tt_opencvDNN = 0;
        double fpsOpencvDNN = 0;

        // Processing loop
        while (true) {
            source >> frame;
            if (frame.empty()) {
                cerr << "No frame captured from the source!" << endl;
                break;
            }

            double t = getTickCount();
            detectFaceOpenCVDNN(net, frame, framework);
            tt_opencvDNN = ((double)getTickCount() - t) / getTickFrequency();
            fpsOpencvDNN = 1 / tt_opencvDNN;

            stringstream ss;
            ss << "OpenCV DNN " << device << " FPS = " << fixed << setprecision(2) << fpsOpencvDNN;
            string label = ss.str();
            putText(frame, label, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(0, 0, 255), 4);

            imshow("OpenCV - DNN Face Detection", frame);

            int k = waitKey(5);
            if (k == 27) {
                destroyAllWindows();
                break;
            }
        }
    }
    // Catch OpenCV exceptions
    catch (const cv::Exception& e) {
        cerr << "OpenCV exception: " << e.what() << endl;
    }
    // Catch all other standard exceptions
    catch (const std::exception& e) {
        cerr << "Standard exception: " << e.what() << endl;
    }
    // Catch any other unhandled exceptions
    catch (...) {
        cerr << "Unknown exception occurred!" << endl;
    }

    return 0;
}