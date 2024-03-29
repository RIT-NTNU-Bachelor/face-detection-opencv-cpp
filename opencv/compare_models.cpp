#include <opencv2/opencv.hpp>
#include <C:/dlib-19.24/dlib/image_processing/frontal_face_detector.h>
#include <C:/dlib-19.24/dlib/dnn.h>
#include <C:/dlib-19.24/dlib/opencv/cv_image.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <mgl2/mgl.h>
using namespace cv;
using namespace std;
using namespace dlib;

// This function uses the standard C++ library, std::function, to accept any callable that fits the signature.
double benchmarkMethod(const std::function<void(cv::Mat&)>& method, cv::Mat& frame, int iterations = 10) {
    auto start = chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        method(frame);
    }
    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count() / iterations;
}

// This function should use cv::dnn::Net from OpenCV's DNN module.
void dnnDetection(cv::dnn::Net& dnnModel, cv::Mat& frame) {
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 117, 123), false, false);
    dnnModel.setInput(blob);
    cv::Mat detections = dnnModel.forward();
}

// Haar detection using OpenCV's CascadeClassifier.
void haarDetection(cv::CascadeClassifier& haarCascade, cv::Mat& frame) {
    std::vector<cv::Rect> faces;
    haarCascade.detectMultiScale(frame, faces, 1.1, 10);
}

// HOG detection using dlib.
void hogDetection(dlib::frontal_face_detector& hogDetector, cv::Mat& frame) {
    dlib::cv_image<unsigned char> dlibImg(frame);
    std::vector<dlib::rectangle> dets = hogDetector(dlibImg);
}

int main() {
    // Load models
    cv::CascadeClassifier haarCascade("../models/haarcascade_frontalface_default.xml");
    cv::dnn::Net dnnModel = cv::dnn::readNetFromCaffe("../models/deploy.prototxt", "../models/res10_300x300_ssd_iter_140000_fp16.caffemodel");
    dlib::frontal_face_detector hogDetector = dlib::get_frontal_face_detector();

    // Load and preprocess image
    cv::Mat image = cv::imread("../models/imgname.JPG");
    cv::resize(image, image, cv::Size(300, 300));

    // Convert to grayscale for Haar and HOG
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Benchmarking usage
    double dnnAvg = benchmarkMethod([&](cv::Mat& img) { dnnDetection(dnnModel, img); }, image, 10);
    double haarAvg = benchmarkMethod([&](cv::Mat& img) { haarDetection(haarCascade, img); }, grayImage, 10);
    double hogAvg = benchmarkMethod([&](cv::Mat& img) { hogDetection(hogDetector, img); }, grayImage, 10);

    // Print results
    cout << "Average time (in seconds) for Haar: " << haarAvg << endl;
    cout << "Average time (in seconds) for DNN: " << dnnAvg << endl;
    cout << "Average time (in seconds) for HOG: " << hogAvg << endl;

    // Calculate FPS values
    double haar_fps = 1 / haar_avg;
    double dnn_fps = 1 / dnn_avg;
    double hog_fps = 1 / hog_avg;

    cout haar_fps << " FPS for Haar" << endl;
    cout dnn_fps << " FPS for DNN" << endl;
    cout hog_fps << " FPS for HoG" << endl;

    // Create MathGL graph and data
    mglGraph gr;
    mglData dat(3);
    dat.a[0] = haar_fps;
    dat.a[1] = dnn_fps;
    dat.a[2] = hog_fps;

    // Create the bar graph
    gr.SubPlot(1, 1, 0);
    gr.Title("Speed Comparison of Face Detection Methods");
    gr.SetRanges(0, 3, 0, *std::max_element(dat.a, dat.a + 3) + 10);
    gr.Axis();
    gr.Bar(dat);
    gr.Labels(dat, "Haar DNN HoG");
    gr.Label('y', "Speed (FPS)", 0);
gr.WriteFrame("../models/trained_models/bar_graph.png");

    return 0;
}