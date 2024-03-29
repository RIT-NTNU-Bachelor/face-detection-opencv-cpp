#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/dnn.h>
#include <dlib/opencv/cv_image.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>

using namespace cv;
using namespace std;
using namespace dlib;

// Define the MMOD network
//template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
//using residual_down = add_prev1<block<N, BN, 2, skip1<tag1<max_pool<3, 3, 2, 2, SUBNET>>>>>;
//
//template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
//using residual = add_prev2<block<N, BN, 1, tag2<block<N, BN, 1, tag1<skip1<SUBNET>>>>>>;
//
//template <int N, template <typename> class BN, int stride, typename SUBNET>
//using block = BN<con<N, 3, 3, stride, stride, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;
//
//template <int N, typename SUBNET> using res = relu<residual<block, N, affine, SUBNET>>;
//template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
//template <int N, typename SUBNET> using res_down = relu<residual_down<block, N, affine, SUBNET>>;
//
//using net_type = loss_mmod<con<1, 9, 9, 1, 1, res<512, res_down<256, res<256, res_down<128, res<128, res_down<64, res<64, input_rgb_image_pyramid<pyramid_down<6>>>>>>>>>>>;

double benchmarkMethod(function<void(Mat&)> method, Mat& frame, int iterations = 1000) {
    auto start = chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        method(frame);
    }
    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count() / iterations;
}

void dnnDetection(Net& dnnModel, Mat& frame) {
    Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 117, 123), false, false);
    dnnModel.setInput(blob);
    Mat detections = dnnModel.forward();
}

//void mmodDetection(net_type& mmodDetector, Mat& frame) {
//    cv_image<rgb_pixel> dlibImg(frame);
//    matrix<rgb_pixel> dlibMatrix;
//    assign_image(dlibMatrix, dlibImg);
//    auto dets = mmodDetector(dlibMatrix);
//}

void haarDetection(CascadeClassifier& haarCascade, Mat& frame) {
    vector<Rect> faces;
    haarCascade.detectMultiScale(frame, faces, 1.1, 10);
}

void hogDetection(frontal_face_detector& hogDetector, Mat& frame) {
    cv_image<unsigned char> dlibImg(frame);
    std::vector<rectangle> dets = hogDetector(dlibImg);
}

int main() {
    // Load models
    CascadeClassifier haarCascade("../src/models/trained_models/haarcascade_frontalface_default.xml");
    Net dnnModel = dnn::readNetFromCaffe("../src/models/trained_models/deploy.prototxt", "../src/models/trained_models/res10_300x300_ssd_iter_140000_fp16.caffemodel");
    //net_type mmodDetector;
    //deserialize("../src/models/trained_models/mmod_human_face_detector.dat") >> mmodDetector;
    frontal_face_detector hogDetector = get_frontal_face_detector();

    // Load and preprocess image
    Mat image = imread("../data/test_data/imgname");
    resize(image, image, Size(300, 300));

    // Convert to grayscale for Haar and HOG
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Benchmarking usage
    double dnnAvg = benchmarkMethod([&](Mat& img) { dnnDetection(dnnModel, img); }, image, 10);
    double haarAvg = benchmarkMethod([&](Mat& img) { haarDetection(haarCascade, img); }, grayImage, 10);
    //double mmodAvg = benchmarkMethod([&](Mat& img) { mmodDetection(mmodDetector, img); }, image, 10);
    double hogAvg = benchmarkMethod([&](Mat& img) { hogDetection(hogDetector, img); }, grayImage, 10);

    // Print results
    cout << "Average time (in seconds) for Haar: " << haarAvg << endl;
    cout << "Average time (in seconds) for DNN: " << dnnAvg << endl;
    //cout << "Average time (in seconds) for MMOD: " << mmodAvg << endl;
    cout << "Average time (in seconds) for HOG: " << hogAvg << endl;

    return 0;
}
