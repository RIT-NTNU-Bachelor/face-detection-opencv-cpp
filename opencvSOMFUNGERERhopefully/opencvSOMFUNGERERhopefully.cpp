#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include<string>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    Mat video_stream; // Matrix to hold frames from video stream
    VideoCapture real_time(0); // Capturing video from default webcam

    if (!real_time.isOpened()) { // Check if we succeeded
        cerr << "ERROR: Could not open video stream." << endl;
        return -1;
    }

    namedWindow("Face Detection"); // Window to show the result
    string trained_classifier_location = "../x64/Debug/haarcascade_frontalface_alt.xml"; // XML Trained Classifier location
    CascadeClassifier faceDetector;

    if (!faceDetector.load(trained_classifier_location)) { // Loading the XML trained classifier
        cerr << "ERROR: Could not load classifier." << endl;
        return -1;
    }

    vector<Rect> faces; // Vector to hold detected faces

    while (true) {
        bool frameRead = real_time.read(video_stream); // Reading frames from camera

        if (!frameRead) {
            cerr << "ERROR: Could not read frame." << endl;
            break;
        }

        // Resize the frame for faster processing
        Mat resizedFrame;
        float scale = 0.5; 
        resize(video_stream, resizedFrame, Size(), scale, scale);

        // Convert the frame to grayscale and equalize histogram for better detection
        Mat gray;
        cvtColor(resizedFrame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        // Detect faces with adjusted parameters
        faceDetector.detectMultiScale(gray, faces, 1.1, 4, CASCADE_SCALE_IMAGE, Size(60, 60)); // Adjust the minimum size if needed

        // Draw rectangles around the faces
        for (int i = 0; i < faces.size(); i++) {
            Rect scaledFace(faces[i].x / scale, faces[i].y / scale, faces[i].width / scale, faces[i].height / scale);
            rectangle(video_stream, scaledFace, Scalar(255, 0, 255), 2, 8, 0);
        }

        imshow("Face Detection", video_stream); // Display the frame with detected faces

        if (waitKey(10) == 27) { // Exit loop if 'ESC' is pressed
            break;
        }
    }
    real_time.release(); // Release the video capture object
    destroyAllWindows(); // Close all OpenCV windows
    return 0;
}
