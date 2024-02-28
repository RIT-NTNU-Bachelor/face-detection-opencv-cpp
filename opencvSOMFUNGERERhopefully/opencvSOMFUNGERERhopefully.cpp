#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Function to calculate the distance from the camera to the face
float calculateDistance(int faceWidth, int knownWidth, float focalLength) {
    return (knownWidth * focalLength) / faceWidth;
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "ERROR: Could not open camera." << endl;
        return -1;
    }

    // Load the cascades
    CascadeClassifier faceCascade("../x64/Debug/haarcascade_frontalface_alt.xml");
    CascadeClassifier eyesCascade("../x64/Debug/haarcascade_eye_tree_eyeglasses.xml");

    if (faceCascade.empty() || eyesCascade.empty()) {
        cerr << "ERROR: Could not load cascades." << endl;
        return -1;
    }

    // Known width of a face in cm (adjust based on your use case)
    const float knownWidth = 14;

    // Focal length of the camera (can be calibrated using a reference object)
    const float focalLength = 600;

    Mat frame, gray;
    vector<Rect> faces;

    while (cap.read(frame)) {
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        // Detect faces
        faceCascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        for (const auto& face : faces) {
            // Draw a rectangle around the face
            rectangle(frame, face, Scalar(255, 0, 0), 2);

            // Calculate and display the distance to the face
            float distance = calculateDistance(face.width, knownWidth, focalLength);
            putText(frame, to_string(distance) + " cm", Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);

            // Detect eyes within the face ROI
            Mat faceROI = gray(face);
            vector<Rect> eyes;
            eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

            for (const auto& eye : eyes) {
                // Draw a rectangle around each eye
                Point eyeCenter(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2);
                rectangle(frame, Point(face.x + eye.x, face.y + eye.y), Point(face.x + eye.x + eye.width, face.y + eye.y + eye.height), Scalar(0, 255, 0), 2);
            }
        }

        imshow("Face and Eye Tracking", frame);

        if (waitKey(10) == 27) {
            break; // Exit if ESC is pressed
        }
    }

    return 0;
}

