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

    // Avg width of a face in cm 
    const float knownWidth = 14;

    // Focal length of the camera in cm
    const float focalLength = 600;

    Mat frame, gray;
    vector<Rect> faces;
    Rect lastFace;
    int frameCounter = 0;

    while (cap.read(frame)) {
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        // Reduce frame size for faster processing
        Mat smallGray;
        float scale = 0.5;
        resize(gray, smallGray, Size(), scale, scale);

        // Detect faces every nth frame or if no face was detected in the last frame
        if (frameCounter % 5 == 0 || lastFace.area() == 0) {
            faceCascade.detectMultiScale(smallGray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
            if (!faces.empty()) {
                lastFace = faces[0];
            }
        }
        else {
            // Search in the vicinity of the last detected face
            Rect searchRegion = lastFace + Size(lastFace.width / 2, lastFace.height / 2);
            searchRegion -= Point(searchRegion.width / 4, searchRegion.height / 4);
            searchRegion &= Rect(0, 0, smallGray.cols, smallGray.rows); // Ensure ROI is within image bounds

            Mat roiGray = smallGray(searchRegion);
            faceCascade.detectMultiScale(roiGray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

            // Adjust face positions based on the search region
            for (auto& face : faces) {
                face += Point(searchRegion.x, searchRegion.y);
                lastFace = face;
            }
        }

        // Draw rectangles around the faces and detect eyes
        for (const auto& face : faces) {
            Rect scaledFace = face;
            scaledFace.x /= scale;
            scaledFace.y /= scale;
            scaledFace.width /= scale;
            scaledFace.height /= scale;

            // Calculate and display the distance to the face
            float distance = calculateDistance(scaledFace.width, knownWidth, focalLength);
            putText(frame, to_string(distance) + " cm", Point(scaledFace.x, scaledFace.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);

            // Draw a rectangle around the face
            rectangle(frame, scaledFace, Scalar(255, 0, 0), 2);

            // Detect eyes within the face ROI
            Mat faceROI = gray(scaledFace);
            vector<Rect> eyes;
            eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

            for (const auto& eye : eyes) {
                // Draw a rectangle around each eye
                Point eyeCenter(scaledFace.x + eye.x + eye.width / 2, scaledFace.y + eye.y + eye.height / 2);
                rectangle(frame, Point(scaledFace.x + eye.x, scaledFace.y + eye.y), Point(scaledFace.x + eye.x + eye.width, scaledFace.y + eye.y + eye.height), Scalar(0, 255, 0), 2);
            }
        }

        imshow("Face and Eye Tracking", frame);

        if (waitKey(10) == 27) {
            break; // Exit if ESC is pressed
        }

        frameCounter++;
    }

    return 0;
}