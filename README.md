# Face Tracking using OpenCV and Dlib in C++
This repository contains the C++ codebase for face detection through 4 different models using OpenCV and Dlib. 
It is capable of processing images and video streams to identify and track human faces.

## Prerequisites
Before you can run the application, ensure you have the following prerequisites installed:
- OpenCV 4.x or later
- Dlib 19.x or later (integrated through local memory)
- A C++ compiler compatible with C++11 standards (GCC, Visual Studio, etc.)

## Structure
- models/: Contains pre-trained models for face detection, along with associated configuration files.
- opencv/: The directory where all the source .cpp files are located.
- videos/: A folder to place video files for testing.
- opencv.sln: Solution file for Visual Studio.

## Configuration
Please note there are known issues with the project setup. Users are required to modify the opencv.cpp file to 
correctly locate the OpenCV installation and models. The integration with Dlib is done through local memory due 
to some technical faults.

## Installation
1. Clone the repository to your local machine.
2. If you're using Visual Studio, open opencv.sln. Otherwise, set up your build system to include the opencv/ and models/ directories.
3. Locate the opencv.cpp file in the opencv/ directory. Adjust the file paths for the models and OpenCV directories as per your system's configuration.
4. Build the project using your C++ compiler.

## Running the Application
Once the application is built:
1. Place your test video files inside the videos/ directory, and specify the video or image you want to test.
2. Execute the built application, typically via command line or your IDE's run function.

## Troubleshooting
If you encounter issues, verify the following:
- The opencv.cpp file has been correctly modified to reflect your local setup.
- All paths to models and external libraries are correctly specified.
- Dlib is properly integrated through local memory; check for any missed configurations.

## Contribution
If you'd like to contribute to the project or suggest fixes for the existing setup issues, please submit a pull request or open an issue in the repository.

## License
Please refer to the LICENSE file for the terms and conditions associated with the use of this software.

