/*
 * Author: Banafshe Bamdad
 * Created on Fri Jan 31 2025 11:46:00 CET
 *
 */

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono> // to get the current date

#include<opencv2/core/core.hpp>

#include<System.h>

#include <iomanip> // to set date format
#include "BBLogger.hpp"

std::string createFilenameWithDate(const std::string& baseFilename, std::string extension) {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);

    std::tm now_tm = *std::localtime(&now_c);

    // Format date as YYYYMMDD
    std::ostringstream date_stream;
    date_stream << std::put_time(&now_tm, "%Y%m%d%H%M");
    std::string date_str = date_stream.str();

    std::string newFilename = baseFilename + date_str + "." + extension;

    return newFilename;
}

using namespace std;

void LoadImages(const string &timestampsFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps);

// ./Examples/Monocular/mono_Incrowd-VI Vocabulary/ORBvoc.txt Examples/Monocular/InCrowd-VI_LeftCam.yaml /media/banafshe/664605c3-03d7-47b4-ad4f-f31a8f8418d4/home/banafshe/InCrowd-VI/IMS_Labs/sequences/IMS_TE21_LEA_lab2 left_images 
/**
 * right_images/ns/undistorted
 * right_images/ns/timestamps.txt // the same as the file names in the "undistorted" folder
 * 
 * left_images/ns/undistorted
 * left_images/ns/timestamps.txt // the same as the file names in the "undistorted" folder
 * 
 * rgb_images/ns/undistorted
 * rgb_images/ns/timestamps.txt
 */

int main(int argc, char **argv) {

    BBLogger::setFilename("/home/banafshe/SUPERSLAM3/my_logs/" + createFilenameWithDate("BB_monitoring", "log"));

    if(argc != 5) {
        cerr << endl << "Usage: ./Examples/Monocular/mono_Incrowd-VI path_to_vocabulary path_to_settings path_to_sequence  sensor_type(left_images|right_images|rgb_images)" << endl;
        return 1;
    }

    std::string bb_vocab_path = argv[1]; // Vocabulary/ORBvoc.txt
    std::string bb_setting_path = argv[2]; // e.g. Examples/Monocular/InCrowd-VI_LeftCam.yaml
    std::string bb_seq_path = argv[3]; //e.g /media/banafshe/664605c3-03d7-47b4-ad4f-f31a8f8418d4/home/banafshe/InCrowd-VI/IMS_Labs/sequences/IMS_TE21_LEA_lab2
    std::string bb_sensor_type = argv[4]; // left_images | right_images | rgb_images

    std::string imagePath = string(bb_seq_path) + "/" + bb_sensor_type + "/ns/undistorted";
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string timestampsFile = string(bb_seq_path) + "/" + bb_sensor_type + "/ns/timestamps.txt";
    LoadImages(timestampsFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    ORB_SLAM3::System SLAM(bb_vocab_path, bb_setting_path, ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "# Images in the sequence: " << nImages << endl << endl;

    double t_resize = 0.f;
    double t_track = 0.f;

    BBLogger::getInstance().log("--- " + imagePath + " ---");
    BBLogger::getInstance().log("# of fimages: " + std::to_string(nImages));
    BBLogger::getInstance().log("-----------------------------------------");
    
    cv::Mat im;
    for(int ni = 0; ni < nImages; ni++) {

        BBLogger::getInstance().log("Image ID: " + std::to_string(ni) + ", file name: " + vstrImageFilenames[ni]);

        // Read image from file
        im = cv::imread(string(imagePath) + "/" + vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);

        double tframe = vTimestamps[ni];

        if(im.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(imagePath) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f) {

            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));

        }

        #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        #else
                std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
        #endif

        std::cout << std::endl << "B.B In mono_Incrowd-VI class. Before SLAM.TrackMonocular." << std::endl;

        SLAM.TrackMonocular(im, tframe);

        std::cout << std::endl << "B.B In mono_Incrowd-VI class. Before SLAM.TrackMonocular. Press Enter ...";cin.get();

        #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        #else
                std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
        #endif

        #ifdef REGISTER_TIMES
                t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
                SLAM.InsertTrackTime(t_track);
        #endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

        double T = 0;
        if(ni < nImages - 1){
            T = vTimestamps[ni + 1] - tframe;
        } else if(ni > 0) {
            T = tframe - vTimestamps[ni - 1];
        }

        if(ttrack < T) {
            usleep((T - ttrack) * 1e6);
        }
    }

    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for(int ni = 0; ni < nImages; ni++) {
        totaltime += vTimesTrack[ni];
    }

    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    SLAM.SaveKeyFrameTrajectoryTUM(createFilenameWithDate("KeyFrameTrajectory_monocular", "txt"));
    SLAM.SaveTrajectoryTUM(createFilenameWithDate("CameraTrajectory_monocular", "txt"));

    return 0;
}

void LoadImages(const string &timestampsFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps) {

    ifstream f(timestampsFile.c_str());

    if (!f.is_open()) {
        cerr << "Error: Unable to open file " << timestampsFile << endl;
        return;
    }

    string s;
    while (getline(f, s)) {
        if (!s.empty()) {
            stringstream ss(s);
            double t;
            ss >> t;

            vTimestamps.push_back(t);

            string sImage = s + ".png";
            vstrImageFilenames.push_back(sImage);
        }
    }
}

