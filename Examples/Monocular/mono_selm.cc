/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono> // B.B get the current date

#include<opencv2/core/core.hpp>

#include<System.h>

// B.B
#include <iomanip> // B.B set date format
#include "BBLogger.hpp"


// Banafshe Bamdad
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

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps);

// ./Examples/Monocular/mono_selm Vocabulary/ORBvoc.txt Examples/Monocular/TUM1_SELM.yaml /media/banafshe/Banafshe_2TB/Datasets/TUM/Testing_and_Debugging/xyz/rgbd_dataset_freiburg1_xyz
// ./Examples/Monocular/mono_selm Vocabulary/ORBvoc.txt Examples/Monocular/TUM1_SELM.yaml /media/banafshe/Banafshe_2TB/Datasets/TUM/Handheld_SLAM/rgbd_dataset_freiburg1_desk
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

    if(argc != 4) {
        cerr << endl << "Usage: ./Examples/Monocular/mono_selm path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // B.B Command-line arguments
    std::string bb_vocab_path = argv[1];
    std::string bb_setting_path = argv[2];
    std::string bb_seq_path = argv[3];

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(bb_seq_path) + "/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(bb_vocab_path, bb_setting_path, ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    double t_resize = 0.f;
    double t_track = 0.f;

    BBLogger::getInstance().log("--- " + bb_seq_path + " ---");
    BBLogger::getInstance().log("# of fimages: " + std::to_string(nImages));
    BBLogger::getInstance().log("-----------------------------------------");
    // Main loop
    cv::Mat im;
    for(int ni = 0; ni < nImages; ni++) {

        BBLogger::getInstance().log("Image ID: " + std::to_string(ni) + ", file name: " + vstrImageFilenames[ni]);

        // Read image from file
        im = cv::imread(string(bb_seq_path) + "/" + vstrImageFilenames[ni], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);

        // BLUR IMAGE
        // cv::blur(im, im, cv::Size(15, 15));

        double tframe = vTimestamps[ni];

        if(im.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(bb_seq_path) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f) {
            #ifdef REGISTER_TIMES
                #ifdef COMPILEDWITHC11
                        std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
                #else
                        std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
                #endif
            #endif

            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));

            #ifdef REGISTER_TIMES
                #ifdef COMPILEDWITHC11
                        std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
                #else
                        std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
                #endif
                        t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
                        SLAM.InsertResizeTime(t_resize);
            #endif
        } //B.B change image scale

        #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        #else
                std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
        #endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, tframe);

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

        // Wait to load the next frame
        double T = 0;
        if(ni < nImages - 1){
            T = vTimestamps[ni + 1] - tframe;
        } else if(ni > 0) {
            T = tframe - vTimestamps[ni - 1];
        }

        if(ttrack < T) {
            usleep((T - ttrack) * 1e6);
        }
    } // B.B Loop on images

    // Stop all threads
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

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM(createFilenameWithDate("KeyFrameTrajectory_monocular", "txt"));
    SLAM.SaveTrajectoryTUM(createFilenameWithDate("CameraTrajectory_monocular", "txt"));

    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}
