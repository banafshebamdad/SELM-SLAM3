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

#define WITH_TICTOC
#include "tictoc.hpp"

#include "Tracking.h"

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "Optimizer.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "MLPnPsolver.h"
#include "GeometricTools.h"

#include <iostream>
#include <fstream> // B.B to write in a text file

#include <mutex>
#include <chrono>
#include <iomanip> // B.B To wtite date and time in a text file.provides various manipulators and functions that allow you to control the formatting of input and output streams, such as setting the width of fields, specifying the precision of floating-point numbers, and formatting dates and times.

// B.B
#include "BBLGMatcher.hpp"
#include "BBLogger.hpp"

using namespace std;
using namespace SELMSLAM;

namespace ORB_SLAM3
{


Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame*>(NULL))
{
    // Load camera parameters from settings file
    if(settings){
        newParameterLoader(settings); // B.B suspicious for 
    } else {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool b_parse_cam = ParseCamParamFile(fSettings);
        if(!b_parse_cam)
        {
            std::cout << "*Error with the camera parameters in the config file*" << std::endl;
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if(!b_parse_orb)
        {
            std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
        }

        bool b_parse_imu = true;
        if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO || sensor==System::IMU_RGBD)
        {
            b_parse_imu = ParseIMUParamFile(fSettings);
            if(!b_parse_imu)
            {
                std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
            }

            mnFramesToResetIMU = mMaxFrames;
        }

        if(!b_parse_cam || !b_parse_orb || !b_parse_imu)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    initID = 0; lastID = 0;
    mbInitWith3KFs = false;
    mnNumDataset = 0;

    vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
    std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
    for(GeometricCamera* pCam : vpCams) {
        std::cout << "Camera " << pCam->GetId();
        if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            std::cout << " is pinhole" << std::endl;
        }
        else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            std::cout << " is fisheye" << std::endl;
        }
        else
        {
            std::cout << " is unknown" << std::endl;
        }
    }

#ifdef REGISTER_TIMES
    vdRectStereo_ms.clear();
    vdResizeImage_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();
#endif
}

#ifdef REGISTER_TIMES
double calcAverage(vector<double> v_times)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File()
{
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<mpLocalMapper->vdLMTotal_ms.size(); ++i)
    {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
          << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
          << mpLocalMapper->vdKFCullingSync_ms[i] <<  "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for(int i=0; i<mpLocalMapper->vdLBASync_ms.size(); ++i)
    {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
          << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }


    f.close();
}

void Tracking::TrackStats2File()
{
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

    for(int i=0; i<vdTrackTotal_ms.size(); ++i)
    {
        double stereo_rect = 0.0;
        if(!vdRectStereo_ms.empty())
        {
            stereo_rect = vdRectStereo_ms[i];
        }

        double resize_image = 0.0;
        if(!vdResizeImage_ms.empty())
        {
            resize_image = vdResizeImage_ms[i];
        }

        double stereo_match = 0.0;
        if(!vdStereoMatch_ms.empty())
        {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if(!vdIMUInteg_ms.empty())
        {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
          << vdPosePred_ms[i] <<  "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats()
{
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();


    ofstream f;
    f.open("ExecMean.txt");
    f << fixed;
    //Report the mean and std of each one
    std::cout << std::endl << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    f << "OpenCV version: " << CV_VERSION << endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if(!vdRectStereo_ms.empty())
    {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdResizeImage_ms.empty())
    {
        average = calcAverage(vdResizeImage_ms);
        deviation = calcDeviation(vdResizeImage_ms, average);
        std::cout << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
        f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if(!vdStereoMatch_ms.empty())
    {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdIMUInteg_ms.empty())
    {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Local Mapping" << std::endl << std::endl;
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBA_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
    std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
    std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Map complexity" << std::endl;
    std::cout << "KFs in map: " << mpAtlas->GetAllKeyFrames().size() << std::endl;
    std::cout << "MPs in map: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBestMap = vpMaps[0];
    for(int i=1; i<vpMaps.size(); ++i)
    {
        if(pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size())
        {
            pBestMap = vpMaps[i];
        }
    }

    f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

    f << "---------------------------" << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdDataQuery_ms);
    deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdEstSim3_ms);
    deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdPRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
    f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
    f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
    f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nLoop << std::endl;
    average = calcAverage(mpLoopClosing->vnLoopKFs);
    deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
    f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
    f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
    f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
    f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nMerges << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeKFs);
    deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeMPs);
    deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdGBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
    f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
    deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
    f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
    f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    std::cout << "Num abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAKFs);
    deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAMPs);
    deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f.close();

}

#endif

Tracking::~Tracking()
{
    //f_track_stats.close();

}

void Tracking::newParameterLoader(Settings *settings) {
    mpCamera = settings->camera1();
    mpCamera = mpAtlas->AddCamera(mpCamera);

    if(settings->needToUndistort()) {
        mDistCoef = settings->camera1DistortionCoef();
    } else {
        mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    }

    //TODO: missing image scaling and rectification
    mImageScale = 1.0f;

    mK = cv::Mat::eye(3,3,CV_32F);
    mK.at<float>(0,0) = mpCamera->getParameter(0);
    mK.at<float>(1,1) = mpCamera->getParameter(1);
    mK.at<float>(0,2) = mpCamera->getParameter(2);
    mK.at<float>(1,2) = mpCamera->getParameter(3);

    mK_.setIdentity();
    mK_(0,0) = mpCamera->getParameter(0);
    mK_(1,1) = mpCamera->getParameter(1);
    mK_(0,2) = mpCamera->getParameter(2);
    mK_(1,2) = mpCamera->getParameter(3);

    if((mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD) &&
        settings->cameraType() == Settings::KannalaBrandt){
        mpCamera2 = settings->camera2();
        mpCamera2 = mpAtlas->AddCamera(mpCamera2);

        mTlr = settings->Tlr();

        mpFrameDrawer->both = true;
    }

    // B.B
    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD ){
        mbf = settings->bf();
        mThDepth = settings->b() * settings->thDepth();
    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD){
        mDepthMapFactor = settings->depthMapFactor();
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    mMinFrames = 0;
    mMaxFrames = settings->fps();
    mbRGB = settings->rgb();

    //ORB parameters
    int nFeatures = settings->nFeatures();
    int nLevels = settings->nLevels();
    float fIniThFAST = settings->initThFAST();
    float fMinThFAST = settings->minThFAST();
    float fScaleFactor = settings->scaleFactor();

    #ifndef USE_SELM_EXTRACTOR
        mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST); // B.B suspicious

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

        if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
            mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    #endif

    //IMU parameters
    Sophus::SE3f Tbc = settings->Tbc();
    mInsertKFsLost = settings->insertKFsWhenLost();
    mImuFreq = settings->imuFrequency();
    mImuPer = 0.001; //1.0 / (double) mImuFreq;     //TODO: ESTO ESTA BIEN?
    float Ng = settings->noiseGyro();
    float Na = settings->noiseAcc();
    float Ngw = settings->gyroWalk();
    float Naw = settings->accWalk();

    const float sf = sqrt(mImuFreq);
    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if(sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(0) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(1) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(2) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(3) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(b_miss_params)
        {
            return false;
        }

        if(mImageScale != 1.f)
        {
            // K matrix parameters must be scaled.
            fx = fx * mImageScale;
            fy = fy * mImageScale;
            cx = cx * mImageScale;
            cy = cy * mImageScale;
        }

        vector<float> vCamCalib{fx,fy,cx,cy};

        mpCamera = new Pinhole(vCamCalib);

        mpCamera = mpAtlas->AddCamera(mpCamera);

        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- Image scale: " << mImageScale << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
        std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;


        std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
        std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

        if(mDistCoef.rows==5)
            std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

        mK = cv::Mat::eye(3,3,CV_32F);
        mK.at<float>(0,0) = fx;
        mK.at<float>(1,1) = fy;
        mK.at<float>(0,2) = cx;
        mK.at<float>(1,2) = cy;

        mK_.setIdentity();
        mK_(0,0) = fx;
        mK_(1,1) = fy;
        mK_(0,2) = cx;
        mK_(1,2) = cy;
    }
    else if(sCameraName == "KannalaBrandt8")
    {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            k3 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if(!node.empty() && node.isReal())
        {
            k4 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(!b_miss_params)
        {
            if(mImageScale != 1.f)
            {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
            mpCamera = new KannalaBrandt8(vCamCalib);
            mpCamera = mpAtlas->AddCamera(mpCamera);
            std::cout << "- Camera: Fisheye" << std::endl;
            std::cout << "- Image scale: " << mImageScale << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << k1 << std::endl;
            std::cout << "- k2: " << k2 << std::endl;
            std::cout << "- k3: " << k3 << std::endl;
            std::cout << "- k4: " << k4 << std::endl;

            mK = cv::Mat::eye(3,3,CV_32F);
            mK.at<float>(0,0) = fx;
            mK.at<float>(1,1) = fy;
            mK.at<float>(0,2) = cx;
            mK.at<float>(1,2) = cy;

            mK_.setIdentity();
            mK_(0,0) = fx;
            mK_(1,1) = fy;
            mK_(0,2) = cx;
            mK_(1,2) = cy;
        }

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD){
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if(!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if(!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if(!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if(!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if(!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if(!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if(!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if(!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }


            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                leftLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                leftLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                rightLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                rightLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
            }

            node = fSettings["Tlr"];
            cv::Mat cvTlr;
            if(!node.empty())
            {
                cvTlr = node.mat();
                if(cvTlr.rows != 3 || cvTlr.cols != 4)
                {
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    b_miss_params = true;
                }
            }
            else
            {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                b_miss_params = true;
            }

            if(!b_miss_params)
            {
                if(mImageScale != 1.f)
                {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;

                    leftLappingBegin = leftLappingBegin * mImageScale;
                    leftLappingEnd = leftLappingEnd * mImageScale;
                    rightLappingBegin = rightLappingBegin * mImageScale;
                    rightLappingEnd = rightLappingEnd * mImageScale;
                }

                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx,fy,cx,cy,k1,k2,k3,k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);
                mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                mTlr = Converter::toSophus(cvTlr);

                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- Image scale: " << mImageScale << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                std::cout << "- mTlr: \n" << cvTlr << std::endl;

                std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
            }
        }

        if(b_miss_params)
        {
            return false;
        }

    }
    else
    {
        std::cerr << "*Not Supported Camera Sensor*" << std::endl;
        std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD )
    {
        cv::FileNode node = fSettings["Camera.bf"];
        if(!node.empty() && node.isReal())
        {
            mbf = node.real();
            if(mImageScale != 1.f)
            {
                mbf *= mImageScale;
            }
        }
        else
        {
            std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
    {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if(!node.empty()  && node.isReal())
        {
            mThDepth = node.real();
            mThDepth = mbf*mThDepth/fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }
        else
        {
            std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }


    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
    {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if(!node.empty() && node.isReal())
        {
            mDepthMapFactor = node.real();
            if(fabs(mDepthMapFactor)<1e-5)
                mDepthMapFactor=1;
            else
                mDepthMapFactor = 1.0f/mDepthMapFactor;
        }
        else
        {
            std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    if(b_miss_params)
    {
        return false;
    }

    return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    int nFeatures, nLevels;
    float fIniThFAST, fMinThFAST;
    float fScaleFactor;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if(!node.empty() && node.isInt())
    {
        nFeatures = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if(!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if(!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if(!node.empty() && node.isReal())
    {
        fIniThFAST = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if(!node.empty() && node.isReal())
    {
        fMinThFAST = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat cvTbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        cvTbc = node.mat();
        if(cvTbc.rows != 4 || cvTbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }
    cout << endl;
    cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
    Eigen::Matrix<float,4,4,Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
    Sophus::SE3f Tbc(eigTbc);

    node = fSettings["InsertKFsWhenLost"];
    mInsertKFsLost = true;
    if(!node.empty() && node.isInt())
    {
        mInsertKFsLost = (bool) node.operator int();
    }

    if(!mInsertKFsLost)
        cout << "Do not insert keyframes when lost visual tracking " << endl;



    float Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        mImuFreq = node.operator int();
        mImuPer = 0.001; //1.0 / (double) mImuFreq;
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.fastInit"];
    mFastInit = false;
    if(!node.empty())
    {
        mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
    }

    if(mFastInit)
        cout << "Fast IMU initialization. Acceleration is not checked \n";

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(mImuFreq);
    cout << endl;
    cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);


    return true;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

bool Tracking::GetStepByStep()
{
    return bStepByStep;
}



Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    //cout << "GrabImageStereo" << endl;

    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if(mImGray.channels()==3)
    {
        //cout << "Image with 3 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        //cout << "Image with 4 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGRA2GRAY);
        }
    }

    //cout << "Incoming frame creation" << endl;

    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
    else if(mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    else if(mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);

    //cout << "Incoming frame ended" << endl;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif

    //cout << "Tracking start" << endl;
    Track();
    //cout << "Tracking end" << endl;

    return mCurrentFrame.GetPose();
}

/**
 * B.B 
 * This function processes an incoming RGB-D frame to estimate the camera's pose (3D rigid body transformation ) and update the tracking state.
*/
Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, string filename) {

    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels() == 3) {
        if(mbRGB)
            cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
    } else if(mImGray.channels() == 4) {
        if(mbRGB)
            cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F){ 
        // B.B converts the imDepth image to a 32-bit floating-point image and scales it by a factor of mDepthMapFactor.
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);
    }

    if (mSensor == System::RGBD) {

        // Banafshe Bamdad
        TIC
        #ifdef USE_SELM_EXTRACTOR
            mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpBBSPExtractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
        #else
            // B.B Achtung
            // B.B Pourquoi mCurrentFrame.N est-il zero?
            mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
        #endif
        TOC

    } else if(mSensor == System::IMU_RGBD) {
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    }

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

    #ifdef REGISTER_TIMES
        vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    #endif

    Track();

    // B.B to log the Reference KeyFrame ID for each frame
    if (mCurrentFrame.mpReferenceKF) {
        std::string bbLogFilePath = std::string(BBLOGFILE_PATH) + "BB_reference_KFs_RGBD.log";
        std::ofstream BBLogFile(bbLogFilePath, std::ios::app);
        if (BBLogFile.is_open()) {
            BBLogFile << endl << "FrameId: " << mCurrentFrame.mnId << "\tRefKFId: " << mCurrentFrame.mpReferenceKF->mnId;
            BBLogFile.close();
        }
    }

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename) {

    mImGray = im;
    if(mImGray.channels() == 3) {
        if(mbRGB)
            cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
    } else if(mImGray.channels() == 4) {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if (mSensor == System::MONOCULAR) {

        if(mState == NOT_INITIALIZED || mState == NO_IMAGES_YET || (lastID - initID) < mMaxFrames) {
            #ifdef USE_SELM_EXTRACTOR
                cout << endl << "B.B Before Frame Initialization with mpIniBBSPExtractor." << endl;
                mCurrentFrame = Frame(mImGray, timestamp, mpIniBBSPExtractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
            #else
                // B.B Frame constructor in ~ line 445
                mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
            #endif
        } else {
            #ifdef USE_SELM_EXTRACTOR
                cout << endl << "B.B Before Frame Initialization with mpBBSPExtractorLeft." << endl;
                mCurrentFrame = Frame(mImGray, timestamp, mpBBSPExtractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
            #else
                mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
            #endif
            
            ////////////////////////////////
            // TEST SUPERPOINTS DETECTION //
            ////////////////////////////////
            // TEST_EvaluateSuperpoints(mImGray);
        }

    } else if(mSensor == System::IMU_MONOCULAR) {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        {
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
        }
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
    }

    if (mState == NO_IMAGES_YET)
        t0 = timestamp;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

    #ifdef REGISTER_TIMES
        vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    #endif

    lastID = mCurrentFrame.mnId;

    Track();

    // B.B to log the Reference KeyFrame ID for each frame
    if (mCurrentFrame.mpReferenceKF) {
        std::string bbLogFilePath = std::string(BBLOGFILE_PATH) + "BB_reference_KFs_mono.log";
        std::ofstream BBLogFile(bbLogFilePath, std::ios::app);
        if (BBLogFile.is_open()) {
            BBLogFile << endl << "FrameId: " << mCurrentFrame.mnId << "\tRefKFId: " << mCurrentFrame.mpReferenceKF->mnId;
            BBLogFile.close();
        }
    }

    return mCurrentFrame.GetPose();
}

void Tracking::TEST_EvaluateSuperpoints(const cv::InputArray &_image)
{
    // if(mpSystem->SPF != nullptr)
    // {
    //     std::vector<cv::KeyPoint> Keypoints;
    //     cv::Mat Descriptors;
    //     mpSystem->SPF->detect(_image, Keypoints, Descriptors);

    //     // print some stats
    //     cv::Size s = Descriptors.size();
    //     int rows = s.height;
    //     int cols = s.width;
    //     cout << __PRETTY_FUNCTION__  << "--> Keypoints founded: " << Keypoints.size() << ", Descriptors founded: " << rows << ", Descriptors size: " << cols << endl;
    // }
    // else
    // {
    //     cout << __PRETTY_FUNCTION__ << "--> Trying to detect superpoints but superpoint detector is not initialized!" << endl;       
    // }
}   

void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU()
{

    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    if(mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    while(true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty())
            {
                IMU::Point* m = &mlQueueImuData.front();
                cout.precision(17);
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-mImuPer)
                {
                    mlQueueImuData.pop_front();
                }
                else if(m->t<mCurrentFrame.mTimeStamp-mImuPer)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }

    const int n = mvImuFromLastFrame.size()-1;
    if(n==0){
        cout << "Empty IMU measurements vector!!!\n";
        return;
    }

    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);

    for(int i=0; i<n; i++)
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

    mCurrentFrame.setIntegrated();

    //Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}


bool Tracking::PredictStateIMU()
{
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    if(mbMapUpdated && mpLastKeyFrame)
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else if(!mbMapUpdated)
    {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false;
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}


void Tracking::Track() {

    // 
    // B.B logging
    // 
    // B.B to monitor mState for each frame
    std::string BBLogFile_Path = std::string(BBLOGFILE_PATH) + "BB_monitoring.log";
    std::ofstream BBLogFile(BBLogFile_Path, std::ios::app);
    if (BBLogFile.is_open()) {
        auto now = std::chrono::system_clock::now();
        std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
        std::tm* localTime = std::localtime(&currentTime);
        std::stringstream ss;
        ss << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");
        BBLogFile << endl << "********** [" << ss.str() << "] **********";
        BBLogFile << endl << "\t***** FrameId: " << mCurrentFrame.mnId << " *****";
    }

    ////////////////////////////////////////////
    // CHECK AND SAVE TO FILE GPU STATS       //
    // torch::cuda::memory_stats(device=None) //
    ////////////////////////////////////////////

    printf("\n\n\n INIT TRACKING PIPELINE -------- (B.B start) ------------");
    cout << endl << "\t\tFrameId: " << mCurrentFrame.mnId << endl;

    std::set<KeyFrame*> bb_mspKeyFrames =  mpAtlas->GetCurrentMap()->mspKeyFrames;
    cout << endl << "B.B The number of keyframe in map: " << bb_mspKeyFrames.size() << ". Press Enter...";

    std::vector<KeyFrame*> bb_vpCandidateKFs(bb_mspKeyFrames.begin(), bb_mspKeyFrames.end());
    cout << endl << "B.B The number of candidate keyframes: " << bb_vpCandidateKFs.size() << ". Press Enter..."; 

    // B.B sort the vector based on KeyFrame.mnId
    std::sort(bb_vpCandidateKFs.begin(), bb_vpCandidateKFs.end(),
              [](const KeyFrame* kf1, const KeyFrame* kf2) {
                std::cout << "Comparing mnId: " << kf1->mnId << " and " << kf2->mnId << std::endl;
                return kf1->mnId < kf2->mnId;
              });

    std::reverse(bb_vpCandidateKFs.begin(), bb_vpCandidateKFs.end());
    for (const auto& keyFramePtr : bb_vpCandidateKFs) {
        std::cout << "sorted mnId: " << keyFramePtr->mnId << std::endl;
    }

    for (const auto& kfp: bb_mspKeyFrames) {
        cout << endl << "KF mnId: " << kfp->mnId;
    }

    // 
    // 
    //
    if (bStepByStep) {
        std::cout << "Tracking: Waiting to the next step" << std::endl;
        while(!mbStep && bStepByStep)
            usleep(500);
        mbStep = false;
    }

    if(mpLocalMapper->mbBadImu) {
        cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
        mpSystem->ResetActiveMap();
        return;
    }

    // B.B retrieves the current map
    Map* pCurrentMap = mpAtlas->GetCurrentMap();
    if(!pCurrentMap) {
        cout << "ERROR: There is not an active map in the atlas" << endl;
    } 

    // B.B mState = 1: NOT_INITIALIZED
    /**
     * B.B initialization process
     *  setting up the camera
     *  initializing keyframe data structures
     *  performing feature extraction and matching
     *  establishing the initial pose of the camera in the world frame
     * !!! ACHTUNG !!! Banafshe's orange idea: It seams the features have not been correctly extracted or established in keyframe data structure.
    */

    if(mState != NO_IMAGES_YET) {

        cout << endl << "B.B in Tracking::Track(), mState != NO_IMAGES_YET. mState= " << mState << endl;

        if(mLastFrame.mTimeStamp > mCurrentFrame.mTimeStamp) {
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        } else if(mCurrentFrame.mTimeStamp > mLastFrame.mTimeStamp + 1.0) {
            
            if(mpAtlas->isInertial()) {

                if(mpAtlas->isImuInitialized()) {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    if(!pCurrentMap->GetIniertialBA2()) {
                        mpSystem->ResetActiveMap();
                    } else {
                        CreateMapInAtlas();
                    }
                } else {
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap();
                }
                return;
            }

        }
    } // B.B End of if(mState != NO_IMAGES_YET) 

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame){
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());
    }

    if(mState == NO_IMAGES_YET) {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState = mState;

    BBLogFile << endl << "\tmState: " << mState;

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap) {
        #ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
        #endif
                PreintegrateIMU();
        #ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

                double timePreImu = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPreIMU - time_StartPreIMU).count();
                vdIMUInteg_ms.push_back(timePreImu);
        #endif

    }

    mbCreatedMap = false;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated = false;

    // B.B checks for changes in the map since the last update
    // B.B The map change index is a value that represents the state or version of the map.
    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();

    // B.B represents the map's state at the time of the last update.
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();

    // B.B if changes have occurred in the map since the last update.
    if(nCurMapChangeIndex > nMapChangeIndex) {

        // B.B updates the last map change index to the current map change index. 
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }

    // 
    // B.B The system is not initializaed
    // 
    if(mState == NOT_INITIALIZED) {

        if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {

           StereoInitialization();

        } else {
            cout << endl << "B.B --- before MonocularInitialization --- mState: " << mState;
            MonocularInitialization();
            cout << endl << "B.B --- after MonocularInitialization --- mState: " << mState;
        }

        mpFrameDrawer->Update(this); // B.B removed the comment from this line.

        if(mState != OK) { // If rightly initialized, mState=OK
            mLastFrame = Frame(mCurrentFrame);
            return;
        }

        if(mpAtlas->GetAllMaps().size() == 1) {
            mnFirstFrameId = mCurrentFrame.mnId;
        }

    } else {
        // System is initialized. Track Frame.
        bool bOK;

        #ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
        #endif

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // B.B section V.Tracking B. and C. of paper "ORB-SLAM: A Versatile and Accurate Monocular SLAM System"

        if(!mbOnlyTracking) {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if(mState == OK) {

                cout << endl << "B.B In Tracking::Track. Before CheckReplacedInLastFrame ...";

                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                // B.B tracking behavior depends on whether the system has velocity information and whether the IMU is initialized 
                if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                    
                    Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);

                    BBLogFile << endl << "\tTrack with Ref.KF.";

                    cout << endl << "B.B Track with Ref. KF. Press Enter...";
                    bOK = TrackReferenceKeyFrame();

                } else {
                    
                    Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);

                    /**
                     * B.B
                     * Paper: ORB-SLAM: A Versatile and Accurate Monocular SLAM System
                     * If tracking was successful for last frame, we use a constant 
                     * velocity motion model to predict the camera pose and perform
                     * a guided search of the map points observed in the last frame (ORBmatcher::SearchByProjection)
                    */

                    BBLogFile << endl << "\tTrack with Motion Model.";
                    bOK = TrackWithMotionModel();

                    if(!bOK) {
                        printf("TRACK: Track with motion model FAILED, try tracking with reference keyframe\n");
                        BBLogFile << endl << "\tTrack with Ref. KF. due to Motion Model failure.";
                        bOK = TrackReferenceKeyFrame();
                    }
                }

                BBLogFile << endl << "\tbOK: " << bOK;

                // B.B managing the system's state
                if (!bOK) {
                    if ( mCurrentFrame.mnId <= (mnLastRelocFrameId + mnFramesToResetIMU) &&
                            (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)) {
                        std::cout << "STATE LOST!!! !bOK" << std::endl;
                        mState = LOST;
                    } else if(pCurrentMap->KeyFramesInMap() > 10) {
                        cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                    } else {
                        mState = LOST;
                    }

                    BBLogFile << endl << "\tmState after managing the system's state due to bOK=False: " << mState;
                }
            } else { // B.B if(mState == OK)

                if (mState == RECENTLY_LOST) {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    bOK = true;

                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)) {
                        if(pCurrentMap->isImuInitialized())
                            PredictStateIMU();
                        else
                            bOK = false;

                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost) {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    } else {
                        // Relocalization
                        bOK = Relocalization();
                        //std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                        //std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                        if(mCurrentFrame.mTimeStamp - mTimeStampLost > 3.0f && !bOK) {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK = false;
                        }
                    }
                } else if (mState == LOST) {

                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                    if (pCurrentMap->KeyFramesInMap() < 10) {

                        // B.B indicating that the system is starting with a new map
                        mpSystem->ResetActiveMap();
                        Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
                    } else { // B.B If there are 10 or more keyframes in the map, create a new map in the atlas. 
                        CreateMapInAtlas();
                    }

                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return;
                }
            } // B.B end of else relevant to if(mState == OK)

        } else { // B.B if(!mbOnlyTracking)
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if(mState == LOST) {
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                bOK = Relocalization();
            } else {
                if(!mbVO) {
                    // In last frame we tracked enough MapPoints in the map
                    if(mbVelocity) {
                        bOK = TrackWithMotionModel();
                    } else {
                        bOK = TrackReferenceKeyFrame();
                    }
                } else {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    Sophus::SE3f TcwMM;
                    if(mbVelocity) {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.GetPose();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc) {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO) {
                            for(int i = 0; i < mCurrentFrame.N; i++) {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    } else if(bOKReloc) {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        if(!mCurrentFrame.mpReferenceKF) {
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
        #ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

            double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
            vdPosePred_ms.push_back(timePosePred);
        #endif

        #ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
        #endif

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking) {
            if(bOK) {
                // B.B The function returns true or false based on the number of inliers and the type of sensor being used.
                // B.B to determine whether the tracking is considered successful or not.

                cout << "B.B Systems is ready to enter to the TrackLocalMap module. Who cares?";
                bOK = TrackLocalMap();

            }
            if(!bOK) {
                cout << "Fail to track local map!" << endl;
                BBLogFile << endl << "\tFail to track local map.";
            }
        } else {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO){
                BBLogFile << endl << "\tPerform TrackLocalmap.";
                
                bOK = TrackLocalMap();

                BBLogFile << endl << "\tbOK after Tracking Local map: " << bOK; 
            } else {
                BBLogFile << endl << "\tthere are few matches to MapPoints in the map. not perform TrackLocalMap.";
            }
        }

        if(bOK) {
            mState = OK;
        } else if (mState == OK) {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2()) {
                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap();
                }

                mState = RECENTLY_LOST;
            } else
                mState = RECENTLY_LOST; // visual to lost

            /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {*/
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            //}
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        if((mCurrentFrame.mnId < (mnLastRelocFrameId + mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
            (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized()) {
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        if(pCurrentMap->isImuInitialized()) {
            if(bOK)
            {
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                {
                    cout << "RESETING FRAME!!!" << endl;
                    ResetFrameIMU();
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
                    mLastBias = mCurrentFrame.mImuBias;
            }
        }

        #ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

            double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
            vdLMTrack_ms.push_back(timeLMTrack);
        #endif

        // Update drawer
        mpFrameDrawer->Update(this);
        if(mCurrentFrame.isSet()) {
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose()); // B.B !!! ACHTUNG ACHTUNCH !!!
        }

        if(bOK || mState == RECENTLY_LOST) {
            // Update motion model
            if(mLastFrame.isSet() && mCurrentFrame.isSet()) {

                // B.B calculates the velocity of the camera's motion based on the transformation between the current and last frames
                Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse();
                mVelocity = mCurrentFrame.GetPose() * LastTwc;
                mbVelocity = true;
            } else {
                mbVelocity = false;
            }

            // B.B sets the current camera pose in the map drawer based on the pose of the current frame.
            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {  
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());
            }

            // Clean VO matches
            // B.B iterates through the feature points in the current frame and checks if they are associated with a map point 
            for(int i = 0; i < mCurrentFrame.N; i++) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP) {
                    // B.B If a map point has no observations (indicating that it's no longer valid), 
                    // B.B it updates the frame's mvbOutlier array to mark the corresponding feature as an inlier (not an outlier), and it sets the map point to NULL.
                    if(pMP->Observations() < 1) {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                    }
                }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++) {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            #ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
            #endif

            // B.B checks if a new keyframe is needed
            bool bNeedKF = NeedNewKeyFrame();

            // B.B to log if a new keyframe is needed
            std::string bbLogFilePath = std::string(BBLOGFILE_PATH) + "BB_if_KF_is_needed.log";
            std::ofstream BBLogFile(bbLogFilePath, std::ios::app);
            if (BBLogFile.is_open()) {
                BBLogFile << endl << "FrameID: " << mCurrentFrame.mnId << "\tbOK: " << boolalpha << bOK << "\tKF needed?: " << boolalpha << bNeedKF << "\t# Local KF: " << mvpLocalKeyFrames.size();
                BBLogFile.close();
            }

            // Check if we need to insert a new keyframe
            // if(bNeedKF && bOK)
            // B.B The decision to insert a new keyframe depends on various factors, such as the camera's motion, the number of observed features, or other criteria.
            if(bNeedKF && (bOK || (mInsertKFsLost && mState==RECENTLY_LOST && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)))) {

                CreateNewKeyFrame();
            }

            #ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

                double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
                vdNewKF_ms.push_back(timeNewKF);
            #endif


            // B.B high innovation: deviate significantly, residual
            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for(int i = 0; i < mCurrentFrame.N; i++) {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i]){
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState == LOST) {
            if(pCurrentMap->KeyFramesInMap()<=10) {
                mpSystem->ResetActiveMap();
                return;
            }
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                if (!pCurrentMap->isImuInitialized())
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap();
                    return;
                }

            CreateMapInAtlas();

            return;
        }

        if(!mCurrentFrame.mpReferenceKF){
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }

        mLastFrame = Frame(mCurrentFrame);
    } // B.B end of else for if(mState == NOT_INITIALIZED) statement.

    if(mState == OK || mState == RECENTLY_LOST) {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.

        // B.B if curent frame contains valid data and the tracking is not lost
        if(mCurrentFrame.isSet()) {

            // B.B calculates the relative pose between the current frame and its reference keyframe
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();

            // B.B appends relative pose to the mlRelativeFramePoses vector to keep track of the camera's trajectory over time.
            mlRelativeFramePoses.push_back(Tcr_);

            // B.B appends the reference keyframe to the mlpReferences vector, 
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);

            // B.b appends information about the system's state to the mlbLost vector. 
            // B.B indicating whether the system was in the "LOST" state at the time of the frame's capture.
            mlbLost.push_back(mState == LOST);
        } else {
            // This can happen if tracking is lost
            // B.B duplicates the last stored relative frame pose, reference keyframe, frame timestamp, and system state. 
            // B.B This ensures that there is continuity in the data even when tracking is temporarily lost.
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }

    }

    #ifdef REGISTER_LOOP
        if (Stop()) {

            // Safe area to stop
            while(isStopped())
            {
                usleep(3000);
            }
        }
    #endif

    printf("\n INIT TRACKING PIPELINE -------- (B.B end) ------------ \n\n\n");
    
    // B.B to log required information in a log file
    if (BBLogFile.is_open()) {
        BBLogFile << endl;
        BBLogFile.close();
    }
}


/**
 * B.B Cette methode est appellee, car la valeur de mState=NOT_INITIALIZED et mSensor=RGB-D, dans mon cas.
*/
void Tracking::StereoInitialization() {

    // B.B measure the initialization execution time 
    TIC

    // if(mCurrentFrame.N > 500) { // B.B commented this line
    if(true) {
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated) {
                cout << "not IMU meas" << endl;
                return;
            }

            if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA).norm() < 0.5) {
                cout << "not enough acceleration" << endl;
                return;
            }

            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        } else {
            // B.B a rigid transformation in three-dimensional space using the Special Euclidean group SE(3)
            mCurrentFrame.SetPose(Sophus::SE3f()); // B.B ORB-SLAM3 mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
        }

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        if(!mpCamera2) {

            int num_of_positve_depth = 0; //B.B
            
            for(int i = 0; i < mCurrentFrame.N; i++) {

                // B.B distance from the camera
                float z = mCurrentFrame.mvDepth[i];

                // B.B a valid depth measurement exists for this feature
                if(z > 0) {

                    num_of_positve_depth++; //B.B

                    // B.B used to store the resulting 3D coordinates of the feature
                    Eigen::Vector3f x3D;

                    // B.B unproject the 2D feature at index i into 3D space
                    mCurrentFrame.UnprojectStereo(i, x3D);

                    // B.B a new MapPoint object is created using the 3D coordinates x3D. 
                    // B.B This map point is associated with a keyframe pKFini and the current map in the system.
                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                    /**
                     * B.B
                     * The observation information is added to the new map point. 
                     * It associates the map point with the keyframe pKFini at feature index i, 
                     * indicating that this map point is visible in this keyframe.
                    */
                    pNewMP->AddObservation(pKFini, i);

                    // B.B adds the map point to the keyframe pKFini, indicating that this keyframe observes this map point at feature index i.
                    pKFini->AddMapPoint(pNewMP, i);

                    // B.B computes distinctive descriptors for the newly created map point. Descriptors are essential for feature matching in later frames.
                    pNewMP->ComputeDistinctiveDescriptors(); // B.B ACHTUNG ACHTUNG. Do I need override this method to compute LightGlue Descriptor? No, this method does not do anything with ORB parameters.

                    /**
                     * B.B
                     * updates the normal vector and depth information for the map point. 
                     * Normal vectors are often used for culling or filtering map points.
                    */
                   /**
                    * B.B
                    * Normal refers to the surface normal of a 3D point in the reconstructed map, 
                    * and Depth refers to the valid min and max Distances for the MP to the camera center.
                    * These values are computed to initialized pMP->mnTrackScaleLevel, which is used in LoopClosing module.
                    * So, for now, (Di Jan 2, 2024), I can ignore it.
                    * 
                    * The surface normal of a 3D point is a vector that is perpendicular to the surface at that point. 
                    * It provides information about the orientation of the surface.
                   */
                    pNewMP->UpdateNormalAndDepth(); // B.B culprit (Di Jan 2, 2024- This method was cleared by the explanation above)

                    // B.B newly created map point is added to the map maintained by the SLAM system 
                    mpAtlas->AddMapPoint(pNewMP);

                    // B.B the map point is associated with the feature in the current frame, indicating that this map point corresponds to this feature.
                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                } // B.B if(z > 0)
            } // B.B loop on mCurrentFrame.N

            // cout << endl << " B.B num_of_positve_depth: " << num_of_positve_depth;
            // cin.get();

        } else { // B.B relevant to if(!mpCamera2) statement
            
            for(int i = 0; i < mCurrentFrame.Nleft; i++){
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if(rightIndex != -1){
                    Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini,i);
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP,i);
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP;
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
        cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId();

        // B.B inserts the newly created keyframe into the local mapper module. 
        // B.B The local mapper is responsible for managing the map and keyframes in the vicinity of the camera.
        
        mpLocalMapper->InsertKeyFrame(pKFini); // B.B !!! ACHTUNG ACHTUNG !!! clear

        // B.B to keep a reference to the previous frame for tracking and mapping purposes.
        mLastFrame = Frame(mCurrentFrame);  // B.B !!! ACHTUNG ACHTUNG !!! goes to the DBoW3 DescManip::distance
        mnLastKeyFrameId = mCurrentFrame.mnId;

        // B.B sets the last keyframe to the newly inserted keyframe 
        mpLastKeyFrame = pKFini;

        //mnLastRelocFrameId = mCurrentFrame.mnId;

        /**
         * B.B
         * The newly inserted keyframe is added to the list of local keyframes. 
         * Local keyframes typically represent keyframes that are within a certain distance or time frame 
         * from the current frame and are relevant for tracking.
        */

        mvpLocalKeyFrames.push_back(pKFini);

        // B.B 9. retrieves all map points from the map are used for tracking and mapping in the local area.
        mvpLocalMapPoints = mpAtlas->GetAllMapPoints();

        // B.B The reference keyframe is used as a reference frame for tracking and mapping.
        mpReferenceKF = pKFini;

        // B.B The reference keyframe for the current frame
        mCurrentFrame.mpReferenceKF = pKFini;

        // B.B 12. sets the reference map points in the map stored in mpAtlas. 
        // B.B These map points are used as a reference for tracking and mapping in the local area.
        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        /**
         * B.B
         * 13. The newly inserted keyframe is added to the list of keyframe origins in the current map. 
         * This information can be useful for visualization and map management
        */
        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        // B.B 14. used for visualizing the camera's pose in a user interface.
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        // B.B system has been successfully initialized and is ready to continue tracking and mapping
        mState = OK;

        //  B.B measure the initialization execution time 
        TOC

        cout << endl << "Initialization is finished. Press Enter...";
        // cin.get();

    } else { // B.B 
        cout << endl << "B.B In Tracking::StereoInitialization(), Malheureusement, le nombre de Features extraites est mois de 500." << endl;
    }
}


void Tracking::MonocularInitialization() {

    // B.B checks if the system is not ready to initialize.
    if(!mbReadyToInitializate) {
        // Set Reference Frame
        // B.B if the frame contains enough information for initialization
        if(mCurrentFrame.mvKeys.size() > 100) {

            // B.B indicating the starting point for tracking.
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);

            // B.B copies the undistorted keypoints from the current frame to mvbPrevMatched, preparing for matching keypoints in subsequent frames.
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++){
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;
            }

            // B.B initializes mvIniMatches, which stores keypoint matches between frames, with -1 indicating no matches.
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            if (mSensor == System::IMU_MONOCULAR) {
                if(mpImuPreintegratedFromLastKF) {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            }

            mbReadyToInitializate = true;

            return;
        }
    } else {
        if (((int)mCurrentFrame.mvKeys.size() <= 100) || ((mSensor == System::IMU_MONOCULAR) && (mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp > 1.0))) {
            mbReadyToInitializate = false;
            return;
        }
        
        // 
        // B.B Step 1. find matches between two frames
        // 

        /**
         * B.B
         * to find correspondences between keypoints in the initial and current frames, 
         * which is crucial for establishing a baseline for triangulation and map initialization.
        */
        // Find correspondences
        int nmatches = 0;
        #ifdef USE_SELM_EXTRACTOR
            SELMSLAM::BBLGMatcher bbmatcher;
            nmatches = bbmatcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches);
        #else
            ORBmatcher matcher(0.9,true);
            nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);
        #endif

        // B.B to log the number of matches for Mono initialization
        std::string bbLogFilePath = std::string(BBLOGFILE_PATH) + "BB_num_matche_mono_initialization.log";
        std::ofstream BBLogFile(bbLogFilePath, std::ios::app);
        if (BBLogFile.is_open()) {
            BBLogFile << endl << "Init FrameId: " << mInitialFrame.mnId << "\tCur FrameId: " << mCurrentFrame.mnId << "\t# Matches: " << nmatches;
            BBLogFile.close();
        }

        BBLogger::getInstance().log("\n# of matches in Mono Initialization step. Init FrameId: " + std::to_string(mInitialFrame.mnId) + "\tCur FrameId: " + std::to_string(mCurrentFrame.mnId) + "\t# Matches: " + std::to_string(nmatches));

        cout << endl << "B.B in Tracking::MonocularInitialization, nmatches: " << nmatches;
        // Check if there are enough correspondences
        if(nmatches < 100) {
            mbReadyToInitializate = false;
            return;
        }

        // 
        // B.B Step 2. pose estimation
        // 

        Sophus::SE3f Tcw;
        // B.B to track which correspondences have been successfully triangulated
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        /**
         * B.B
         * attempts to reconstruct the scene from two views (initial and current frames) using matched keypoints. 
         * If successful, it updates the camera pose and the initial 3D points.
         * mvIniP3D: 3D points reconstructed from the two views (B.B in camera's coordinate frame)
         * vbTriangulated: indicates whether each keypoint match was successfully triangulated into a 3D point.
        */

        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn, mCurrentFrame.mvKeysUn, mvIniMatches, Tcw, mvIniP3D, vbTriangulated)) {

            cout << endl << "B.B In Tracking::MonocularInitialization, ReconstructWithTwoViews was SUCCESSFULL ...";

            int bb_toLog = nmatches;

            // B.B clean up matches that were not successfully triangulated 
            for(size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
                if(mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }
            BBLogger::getInstance().log("\nTracking::MonocularInitialization, ReconstructWithTwoViews operation has been don SUCCESSFULLY ...");
            BBLogger::getInstance().log("# of triangulated matches. Init FrameId: " + std::to_string(mInitialFrame.mnId) + "\tCur FrameId: " + std::to_string(mCurrentFrame.mnId) + "\t# Matches: " + std::to_string(nmatches) + "/" + std::to_string(bb_toLog) + "\n");

            // Set Frame Poses
            mInitialFrame.SetPose(Sophus::SE3f());
            mCurrentFrame.SetPose(Tcw);

            // 
            // B.B Step 3. Map initialization
            // 

            // B.B creates the initial map using the monocular camera data, based on the initialized frame poses and triangulated points.
            CreateInitialMapMonocular();
            cout << endl << "B.B ReconstructWithTwoViews operation was SUCCESSFULL.";
        } else {
            cout << endl << "B.B ReconstructWithTwoViews operation was unsuccessful.";
        }
    }
}



void Tracking::CreateInitialMapMonocular() {

    cout << endl << "B.B ready for Tracking::CreateInitialMapMonocular...";
    // Create KeyFrames
    // B.B create two keyframes: one for the initial frame and one for the current frame, associating them with the current map and the keyframe database.
    KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

    if(mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);


    // B.B compute the Bag of Words representation for each keyframe, which is used for feature matching and loop closure detection.
    #ifndef USE_SELM_EXTRACTOR
        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();
    #endif

    // Insert KFs in the map
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    /**
     * B.B
     * for each match that is valid (match index is not -1), 
     * it creates a new MapPoint in the world coordinate system based on the 3D point (mvIniP3D) associated with the match. 
     * This MapPoint is then associated with both the initial and current keyframes and added to the map.
    */
   int bb_toLog = 0;
    for(size_t i = 0; i < mvIniMatches.size(); i++) {

        if(mvIniMatches[i] < 0)
            continue;

        //Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpAtlas->GetCurrentMap());

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpAtlas->AddMapPoint(pMP);

        bb_toLog++;
    }

    BBLogger::getInstance().log("\n In Tracking::CreateInitialMapMonocular. # MPs associated with the initial and current keyframes and added to the map: " + std::to_string(bb_toLog));

    // B.B Update the connectivity graph for each KF, which keeps track of which KFs share obserations of the same MP
    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    // B.B Performs a global bundle adjustment to optimize the map's structure and the camera poses using all current map points and keyframes.
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(), 20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth;
    if(mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f / medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f / medianDepth;

    /**
     * B.B
     * Checks if the median depth is negative or if the current keyframe has tracked fewer than 50 map points. 
     * If either condition is true, it indicates a bad initialization, so the system resets the active map.
    */
    if(medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50) { // TODO Check, originally 100 tracks
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
        mpSystem->ResetActiveMap();
        return;
    }

    /**
     * B.B
     * The initial map is scaled based on the median depth of the initial keyframe to bring the map to a human-understandable scale. 
     * This involves scaling the translation of the current keyframe's pose and all map points.
    */
    // Scale initial baseline
    Sophus::SE3f Tc2w = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
        if(vpAllMapPoints[iMP]) {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    if (mSensor == System::IMU_MONOCULAR) {
        pKFcur->mPrevKF = pKFini;
        pKFini->mNextKF = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
    }


    // B.B The initial and current keyframes are inserted into the local mapper for further processing.
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs = pKFcur->mTimeStamp;

    // B.B Update Current Frame and Last KeyFrame Information
    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;
    //mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
    mbVelocity = false;
    Eigen::Vector3f phi = deltaT.so3().log();

    double aux = (mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp) / (mCurrentFrame.mTimeStamp - mInitialFrame.mTimeStamp);
    phi *= aux;

    mLastFrame = Frame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mState = OK;

    initID = pKFcur->mnId;
}


void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mbSetInit=false;

    mnInitialFrameId = mCurrentFrame.mnId+1;
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mbVelocity = false;
    //mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        mbReadyToInitializate = false;
    }

    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }

    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

    if(mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame*>(NULL);

    mLastFrame = Frame();
    mCurrentFrame = Frame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

// B.B Local Mapping might have changed some MapPoints tracked in last frame
void Tracking::CheckReplacedInLastFrame() {

    for(int i = 0; i < mLastFrame.N; i++) {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP) {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep) {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/**
 * B.B 
 * The method is responsible for tracking the camera's pose with respect to a reference keyframe 
 * and ensuring that only valid map points are considered for tracking. 
 * It also performs pose optimization to improve tracking accuracy.
*/
bool Tracking::TrackReferenceKeyFrame() {
    
    cout << endl << "B.B Je suis dans le Tracking::TrackReferenceKeyFrame. Dahanam ra service nemudi... " << endl;

    // Compute Bag of Words vector
    #ifndef USE_SELM_EXTRACTOR
        mCurrentFrame.ComputeBoW(); // Bravely commented by Banafshe Bamdad 2023/10/28
    #endif

    // ORBmatcher matcher(0.7, true); // B.B 2023/10/28

    // B.B to store matched MapPoint objects.
    vector<MapPoint*> vpMapPointMatches;

    TIC
    #ifdef USE_SELM_EXTRACTOR

        SELMSLAM::BBLGMatcher bbmatcher;
        int nmatches = bbmatcher.SearchByLG(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    #else
        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        // B.B an ORB matcher is initialized with a ratio test threshold of 0.7 and two-way matching enabled
        /**
         * B.B
         * When matching features between two images
         * For each keypoint in the first image, the nearest neighbor (i.e., the closest descriptor) in the second image is identified.
         * The ratio of the distance to the closest neighbor and the distance to the second closest neighbor is computed. 
         * If this ratio is below a certain threshold (e.g., 0.7), the match is retained; otherwise, it is discarded.
         * 
         * two-way matching involves finding matches in both directions: from A to B and from B to A.
        */
        ORBmatcher matcher(0.7, true);
        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches); // B.B 2023/10/28
    #endif

    cout << endl << "B.B 1..................." << endl;
    BBLogger::getInstance().log("\n In Tracking::TrackReferenceKeyFrame. RefKFId: " + std::to_string(mpReferenceKF->mnId) + "\tCur FrameId: " + std::to_string(mCurrentFrame.mnId) + "\t# of Matches: " + std::to_string(nmatches) + "\t# of Features in Cur frame: " + std::to_string(vpMapPointMatches.size()));
    cout << endl << "B.B mpReferenceKF ID: " << mpReferenceKF->mnId << ", mCurrentFrame ID: " << mCurrentFrame.mnId << ", # of matches: " << nmatches << ". In Tracking::TrackReferenceKeyFrame.";
    
    TOC

    if(nmatches < 15) {
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    // B.B initializes the camera's pose for tracking
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    //mCurrentFrame.PrintPointDistribution();

    Optimizer::PoseOptimization(&mCurrentFrame);

    BBLogger::getInstance().log("\nPose Optimization has been done on Curent Fram.\n");

    // Discard outliers
    // B.B discarding outlier feature matches between the current frame and a reference keyframe.
    // B.B to count the number of valid feature matches between the current frame and the map points.
    int nmatchesMap = 0;

    // B.B iterates over the features in the current frame. 
    for(int i = 0; i < mCurrentFrame.N; i++) {
        //if(i >= mCurrentFrame.Nleft) break;

        // B.B checks if there is a valid map point associated with the feature in the current frame. 
        if(mCurrentFrame.mvpMapPoints[i]) {
            
            // B.B checks if the feature is marked as an outlier.
            if(mCurrentFrame.mvbOutlier[i]) {

                // B.B gets the associated map point
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                // B.B Set the map point associated with the feature to NULL.
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

                // B.B Mark the feature as not an outlier
                mCurrentFrame.mvbOutlier[i] = false;

                // B.B Depending on whether the feature is in the left or right part of the stereo image, 
                // B.B it marks the map point's visibility status as mbTrackInView or mbTrackInViewR as false.
                if(i < mCurrentFrame.Nleft) {
                    pMP->mbTrackInView = false;
                } else {
                    pMP->mbTrackInViewR = false;
                }

                pMP->mbTrackInView = false;

                // B.B Updates the mnLastFrameSeen attribute of the map point with the ID of the current frame.
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;

            // B.B checks if the map point associated with the feature has observations from other frames. 
            // B.B If it does, it means the map point is still valid and contributes to the map.
            } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0){

                // B.B If the feature is not an outlier and has observations, the nmatchesMap counter is incremented, 
                // B.B indicating that a valid feature match with the map has been found.
                nmatchesMap++;
            }
        }
    }

    BBLogger::getInstance().log("\n After discarding outlier feature matches between the current frame and a reference keyframe. # of matches: " + std::to_string(nmatches) + "\n# of valid feature matches between the current frame and the map points: " + std::to_string(nmatchesMap));
    BBLogger::getInstance().log("INFO: The system checks if there is a valid map point associated with each feature in the current frame, and if the feature is marked as an outlier: \n\t1) Set the map point associated with that feature to NULL.\n\t2) Decrease the number of matches between Reference KeyFrame and Current Frame.");
    BBLogger::getInstance().log("\n\tIf the feature is not an outlier, checks if the map point associated with the feature has observations from other frames. If it does, it means the map point is still valid and contributes to the map.\n\tIf the feature is not an outlier and has observations, the number of valid feature matches between the current frame and the map points is incremented.");

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap >= 10;
}

/**
 * B.B
 * responsible for updating the last frame's pose according to a reference keyframe and 
 * creating "visual odometry" MapPoints in certain conditions, 
 * helping to maintain and update the map during the tracking process.
*/
void Tracking::UpdateLastFrame() {

    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;

    /**
     * B.B 
     * mlRelativeFramePoses is a list used to recover the full camera trajectory at the end of the execution.
     * The reference keyFrame for each frame and its relative transformation are stored.
    */
    // B.B fetches the relative pose between the last frame and the reference keyframe
    Sophus::SE3f Tlr = mlRelativeFramePoses.back();

    // B.B updates the pose of the last frame by applying the relative pose to the reference keyframe's pose.
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if(mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    // B.B Initializes a vector to store pairs of depth and feature indices.
    vector<pair<float, int> > vDepthIdx;

    // B.b Determines the number of features
    const int Nfeat = mLastFrame.Nleft == -1 ? mLastFrame.N : mLastFrame.Nleft;
    vDepthIdx.reserve(Nfeat);
    for(int i = 0; i < Nfeat; i++) {
        float z = mLastFrame.mvDepth[i];
        if(z > 0) {
            vDepthIdx.push_back(make_pair(z, i));
        }
    }

    // B.B there are no valid depths to work with
    if(vDepthIdx.empty())
        return;

    // B.B sorted in ascending order based on depth values.
    sort(vDepthIdx.begin(), vDepthIdx.end());

    // We insert all close points (depth < mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    // B.B keeps track of the number of points processed.
    int nPoints = 0;

    // B.B checks whether to create a new MapPoint or update an existing one 
    for(size_t j = 0; j < vDepthIdx.size(); j++) {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations() < 1)
            bCreateNew = true;

        if(bCreateNew) {
            Eigen::Vector3f x3D;

            // B.B computes the 3D position of the feature using stereo or stereo fish-eye unprojection
            if(mLastFrame.Nleft == -1){
                mLastFrame.UnprojectStereo(i, x3D);
            } else {
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }

            // B.B A new MapPoint is created and associated with the last frame, and it's added to the list of temporal points
            MapPoint* pNewMP = new MapPoint(x3D, mpAtlas->GetCurrentMap(), &mLastFrame, i);
            mLastFrame.mvpMapPoints[i] = pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        } else {
            nPoints++;
        }

        // B.B The loop breaks if the depth exceeds the mThDepth threshold and a sufficient number of points have been processed.
        if(vDepthIdx[j].first > mThDepth && nPoints > 100) {
            break;
        }
    }
}

bool Tracking::TrackWithMotionModel() {

    std::string BBLogFile_Path = std::string(BBLOGFILE_PATH) + "BB_monitoring.log";
    std::ofstream BBLogFile(BBLogFile_Path, std::ios::app);

    #ifndef USE_SELM_EXTRACTOR
        ORBmatcher matcher(0.9,true);
    #endif

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId > mnLastRelocFrameId + mnFramesToResetIMU)) {
        // Predict state with IMU if it is initialized and it doesnt need reset
        PredictStateIMU();
        return true;
    } else {
        mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose()); // B.B set the initial pose for current frame
    }

    // B.B fill the range [first, last) with copies of value
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;

    if(mSensor == System::STEREO)
        th = 7;
    else
        th = 15;

    // B.B to search for feature matches by projection from the last frame to the current frame. (Project points seen in previous frame)
    #ifdef USE_SELM_EXTRACTOR
        SELMSLAM::BBLGMatcher bbmatcher;
        // int nmatches = bbmatcher.TrackLastFrameMapPoints(mCurrentFrame, mLastFrame);
        int nmatches = bbmatcher.MatchLastAndCurrentFrame(mLastFrame, mCurrentFrame);
    #else
        int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);

        // If few matches, uses a wider window search
        if(nmatches < 20) {
            Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
            fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

            // B.B Another search for matches is performed with a wider search window (twice the size of th)
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
            Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

        }
    #endif

    BBLogger::getInstance().log("\nIn Tracking::TrackWithlastFrame. Last frameID: " + std::to_string(mLastFrame.mnId) + "\tCurrent frameID: " + std::to_string(mCurrentFrame.mnId) + " # matches between last and current frame: " + std::to_string(nmatches));

    if(nmatches < 20) {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
            return false;
    }

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    BBLogFile << endl << "\t # matches between two frames: " << nmatches;

    // Discard outliers
    // B.B to count the number of matches with valid map points
    int nmatchesMap = 0;
    for(int i = 0; i < mCurrentFrame.N; i++) {
        if(mCurrentFrame.mvpMapPoints[i]) {

            // B.B an incorrect or unreliable match
            if(mCurrentFrame.mvbOutlier[i]) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;

                // B.B If the feature is from the left camera view, it marks the associated map point as not being tracked in the left view.
                if(i < mCurrentFrame.Nleft) {
                    pMP->mbTrackInView = false;
                } else {
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;

            } else if(mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        } 
    }

    BBLogger::getInstance().log("\n After discarding outlier feature matches between the last and current frames. Last frameID: " + std::to_string(mLastFrame.mnId) + "\tCurrent frameID: " + std::to_string(mCurrentFrame.mnId) + "\t# of matches: " + std::to_string(nmatches) + "\n# of valid feature matches between the current frame and the map points: " + std::to_string(nmatchesMap));

    BBLogFile << endl << "\t # matches between two frames after discarding outliers: " << nmatches;
    BBLogFile << endl << "\t # of matches between Current Frame and map: " << nmatchesMap;

    if(mbOnlyTracking) {

        BBLogFile << endl << "\tOnlyTracking. (mbVO = nmatchesMap < 10): " << boolalpha << mbVO << ", (nmatches > 20): " << (nmatches > 20);

        mbVO = nmatchesMap < 10;
        return nmatches > 20;
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else {
        BBLogFile << endl << "\t(nmatchesMap >= 10): " << boolalpha << (nmatchesMap >= 10);
        return nmatchesMap >= 10;
    }
}

/**
 * B.B
 * to track the camera's position and orientation using the estimated camera pose 
 * and to update the map of the environment based on the tracked features. 
 * This function returns true or false based on the number of inliers and the type of sensor being used. 
 * The decision to return true or false is based on a set of conditions that determine whether the tracking is considered successful. 
 * The conditions vary based on the sensor type. 
 * If the number of inliers meets certain criteria, it returns true, indicating a successful tracking session; otherwise, it returns false.
*/
bool Tracking::TrackLocalMap() {

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    // B.B the number of frames that have been tracked so far.
    mTrackedFr++;

    // B.B updates the local map, likely by selecting a subset of map points that are relevant to the current frame. (check it Banafshe)
    UpdateLocalMap();

    // B.B to log the number of local Map Points
    std::string bbLogFilePath = std::string(BBLOGFILE_PATH) + "BB_num_local_MPs_KFs.log";
    std::ofstream BBLogFile(bbLogFilePath, std::ios::app);
    if (BBLogFile.is_open()) {
        BBLogFile << endl << "FrameId: " << mCurrentFrame.mnId << "\t# MPs: " << mvpLocalMapPoints.size() << "\t#KFs: " << mvpLocalKeyFrames.size();
        BBLogFile.close();
    }

    // B.B searches for local map points that correspond to features in the current frame, trying to establish feature matches.
    SearchLocalPoints();

    // TOO check outliers before PO
    // B.B to count the number of features with associated map points and the number of features marked as outliers, 

    // ‌What are these variables user for? They are initialized again by zero, without using the calculated values in the following loop.
    // B.B I belive this code snippet is useless,
    int aux1 = 0, aux2 = 0;
    for(int i = 0; i < mCurrentFrame.N; i++) {

        // B.B If a feature has an associated map point
        if( mCurrentFrame.mvpMapPoints[i]) {
            aux1++;

            // B.B If the feature is marked as an outlier 
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }
    }

    // B.B  the number of inliers resulting from the pose optimization.
    int inliers;
    // B.B performs pose optimization depending on the state of the SLAM system
    // B.B If IMU is not initialized, it optimizes the camera's pose using the current frame's data.
    // B.B Otherwise, it checks if the current frame's ID is less than or equal to a certain threshold 
    if (!mpAtlas->isImuInitialized()) {
        Optimizer::PoseOptimization(&mCurrentFrame);
    } else {

        if(mCurrentFrame.mnId <= mnLastRelocFrameId + mnFramesToResetIMU) {
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame);
        } else {
            // B.B chooses between two different optimization methods, based on whether the map has been updated or not.
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
            if(!mbMapUpdated) { //  && (mnMatchesInliers>30))
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            } else {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }

    aux1 = 0, aux2 = 0;
    for(int i = 0; i < mCurrentFrame.N; i++) {
        if( mCurrentFrame.mvpMapPoints[i]) {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }
    }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i = 0; i < mCurrentFrame.N; i++) {
        if(mCurrentFrame.mvpMapPoints[i]) {
            if(!mCurrentFrame.mvbOutlier[i]) {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking) {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        mnMatchesInliers++;
                } else
                    mnMatchesInliers++;
            } else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    mpLocalMapper->mnMatchesInliers = mnMatchesInliers;
    if(mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50) { // B.B changed 50 to 10
        return false;
    }

    if((mnMatchesInliers > 10) && (mState == RECENTLY_LOST)) {
        return true;
    }


    if (mSensor == System::IMU_MONOCULAR) {
        if((mnMatchesInliers < 15 && mpAtlas->isImuInitialized()) || (mnMatchesInliers < 50 && !mpAtlas->isImuInitialized())) {
            return false;
        } else
            return true;
    } else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
        if(mnMatchesInliers < 15) {
            return false;
        } else
            return true;
    } else {
        if(mnMatchesInliers < 10) { // B.B !!! ACHTUNG ACHTUNG !!! Banafshe changed the value from 30 to 20 in order to bypass the relocalization process
            return false;
        } else {
            return true;
        }
    }
}

bool Tracking::NeedNewKeyFrame() {
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized()) {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else
            return false;
    }

    if(mbOnlyTracking){
        return false;
    }

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // cout << endl << "B.B Local Mapping is freezed by a Loop Closure do not insert keyframes: " << boolalpha << (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()); 

    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        /*if(mSensor == System::MONOCULAR)
        {
            std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
        }*/
        return false;
    }

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames) {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs <= 2) {
        nMinObs = 2;
    }    
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;

    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR) {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
        //Verbose::PrintMess("[NEEDNEWKF]-> closed points: " + to_string(nTrackedClose) + "; non tracked closed points: " + to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);// Verbose::VERBOSITY_DEBUG);
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs < 2){
        thRefRatio = 0.4f;
    }

    /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
    const int thStereoClosedPoints = 15;
    if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    if(mSensor==System::MONOCULAR) {
        thRefRatio = 0.9f;
    }

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR) {
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle); //mpLocalMapper->KeyframesInQueue() < 2);
    //Condition 1c: tracking is weak
    const bool c1c = mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR && mSensor!=System::IMU_STEREO && mSensor!=System::IMU_RGBD && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    //std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
    // Temporal condition for Inertial cases
    bool c3 = false;
    if(mpLastKeyFrame)
    {
        if (mSensor==System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
        else if (mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true;
    else
        c4=false;

    if(((c1a||c1b||c1c) && c2)||c3 ||c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle || mpLocalMapper->IsInitializing())
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
            {
                //std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                return false;
            }
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame() {
    if(mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
        return;

    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

    std::string bbLogFilePath = std::string(BBLOGFILE_PATH) + "BB_if_KF_is_needed.log";
    std::ofstream BBLogFile(bbLogFilePath, std::ios::app);
    if (BBLogFile.is_open()) {
        BBLogFile << "\tReady to add KF mnId: " << pKF->mnId << "\tmnFrameId: " << pKF->mnFrameId;
        BBLogFile.close();
    }

    if(mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame.mImuBias);
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mpLastKeyFrame) {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    } else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
    }

    if(mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) { // TODO check if incluide imu_stereo
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            maxPoint = 100;

        vector<pair<float,int> > vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i = 0; i < N; i++) {
            float z = mCurrentFrame.mvDepth[i];
            if(z > 0) {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if(!vDepthIdx.empty()) {
            sort(vDepthIdx.begin(), vDepthIdx.end());

            int nPoints = 0;
            for(size_t j = 0; j < vDepthIdx.size(); j++) {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations() < 1) {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew) {
                    Eigen::Vector3f x3D;

                    if(mCurrentFrame.Nleft == -1){
                        mCurrentFrame.UnprojectStereo(i, x3D);
                    } else {
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]] = pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    nPoints++;
                } else {
                    nPoints++;
                }

                if(vDepthIdx[j].first > mThDepth && nPoints > maxPoint) {
                    break;
                }
            }
            //Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;

}

/**
 * B.B
 * for updating the visibility and tracking status of map points in the current frame, 
 *  . 
 * The (ORB) matching threshold is dynamically adjusted based on the system's state and sensor type.
*/
void Tracking::SearchLocalPoints() {
    int bb_counter = 0;
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++) {
        MapPoint* pMP = *vit;
        if(pMP) {
            if(pMP->isBad()) {
                // B.B removing it from the list.
                *vit = static_cast<MapPoint*>(NULL);
            } else {
                // B.B represents how many times the map point has been seen
                pMP->IncreaseVisible();

                // B.B the frame in which the map point was last observed.
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;

                // B.B indicating that the map point is not currently being tracked in the current frame
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
            }
        } else {
            bb_counter++;
        }
    }

    cout << endl << "B.B in Tracking::SearchLocalPoints. # null pMP in currentFrame: " << bb_counter << ", mCurrentFrameId: " << mCurrentFrame.mnId << ", #MPs: " << mCurrentFrame.mvpMapPoints.size();
    // B.B will be used to count the number of map points that are visible in the current frame and are candidates for matching.
    int nToMatch = 0;

    cout << endl << "B.B In Tracking::SearchLocalPoints. # of local MPs: " << mvpLocalMapPoints.size();

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++) {
        MapPoint* pMP = *vit;

        // B.B checks whether the map point was seen in the current frame 
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        // B.B checks whether the map point is within the camera's frustum with a tolerance of 0.5 units.
        if(mCurrentFrame.isInFrustum(pMP, 0.5)) {
            pMP->IncreaseVisible(); 
            nToMatch++;
        }
        // B.B If a map point is being tracked, it updates the projection coordinates of the map point in the current frame 
        // B.‌B based on its position (mTrackProjX and mTrackProjY).
        if(pMP->mbTrackInView) {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }

    cout << endl << "B.B In Tracking::SearchLocalPoints. # visible MPs in current frame for matching. nToMatch: " << nToMatch;
    if(nToMatch > 0) {
        // ORBmatcher matcher(0.8); // B.B 
        int th = 1;
        if(mSensor == System::RGBD || mSensor == System::IMU_RGBD)
            th = 3;
        if(mpAtlas->isImuInitialized()) {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th=2;
            else
                th=6;
        } else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)) {
            th = 10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            th = 5;

        if(mState == LOST || mState == RECENTLY_LOST) // Lost for less than 1 second
            th = 15; // 15

        #ifdef USE_SELM_EXTRACTOR
            SELMSLAM::BBLGMatcher bbmatcher;
            int nmatches = bbmatcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
            cout << endl << "B.B In Tracking::SearchLocalPoints. nmatches: " << nmatches;
        #else
            ORBmatcher matcher(0.8); // B.B 
            int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
        #endif
    }
}

void Tracking::UpdateLocalMap() {
    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints() {

    cout << endl << "B.B --- START --- of Tracking::UpdateLocalPoints. # of local MPs: " << mvpLocalMapPoints.size();

    mvpLocalMapPoints.clear();

    int count_pts = 0;

    /**
     * B.B
     * pointers to the local keyframes considered relevant for the current frame. 
     * The reverse iteration is used probably because the most recent keyframes are more likely to contain relevant map points,
     * and checking them first can be more efficient.
    */
    for(vector<KeyFrame*>::const_reverse_iterator itKF = mvpLocalKeyFrames.rbegin(), itEndKF = mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF) {
        KeyFrame* pKF = *itKF;

        // B.B map points observed in the keyframe.
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        int count_not_pMP = 0; // B.B to count the number of map points that are not valid
        int count_cur_associated = 0; // B.B to count the number of map points that are already associated with the current frame

        for(vector<MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++) {

            MapPoint* pMP = *itMP;
            if(!pMP) {
                count_not_pMP++; // B.B increments the count of invalid map points
                continue;
            }
            
            // B.B if the current map point pMP has already been associated with the current frame
            if(pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId){
                count_cur_associated++; // B.B increments the count of map points that are already associated with the current frame
                continue;
            }
            // B.B a valid point for tracking
            if(!pMP->isBad()) {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
        cout << endl << "\tKFId: " << pKF->mnId << "\tvpMPs: " << vpMPs.size() << "\t# of valid MPs: " << count_pts << "\t# of invalid MPs: " << count_not_pMP;
    }
    cout << endl << "B.B END of Tracking::UpdateLocalPoints. # of local MPs: " << mvpLocalMapPoints.size();
}


// B.B updates the local keyframes that are relevant for the current frame's tracking process
void Tracking::UpdateLocalKeyFrames() {
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*, int> keyframeCounter; // B.B to count the number of map points observed in each keyframe.

    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId < mnLastRelocFrameId + 2)) {
        for(int i = 0; i < mCurrentFrame.N; i++) {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP) {
                if(!pMP->isBad()) {
                    const map<KeyFrame*, tuple<int, int>> observations = pMP->GetObservations();

                    // B.B iterates over MPs observations (keyframes where the map point was seen) and increments the vote count for those keyframes.
                    for(map<KeyFrame*, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                        keyframeCounter[it->first]++;
                } else {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }
    } else {
        /**
         * B.B 
         * If the IMU is initialized and the current frame is beyond the last relocalization frame by more than 2 frames, 
         * the function processes the last frame's map points
        */
        for(int i = 0; i < mLastFrame.N; i++) {
            // Using lastframe since current frame has not matches yet
            if(mLastFrame.mvpMapPoints[i]) {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP)
                    continue;
                if(!pMP->isBad()) {
                    const map<KeyFrame*, tuple<int, int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                        keyframeCounter[it->first]++;
                } else {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i] = NULL;
                }
            }
        }
    }


    // B.B Initializes variables to find the keyframe with the maximum count of map points
    int max = 0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear(); // B.B clears the list of local keyframes.
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    /**
     * B.B
     * Iterates over keyframeCounter to populate mvpLocalKeyFrames with keyframes that are not marked as bad, 
     * and updates the keyframe with the maximum map point count.
    */
    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++) {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second > max) {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    /**
     * B.B
     * Expands the local keyframes set by including neighboring keyframes of those already included, up to a limit of 80 keyframes
    */
    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++) {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size() > 80) // 80
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);


        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++) {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++) {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent) {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) &&mvpLocalKeyFrames.size()<80) {
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame)
                break;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                tempKeyFrame=tempKeyFrame->mPrevKF;
            }
        }
    }

    if(pKFmax) {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

// B.B This method attempts to relocalize the current frame by finding a keyframe in the database that is similar to the current frame.

/*********************
bool Tracking::Relocalization() {
    cout << endl << "B.B Starting relocalization. Press Enter..."; 
    cout << endl << "B.B The number of keyframe in map: " << mpAtlas->GetCurrentMap()->KeyFramesInMap() << ". Press Enter..."; 

    // 
    // B.B Step 1. Retrieve KeyFrames in current Map
    // 
    std::set<KeyFrame*> bb_mspKeyFrames = mpAtlas->GetCurrentMap()->mspKeyFrames;
    std::vector<KeyFrame*> vpCandidateKFs(bb_mspKeyFrames.begin(), bb_mspKeyFrames.end());
    cout << endl << "B.B The number of candidate keyframes: " << vpCandidateKFs.size(); 

    // B.B sort the vector based on KeyFrame.mnId
    std::sort(vpCandidateKFs.begin(), vpCandidateKFs.end(),
              [](const KeyFrame* kf1, const KeyFrame* kf2) {
                std::cout << "Comparing mnId: " << kf1->mnId << " and " << kf2->mnId << std::endl;
                return kf1->mnId < kf2->mnId;
              });

    std::reverse(vpCandidateKFs.begin(), vpCandidateKFs.end());
    // B.B sorted Keyframes based on mnIds
    for (const auto& keyFramePtr : vpCandidateKFs) {
        std::cout << "sorted mnId: " << keyFramePtr->mnId << std::endl;
    }

    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // 
    // Step 2.
    // 
    // B.B performs an LightGlue/ORB matching and then attempt to find a camera pose using the PnP algorithm.
    // B.B The results of the matching are stored in the vpMapPointMatches vector.
    
    bool bMatch = false;

    for(int i = 0; i < nKFs; i++) {
        bool bMatch = false;
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad()) {
            continue;
        } else {

            // B.B I first perform a LightGlue matching with each KeyFrame in the current map
            // B.B If enough matches are found I setup a PnP solver to estimate the pose of the camera
            
            vector<MapPoint*> vpMapPointMatches;
            SELMSLAM::BBLGMatcher bbmatcher;
            int nmatches = bbmatcher.SearchByLG(pKF, mCurrentFrame, vpMapPointMatches);
            
            if(nmatches < 15) {
                continue;
            } else {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame, vpMapPointMatches);
                pSolver->SetRansacParameters(0.99, 10, 300, 6, 0.5, 5.991);  //This solver needs at least 6 points
                
                // bool bMatch = false;
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                Eigen::Matrix4f eigTcw;
                bool bTcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers, eigTcw);

                // If Ransac reachs max. iterations discard keyframe
                if(bNoMore) {
                    continue;
                }

                // If a Camera Pose is computed, optimize
                if(bTcw) {
                    Sophus::SE3f Tcw(eigTcw);
                    mCurrentFrame.SetPose(Tcw);

                    const int np = vbInliers.size();

                    for(int j = 0; j < np; j++) {
                        if(vbInliers[j]) {
                            mCurrentFrame.mvpMapPoints[j] = vpMapPointMatches[j];
                        } else {
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                        }
                    }

                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    if(nGood < 50) {
                        continue;
                    }

                    for(int io = 0; io < mCurrentFrame.N; io++) {
                        if(mCurrentFrame.mvbOutlier[io]) {
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint*>(NULL);
                        }
                    }

                    // If the pose is supported by enough inliers stop ransacs and continue
                    bMatch = true;
                    break;
                } // B.B end of If a Camera Pose is computed, optimize
            }
        }
    }

    if(!bMatch) {
        return false;
    } else {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        cout << "B.B Press Enter..."; 
        // cin.get();
        return true;
    }

}
*/

// B.B original version, DO NOT delete this method
bool Tracking::Relocalization() {
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    
    cout << endl << "B.B Starting relocalization. Press Enter..."; 
    // cin.get();
    // cout << endl << "B.B The number of keyframe in map: " << mpAtlas->GetCurrentMap()->KeyFramesInMap() << ". Press Enter..."; 
    // cin.get();

    // Compute Bag of Words Vector
    // #ifndef USE_SELM_EXTRACTOR
        mCurrentFrame.ComputeBoW();
    // #endif

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // B.B The folloing line is temporarily commented by Banafshe and replaced with the following code snippet.
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());
    // B.B ??? Does the order of KFs matter???
    // std::set<KeyFrame*> mspKeyFrames = mpAtlas->GetCurrentMap()->mspKeyFrames;
    // std::vector<KeyFrame*> vpCandidateKFs(mspKeyFrames.begin(), mspKeyFrames.end());
    // cout << endl << "B.B The number of candidate keyframes: " << vpCandidateKFs.size() << ". Press Enter..."; 
    // cin.get();
    // Jusqu'ici

    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    
    // #ifdef USE_SELM_EXTRACTOR
    //     SELMSLAM::BBLGMatcher bbmatcher;
    // #else
        ORBmatcher matcher(0.75, true);
    // #endif

    // B.B The PnP algorithm is a method of finding the pose of a camera in 3D space from a set of known 3D-2D point correspondences.
    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);


    // B.B to keep track of which candidate keyframes have been discarded due to insufficient matching.
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    // B.B to keep track of the number of candidate keyframes that are still being considered for relocalization.
    int nCandidates = 0;

    // B.B performs an LightGlue/ORB matching and then attempt to find a camera pose using the PnP algorithm.
    // B.B The results of the matching are stored in the vvpMapPointMatches[i] vector.
    for(int i = 0; i < nKFs; i++) {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad()) {
            vbDiscarded[i] = true;
        } else {

            // #ifdef USE_SELM_EXTRACTOR
            //     int nmatches = bbmatcher.SearchByLG(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            // #else
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            // #endif
            
            if(nmatches < 15) {
                vbDiscarded[i] = true;
                continue;
            } else {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                /**
                 * B.B
                 * Probability of inliers: 0.99
                 * the desired probability that the inliers found by the RANSAC algorithm will actually be inliers. 
                 * A higher probability will result in more robust results, but it will also require more iterations.
                 * 
                 * Initial number of points: 10
                 * the minimum number of points that must be inliers for the RANSAC algorithm to consider a model to be inliers. 
                 * A lower number of points will result in faster results, but it may also result in more outliers.
                 * 
                 * Maximum number of iterations: 300
                 * the maximum number of iterations that the RANSAC algorithm will run. 
                 * If the algorithm does not find a model with enough inliers after this number of iterations, it will fail.
                 * 
                 * Inlier threshold: 6
                 * the threshold that is used to determine if a point is considered to be an inlier. 
                 * A lower threshold will result in more inliers, but it may also result in more false positives.
                 * 
                 * Scaling: 0.5
                 * the scaling factor that is used to scale the input points before the RANSAC algorithm is run. 
                 * This can help to improve the robustness of the algorithm in some cases.
                 * 
                 * Confidence: 5.991
                 *  the confidence level for the RANSAC algorithm. 
                 * A higher confidence level will result in more robust results, but it may also result in more iterations.
                */
                pSolver->SetRansacParameters(0.99, 10, 300, 6, 0.5, 5.991);  //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;

    // #ifdef USE_SELM_EXTRACTOR
    //     SELMSLAM::BBLGMatcher bbmatcher2;
    // #else
        ORBmatcher matcher2(0.9, true);
    // #endif

    while(nCandidates > 0 && !bMatch) {
        for(int i = 0; i < nKFs; i++) {
            if(vbDiscarded[i]){
                continue;
            }

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            Eigen::Matrix4f eigTcw;
            bool bTcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers, eigTcw);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore) {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(bTcw) {
                Sophus::SE3f Tcw(eigTcw);
                mCurrentFrame.SetPose(Tcw);
                // Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j = 0; j < np; j++) {
                    if(vbInliers[j]) {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    } else {
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                    }
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood < 10) {
                    continue;
                }

                for(int io = 0; io < mCurrentFrame.N; io++) {
                    if(mCurrentFrame.mvbOutlier[io]) {
                        mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint*>(NULL);
                    }
                }

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood < 50) {

                    // #ifdef USE_SELM_EXTRACTOR
                    //     int nadditional = 0;
                    //     cout << endl << "B.B In Tracking::Relocalization. Since nGood < 50, SearchByProjection sould be called. Press Enter ..."; 
                    //     // cin.get();
                    // #else
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);
                    // #endif

                    if(nadditional + nGood >= 50) {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood > 30 && nGood < 50) {
                            sFound.clear();
                            for(int ip = 0; ip < mCurrentFrame.N; ip++) {
                                if(mCurrentFrame.mvpMapPoints[ip]) {
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                }
                            }

                            // #ifdef USE_SELM_EXTRACTOR
                            //     cout << endl << "B.B In Tracking::Relocalization. Since nGood > 30 && nGood < 50, SearchByProjection sould be called. Press Enter ..."; 
                            //     // cin.get();
                            // #else
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);
                            // #endif

                            // Final optimization
                            if(nGood + nadditional >= 50) {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io = 0; io < mCurrentFrame.N; io++) {
                                    if(mCurrentFrame.mvbOutlier[io]) {
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                                    }
                                }
                            }
                        }
                    }
                } // B.B end of (nGood < 50) 


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood >= 50) {
                    bMatch = true;
                    break;
                }
            } // B.B end of If a Camera Pose is computed, optimize
        } // end of for loop on nKFs
    } // B.B end of while(nCandidates > 0 && !bMatch) 

    if(!bMatch) {
        return false;
    } else {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        cout << "B.B Press Enter..."; 
        // cin.get();
        return true;
    }

}

void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit=false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    //mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    mbVelocity = false;

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    mK_.setIdentity();
    mK_(0,0) = fx;
    mK_(1,1) = fy;
    mK_(0,2) = cx;
    mK_(1,2) = cy;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(auto lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        while(pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if(pKF->GetMap() == pMap)
        {
            (*lit).translation() *= s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }


    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    //mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap)
{
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if(!strNameFile_kf.empty())
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

float Tracking::GetImageScale()
{
    return mImageScale;
}

#ifdef REGISTER_LOOP
void Tracking::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Tracking STOP" << endl;
        return true;
    }

    return false;
}

bool Tracking::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}
#endif

} //namespace ORB_SLAM
