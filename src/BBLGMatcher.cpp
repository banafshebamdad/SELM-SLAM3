/*
 * Author: Banafshe Bamdad
 * Created on Sat Oct 28 2023 16:53:23 CET
 *
 */

#include "BBLGMatcher.hpp"
#include "BBLightGlue.hpp"

#include <iostream>
#include <string>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

using namespace ORB_SLAM3;

#define WITH_TICTOC
    #include <tictoc.hpp>
#define WITH_TICTOC

#ifndef BBLIGHTGLUE_WEIGHT_PATH
    #define BBLIGHTGLUE_WEIGHT_PATH "/Weights/BBPretrained_Models/superpoint_lightglue.onnx"
#endif

namespace SELMSLAM {
    BBLGMatcher::BBLGMatcher() {
    }
    
    BBLGMatcher::~BBLGMatcher() {
    }

    float BBLGMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {

        float dist = (float)cv::norm(a, b, cv::NORM_L2);
        
        return dist;
    }

    int BBLGMatcher::SearchByLG(KeyFrame* pKF, Frame &F, vector<MapPoint*> &vpMapPointMatches) {
        
        SELMSLAM::BBLightGlue bblg(BBLIGHTGLUE_WEIGHT_PATH, 0.0f); 
        int nmatches = bblg.match(pKF, F, vpMapPointMatches);

        return nmatches;
    }

    int BBLGMatcher::MatchLastAndCurrentFrame(Frame lastFrame, Frame &F) {
        
        SELMSLAM::BBLightGlue bblg(BBLIGHTGLUE_WEIGHT_PATH, 0.0f); 
        int nmatches = bblg.match(lastFrame, F);

        return nmatches;
    }

    /**
     * read Section V. Tracking D. Track Local Map
     * from paper "ORB-SLAM: A Versatile and Accurate Monocular SLAM System"
    */
    int BBLGMatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const bool bFarPoints, const float thFarPoints) {

        std::vector<cv::Mat> vMPDescriptors;
        cv::Mat MPDescriptors; // to send to LightGlue model
        std::vector<cv::KeyPoint> vMPKeypoint; // to send to LightGlue model
        vector<MapPoint*> projectedMapPoints;

        for(size_t iMP = 0; iMP < vpMapPoints.size(); iMP++) {

            MapPoint* pMP = vpMapPoints[iMP];

            
            if(!pMP->mbTrackInView && !pMP->mbTrackInViewR){
            // if(!pMP->mbTrackInView){
                continue;
            }

            if(bFarPoints && pMP->mTrackDepth > thFarPoints)
                continue;

            if(pMP->isBad())
                continue;

            if(pMP->mbTrackInView) {
                const cv::Mat MPdescriptor = pMP->GetDescriptor();
                vMPDescriptors.push_back(MPdescriptor);

                /**
                 * Since we only consider map points that are projected onto the current frame (pMP->mbTrackInView), 
                 * we can be sure that the projected coordinates of MPs are on the current frame. 
                 * Therefore, I can treat these projected coordinates as the coordinates of a virtual frame and send them as input to the LightGlue model.
                 * pMP->mTrackProjX, pMP->mTrackProjY are in Image frame, they are computed in Frame::isInFrustum (see my_notes)
                */
                cv::KeyPoint keyPoint(pMP->mTrackProjX, pMP->mTrackProjY, 1.0);// @todo Do. Nov 2 2023 15:51 (size of the keypoint)
                // cout << endl << "TrackProjCoordinates: (" << pMP->mTrackProjX << ", " << pMP->mTrackProjY << ")"; // e.g. (468.877, 61.7098)
                vMPKeypoint.push_back(keyPoint); 
                projectedMapPoints.push_back(pMP);
            } 

        } 

        if (!vMPDescriptors.empty()) {
            // Concatenate all the descriptors vertically into a single cv::Mat
            cv::vconcat(vMPDescriptors, MPDescriptors);
        }

        SELMSLAM::BBLightGlue bblg(BBLIGHTGLUE_WEIGHT_PATH, 0.0f); 
        int nmatches = bblg.match(projectedMapPoints, MPDescriptors, vMPKeypoint, F, BBLightGlue::CallbackCheckMapPoint);

        return nmatches;
    }

    /**
     * In ORB-SLAM3, this metyhod is called SearchByProjection.
     * to find matches between features in the current frame and the last frame 
     * by projecting 3D points from the last frame into the current frame's image.
     * 
     * The map point associated with the previous frame are projected into the current frame using the predicted camera pose. 
     * This projection provides initial estimates for the location of features in the current frame.
     * 
     * read paper: "ORB-SLAM: A Versatile and Accurate Monocular SLAM System"
     * section V. Tracking B. Initial Pose Estimation from Previous Frame
     * 
     * I changed the logic. I send the last frame features associated with valid map points 
     * and the corresponding descriptors (descriptors of map points) as input to the LightGlue model 
     * and find the matches between them and features in the current frame.
    */
    int BBLGMatcher::TrackLastFrameMapPoints(Frame &CurrentFrame, const Frame &LastFrame) {

        // initialization
        vector<MapPoint*> vpMapPoints; // an empty vector
        cv::Mat MPDescriptors; // MapPoints' descriptors
        vector<cv::KeyPoint> vMPKeypoint; // corresponding MapPoints' Keypoints
        vector<cv::Mat> vMPDescriptors;

        // Current frame 
        const Sophus::SE3f Tcw = CurrentFrame.GetPose();

        // iterates through the features in the last frame
        for(int i = 0; i < LastFrame.N; i++) {
            
            MapPoint* pMP = LastFrame.mvpMapPoints[i];

            // Cond. 1
            if(pMP) {
                const cv::Mat dMP = pMP->GetDescriptor();

                // Cond. 2
                if (dMP.empty()) {
                    continue;
                }

                // Con. 3
                if(!LastFrame.mvbOutlier[i]) {
                    // Project
                    Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                    // projects the 3D point x3Dw from the last frame into the current frame
                    Eigen::Vector3f x3Dc = Tcw * x3Dw;

                    const float invzc = 1.0 / x3Dc(2);

                    // Cond. 4
                    if(invzc < 0) {
                        continue;
                    }

                    // computes the 2D pixel coordinates uv in the current frame corresponding to x3Dc.
                    Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    // Cond. 5
                    // checks whether uv is within the image bounds of the current frame.
                    if(uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX) {
                        continue;
                    }
                    if(uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY) {
                        continue;
                    }

                    // cv::KeyPoint keyPoint(pMP->mTrackProjX, pMP->mTrackProjY, 1.0);// @todo Do. Nov 2 2023 15:51 (size of the keypoint)
                    // cv::KeyPoint keyPoint(LastFrame.mvKeys[i].pt.x, LastFrame.mvKeys[i].pt.y, 1.0);
                    cv::KeyPoint keyPoint = LastFrame.mvKeys[i];

                    vMPKeypoint.push_back(keyPoint); 
                    vpMapPoints.push_back(pMP);
                    vMPDescriptors.push_back(dMP);

                }
            }
        }

        if (!vMPDescriptors.empty()) {
            // MPDescriptors = cv::Mat(vMPDescriptors.size(), vMPDescriptors[0].cols, vMPDescriptors[0].type());
            // cv::vconcat(vMPDescriptors, MPDescriptors);
            MPDescriptors = vMPDescriptors[0].clone();
            for (size_t i = 1; i < vMPDescriptors.size(); ++i) {
                cv::vconcat(MPDescriptors, vMPDescriptors[i], MPDescriptors);
            }
        }

        SELMSLAM::BBLightGlue bblg(BBLIGHTGLUE_WEIGHT_PATH, 0.0f); 

        TIC
        int nmatches = bblg.match(vpMapPoints, MPDescriptors, vMPKeypoint, CurrentFrame, BBLightGlue::CallbackCheckObservations);
        TOC

        return nmatches;
    }


    
}