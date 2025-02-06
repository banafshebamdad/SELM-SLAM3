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


        cout << endl << "B.B In BBLGMatcher::SearchByProjection. At the top of the function #MPs: " << vpMapPoints.size() << ", #features: " << F.N << endl;
        for(size_t iMP = 0; iMP < vpMapPoints.size(); iMP++) {

            MapPoint* pMP = vpMapPoints[iMP];

            if(!pMP->mbTrackInView && !pMP->mbTrackInViewR) {
            // if(!pMP->mbTrackInView){
                continue;
            }

            // bFarPoints = false , thFarPoints = 0
            // pMP->mTrackDepthis the distance of the MapPoint from the camera center (in camera coordinate system) (see my_notes)
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

        cout << endl << "B.B In BBLGMatcher::SearchByProjection. #MPs: " << vpMapPoints.size() << ", # projectedMapPoints: " << projectedMapPoints.size() << ", # Fs: " << F.N << endl;
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

    /**
     * This method is used in Monocular Initialization step
     * vnMatches12: to store the indices of matches
    */
    int BBLGMatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12) {

        // 
        // Preparation for LightGlue
        // 

        // F1: IniFrame
        SELMSLAM::ImageFeatures features1;

        long unsigned int f1Id = F1.mnId;
        cv::Mat f1Descriptors = F1.mDescriptors;
        vector<cv::KeyPoint> f1Keypoints = F1.mvKeysUn;

        features1.img_idx = f1Id;
        features1.img_size = cv::Size(640, 480); // @todo Di Okt. 31 07:45 read from settings
        features1.keypoints = f1Keypoints;
        features1.descriptors = f1Descriptors.getUMat(cv::ACCESS_FAST);

        // F2: current frame
        SELMSLAM::ImageFeatures features2;

        long unsigned int f2Id = F2.mnId;
        cv::Mat f2Descriptors = F2.mDescriptors;
        vector<cv::KeyPoint> f2Keypoints = F2.mvKeysUn;

        features2.img_idx = f2Id;
        features2.img_size = cv::Size(640, 480);
        features2.keypoints = f2Keypoints;
        features2.descriptors = f2Descriptors.getUMat(cv::ACCESS_FAST);

        // 
        // matching process
        // 

        SELMSLAM::MatchesInfo matches_info;
        float matchThresh = 0.0f;
        SELMSLAM::BBLightGlue bblg(BBLIGHTGLUE_WEIGHT_PATH, matchThresh);
        bblg.perform_match(features1, features2, matches_info);

        vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);

        // for (int i = 0; i < F1.mvKeysUn.size(); ++i) {
        for (int i = 0; i < matches_info.match1counts; ++i) {
            vnMatches12[i] = matches_info.vmatch1[i];
        }

        // to store unique pairs of matched keypoints. This set will be used to ensure that duplicates are not included in the final matches.
        std::set<std::pair<int, int> > matches;
        for (int i = 0; i < matches_info.match1counts; i++) {
            if (matches_info.vmatch1[i] > -1 && matches_info.vmscore1[i] > matchThresh && matches_info.vmatch2[matches_info.vmatch1[i]] == i) {
                cv::DMatch mt;
                mt.queryIdx = i;
                mt.trainIdx = matches_info.vmatch1[i];
                matches_info.matches.push_back(mt);
                matches.insert(std::make_pair(mt.queryIdx, mt.trainIdx));
            }
        }
        for (int i = 0; i < matches_info.match2counts; i++) {
            if (matches_info.vmatch2[i] > -1 && matches_info.vmscore2[i] > matchThresh && matches_info.vmatch1[matches_info.vmatch2[i]] == i) {
                cv::DMatch mt;
                mt.queryIdx = matches_info.vmatch2[i];
                mt.trainIdx = i;
                if (matches.find(std::make_pair(mt.queryIdx, mt.trainIdx)) == matches.end()) {
                    matches_info.matches.push_back(mt);
                }
            }
        }

        // B.B Updates the vbPrevMatched vector with the new matches for the next iteration
        //Update prev matched
        for(size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++) {
            if(vnMatches12[i1] >= 0) {
                vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;
            }
        }
        
        return matches_info.matches.size();
    }

    // @todo Mi Feb 7, 2020
    // To search matches that fullfil epipolar constraint, see ORBmatcher::SearchForTriangulation
    int BBLGMatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, vector<pair<size_t, size_t> > &vMatchedPairs) {


        // 
        // Preparation for LightGlue
        // 

        // pKF1
        SELMSLAM::ImageFeatures features1;

        long unsigned int f1Id = pKF1->mnId;
        cv::Mat f1Descriptors = pKF1->mDescriptors;
        vector<cv::KeyPoint> f1Keypoints = pKF1->mvKeysUn;

        features1.img_idx = f1Id;
        features1.img_size = cv::Size(640, 480); // @todo Di Okt. 31 07:45 read from settings
        features1.keypoints = f1Keypoints;
        features1.descriptors = f1Descriptors.getUMat(cv::ACCESS_FAST);

        // pKF2
        SELMSLAM::ImageFeatures features2;

        long unsigned int f2Id = pKF2->mnId;
        cv::Mat f2Descriptors = pKF2->mDescriptors;
        vector<cv::KeyPoint> f2Keypoints = pKF2->mvKeysUn;

        features2.img_idx = f2Id;
        features2.img_size = cv::Size(640, 480);
        features2.keypoints = f2Keypoints;
        features2.descriptors = f2Descriptors.getUMat(cv::ACCESS_FAST);

        // 
        // matching process
        // 

        SELMSLAM::MatchesInfo matches_info;
        float matchThresh = 0.0f;
        SELMSLAM::BBLightGlue bblg(BBLIGHTGLUE_WEIGHT_PATH, matchThresh);
        bblg.perform_match(features1, features2, matches_info);

        // to store unique pairs of matched keypoints. This set will be used to ensure that duplicates are not included in the final matches.
        std::set<std::pair<int, int> > matches;
        for (int i = 0; i < matches_info.match1counts; i++) {
            if (matches_info.vmatch1[i] > -1 && matches_info.vmscore1[i] > matchThresh && matches_info.vmatch2[matches_info.vmatch1[i]] == i) {
                cv::DMatch mt;
                mt.queryIdx = i;
                mt.trainIdx = matches_info.vmatch1[i];
                matches_info.matches.push_back(mt);
                matches.insert(std::make_pair(mt.queryIdx, mt.trainIdx));
            }
        }
        for (int i = 0; i < matches_info.match2counts; i++) {
            if (matches_info.vmatch2[i] > -1 && matches_info.vmscore2[i] > matchThresh && matches_info.vmatch1[matches_info.vmatch2[i]] == i) {
                cv::DMatch mt;
                mt.queryIdx = matches_info.vmatch2[i];
                mt.trainIdx = i;
                if (matches.find(std::make_pair(mt.queryIdx, mt.trainIdx)) == matches.end()) {
                    matches_info.matches.push_back(mt);
                }
            }
        }

        ////////////////////////

        int nmatches = matches_info.matches.size();

        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);

        for (int i = 0; i < matches_info.match1counts; ++i) {
            if(matches_info.vmatch1[i] < 0)
                continue;
            vMatchedPairs.push_back(make_pair(i, matches_info.vmatch1[i]));
        }

        return nmatches;
    }

}