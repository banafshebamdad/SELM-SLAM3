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

#include "Defs.h"
#include "ORBmatcher.h"

#define WITH_TICTOC
#include <tictoc.hpp>
#define WITH_TICTOC

#include<limits.h>

#include<opencv2/core/core.hpp>

// #define DEBUG_PRINTF

#ifdef DEBUG_PRINTF
#define DBG_PRINTF printf
#else
#define DBG_PRINTF(...)
#endif


#ifdef USE_DBOW2
    #include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#else
// namespace DBoW = DBoW2; //alias for namespace
    #include "Thirdparty/DBow3/src/FeatureVector.h"
// namespace DBoW = DBow3; //alias for namespace
#endif

#include<stdint-gcc.h>

// #define ENABLE_CHECK_ORIENTATION

using namespace std;

namespace ORB_SLAM3
{

    const float ORBmatcher::TH_HIGH = 0.90; // B.B change to float by SPT
    const float ORBmatcher::TH_LOW = 0.30; // B.B change to float by SPT
    const int ORBmatcher::HISTO_LENGTH = 30;

    ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
    {
    }

    // B.B to find matches between a frame and a set of map points 
    // B. bFarPoints: whether to consider far points.
    int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints) {
        //DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);

        // B.B counting matches and tracking left and right matches.
        int nmatches = 0, left = 0, right = 0;

        const bool bFactor = th != 1.0;

        // B.B Examine each candidate MapPoint
        for(size_t iMP = 0; iMP < vpMapPoints.size(); iMP++) {
            MapPoint* pMP = vpMapPoints[iMP];

            /**
             * B.B
             * conditional statements are used to filter out map points based on their properties, such as 
             * whether they are tracked in the current view, 
             * their track depth, and 
             * whether they are considered "bad" map points.
            */
            if(!pMP->mbTrackInView && !pMP->mbTrackInViewR)
                continue;

            if(bFarPoints && pMP->mTrackDepth > thFarPoints)
                continue;

            if(pMP->isBad())
                continue;

            if(pMP->mbTrackInView) {
                const int &nPredictedLevel = pMP->mnTrackScaleLevel;

                // The size of the window will depend on the viewing direction
                // B.B calculates a search radius (r) based on the viewing angle and possibly scales it by a factor (th).
                // B.B r is used to determine the region around a map point (or feature???) within which the algorithm will search for potential matching features in the frame
                /**
                 * B.B
                 * "viewing angle": the angle between the optical axis of a camera and the line of sight to a point in the camera's field of view
                 * The viewing angle is typically measured as the angle between the optical axis and the line connecting the camera's optical center 
                 * to the point of interest. This angle can be expressed in degrees or radians.
                 * the cosine of the viewing angle (pMP->mTrackViewCos and pMP->mTrackViewCosR for left and right camera views) 
                 * is used to help determine the search radius for matching features. 
                */
                float r = RadiusByViewingCos(pMP->mTrackViewCos);

                if(bFactor)
                    r *= th;

                // B.B searches for features in the frame within r radius.
                const vector<size_t> vIndices =
                        F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r * F.mvScaleFactors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel);

                if(!vIndices.empty()) {
                    const cv::Mat MPdescriptor = pMP->GetDescriptor();

                    float bestDist = 256;
                    int bestLevel = -1;
                    float bestDist2 = 256;
                    int bestLevel2 = -1;
                    int bestIdx = -1 ;

                    // TIC
                    // Get best and second matches with near keypoints
                    // B.B For each feature found, it computes a descriptor distance and keeps track of the best and second-best matches.
                    for(vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
                        const size_t idx = *vit;

                        if(F.mvpMapPoints[idx]) {
                            // B.B the map point is associated with keypoints in the image.
                            // B.B skip map points that have already been matched to keypoints in the current frame, 
                            // B.B avoiding redundant matching and processing for those points. 
                            if(F.mvpMapPoints[idx]->Observations() > 0)
                                continue;
                        }

                        // B.B F is a right frame in a stereo camera setup
                        if(F.Nleft == -1 && F.mvuRight[idx] > 0) {
                            const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                            if(er>r*F.mvScaleFactors[nPredictedLevel])
                                continue;
                        }

                        const cv::Mat &d = F.mDescriptors.row(idx);

                        cout << "B.B Line 149 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                        // cin.get();
                        const float dist = DescriptorDistance(MPdescriptor, d);

                        if(dist < bestDist) {
                            bestDist2 = bestDist;
                            bestDist = dist;
                            bestLevel2 = bestLevel;
                            bestLevel = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                        : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                          : F.mvKeysRight[idx - F.Nleft].octave;
                            bestIdx=idx;
                        } else if(dist < bestDist2) {
                            bestLevel2 = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                         : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                           : F.mvKeysRight[idx - F.Nleft].octave;
                            bestDist2 = dist;
                        }
                    } // B.B loop over features founded in r area
                    // TOC

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    // B.B applies a ratio test to the best and second-best matches to determine if they are valid matches.
                    if(bestDist <= TH_HIGH) {
                        if(bestLevel == bestLevel2 && bestDist > mfNNratio*bestDist2)
                            continue;

                        if(bestLevel != bestLevel2 || bestDist <= mfNNratio*bestDist2) {
                            F.mvpMapPoints[bestIdx] = pMP;

                            if(F.Nleft != -1 && F.mvLeftToRightMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                                F.mvpMapPoints[F.mvLeftToRightMatch[bestIdx] + F.Nleft] = pMP;
                                nmatches++;
                                right++;
                            }

                            nmatches++;
                            left++;
                        }
                    }
                } // B.B if(!vIndices.empty()): features in r area
            } // B.B if(pMP->mbTrackInView)

            // B.B If stereo information is available (indicated by F.Nleft != -1), it also checks for stereo matches and updates the map points accordingly.
            if(F.Nleft != -1 && pMP->mbTrackInViewR) { 
                const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
                if(nPredictedLevel != -1){
                    float r = RadiusByViewingCos(pMP->mTrackViewCosR);

                    const vector<size_t> vIndices =
                            F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel,true);

                    if(vIndices.empty())
                        continue;

                    const cv::Mat MPdescriptor = pMP->GetDescriptor();

                    float bestDist=256;
                    int bestLevel= -1;
                    float bestDist2=256;
                    int bestLevel2 = -1;
                    int bestIdx =-1 ;

                    // TIC
                    // Get best and second matches with near keypoints
                    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                    {
                        const size_t idx = *vit;

                        if(F.mvpMapPoints[idx + F.Nleft])
                            if(F.mvpMapPoints[idx + F.Nleft]->Observations()>0)
                                continue;


                        const cv::Mat &d = F.mDescriptors.row(idx + F.Nleft);

                        cout << "B.B Line 223 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                        // cin.get();
                        const float dist = DescriptorDistance(MPdescriptor,d);

                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;
                            bestDist=dist;
                            bestLevel2 = bestLevel;
                            bestLevel = F.mvKeysRight[idx].octave;
                            bestIdx=idx;
                        }
                        else if(dist<bestDist2)
                        {
                            bestLevel2 = F.mvKeysRight[idx].octave;
                            bestDist2=dist;
                        }
                    }
                    // TOC

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    if(bestDist<=TH_HIGH) {
                        if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                            continue;

                        if(F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                            F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
                            nmatches++;
                            left++;
                        }


                        F.mvpMapPoints[bestIdx + F.Nleft] = pMP;
                        nmatches++;
                        right++;
                    }
                }
            } // B.B If stereo information is available
        } // B.B for loop on MapPoint

        DBG_PRINTF("Returned nmatches: %d \n", nmatches);
        return nmatches;
    }

    float ORBmatcher::RadiusByViewingCos(const float &viewCos) {
        if(viewCos > 0.998)
            return 2.5;
        else
            return 4.0;
    }

    /**
     * B.B
     * responsible for matching keyframe descriptors (ORB features) to descriptors in a current frame 
     * and returning a list of matched MapPoints (3D points in the environment).
    */
    int ORBmatcher::SearchByBoW(KeyFrame* pKF, Frame &F, vector<MapPoint*> &vpMapPointMatches) {
        //DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);

        // B.B retrieves a vector of MapPoint pointers from the keyframe
        const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

        // B.B will store matched MapPoints for the current frame
        vpMapPointMatches = vector<MapPoint*>(F.N, static_cast<MapPoint*>(NULL));

#ifdef USE_DBOW2
        const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;
#else
        const DBoW3::FeatureVector &vFeatVecKF = pKF->mFeatVec;
#endif

        int nmatches = 0;

        // B.B a histogram rotHist for orientation information
        vector<int> rotHist[HISTO_LENGTH];
        for(int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);

        // B.B for calculating orientation histogram bins
        const float factor = 1.0f / HISTO_LENGTH;

        // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
 #ifdef USE_DBOW2
        DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
        DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
        DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
        DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end(); 
#else
        DBoW3::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
        DBoW3::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
        DBoW3::FeatureVector::const_iterator KFend = vFeatVecKF.end();
        DBoW3::FeatureVector::const_iterator Fend = F.mFeatVec.end(); 
#endif 

        // B.B Feature matching loop
        // B.B calculates the distance between the descriptors
        while(KFit != KFend && Fit != Fend) {

            // B.B if the NodeId IDs of the currently pointed elements in the keyframe and current frame feature vectors are the same.
            if(KFit->first == Fit->first) {

                // B.B contain the indices of the features with the same Node ID in the keyframe and the current frame
                // B.B These indices are used to access the actual feature data for matching
                const vector<unsigned int> vIndicesKF = KFit->second;
                const vector<unsigned int> vIndicesF = Fit->second;

                // B.B initiates a loop that iterates over the indices of features in the keyframe that have the same ID as 
                // B.B the current feature being considered in the current frame
                for(size_t iKF = 0; iKF < vIndicesKF.size(); iKF++) {

                    // B.B retrieve the index realIdxKF and the associated MapPoint object pMP from the vpMapPointsKF vector
                    const unsigned int realIdxKF = vIndicesKF[iKF];
                    MapPoint* pMP = vpMapPointsKF[realIdxKF];

                    // B.B check whether the pMP is a valid MapPoint and not bad the loop continues to the next iteration
                    if(!pMP)
                        continue;

                    if(pMP->isBad())
                        continue;

                    // B.B retrieves the descriptor (dKF) for the realIdxKF-th feature from the keyframe
                    const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF);

                    float bestDist1 = 256;
                    int bestIdxF = -1 ;
                    float bestDist2 = 256;

                    float bestDist1R = 256;
                    int bestIdxFR = -1 ;
                    float bestDist2R = 256;

                    // TIC
                    for(size_t iF = 0; iF < vIndicesF.size(); iF++) {
                        if(F.Nleft == -1){
                            const unsigned int realIdxF = vIndicesF[iF];

                            if(vpMapPointMatches[realIdxF])
                                continue;

                            const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                            // B.B calculates the distance between the descriptors and selects the two best matches
                            cout << "B.B Line 364 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                            // cin.get();
                            const float dist =  DescriptorDistance(dKF, dF);

                            if(dist < bestDist1) {
                                bestDist2 = bestDist1;
                                bestDist1 = dist;
                                bestIdxF = realIdxF;
                            } else if(dist < bestDist2) {
                                bestDist2 = dist;
                            }
                        } else {
                            const unsigned int realIdxF = vIndicesF[iF];

                            if(vpMapPointMatches[realIdxF])
                                continue;

                            const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                            cout << "B.B Line 381 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                            // cin.get();
                            const float dist =  DescriptorDistance(dKF,dF);

                            if(realIdxF < F.Nleft && dist<bestDist1){
                                bestDist2=bestDist1;
                                bestDist1=dist;
                                bestIdxF=realIdxF;
                            } else if(realIdxF < F.Nleft && dist<bestDist2){
                                bestDist2=dist;
                            }

                            if(realIdxF >= F.Nleft && dist<bestDist1R){
                                bestDist2R=bestDist1R;
                                bestDist1R=dist;
                                bestIdxFR=realIdxF;
                            } else if(realIdxF >= F.Nleft && dist<bestDist2R){
                                bestDist2R=dist;
                            }
                        }

                    }
                    // TOC

                    // B.B checks if the distance of the best match is less than or equal to a threshold, indicating a potentially good match
                    if(bestDist1 <= TH_LOW) {

                        /**
                         * checks if the best match distance satisfies a nearest-neighbor ratio test, 
                         * ensuring that it is significantly better than the second-best match. 
                         * If this condition is met, it indicates a successful match.
                        */
                        if(static_cast<float>(bestDist1) < mfNNratio* static_cast<float>(bestDist2)) {

                            // B.B associates the matched MapPoint (pMP) with the feature in the current frame with index bestIdxF
                            vpMapPointMatches[bestIdxF] = pMP; // B.B Important

                            const cv::KeyPoint &kp =
                                    (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                    (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                                : pKF -> mvKeys[realIdxKF];
#ifdef ENABLE_CHECK_ORIENTATION
                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint &Fkp =
                                        (!pKF->mpCamera2 || F.Nleft == -1) ? F.mvKeys[bestIdxF] :
                                        (bestIdxF >= F.Nleft) ? F.mvKeysRight[bestIdxF - F.Nleft]
                                                              : F.mvKeys[bestIdxF];

                                float rot = kp.angle-Fkp.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxF);
                            }
#endif
                            nmatches++;
                        }

                        if(bestDist1R <= TH_LOW) {
                            if(static_cast<float>(bestDist1R)<mfNNratio* static_cast<float>(bestDist2R) || true) {
                                vpMapPointMatches[bestIdxFR] = pMP;

                                const cv::KeyPoint &kp =
                                        (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                        (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                                    : pKF -> mvKeys[realIdxKF];
#ifdef ENABLE_CHECK_ORIENTATION
                                if(mbCheckOrientation)
                                {
                                    cv::KeyPoint &Fkp =
                                            (!F.mpCamera2) ? F.mvKeys[bestIdxFR] :
                                            (bestIdxFR >= F.Nleft) ? F.mvKeysRight[bestIdxFR - F.Nleft]
                                                                   : F.mvKeys[bestIdxFR];

                                    float rot = kp.angle-Fkp.angle;
                                    if(rot<0.0)
                                        rot+=360.0f;
                                    int bin = round(rot*factor);
                                    if(bin==HISTO_LENGTH)
                                        bin=0;
                                    assert(bin>=0 && bin<HISTO_LENGTH);
                                    rotHist[bin].push_back(bestIdxFR);
                                }
#endif
                                nmatches++;
                            }
                        }
                    }

                }

                KFit++;
                Fit++;
            /**
             * B.B 
             * If the feature ID in the keyframe is less than that in the current frame, 
             * it means that the feature in the keyframe is not found in the current frame. 
             * The code advances the keyframe feature iterator (KFit) to the next feature in the keyframe.
            */
            } else if(KFit->first < Fit->first) {
                KFit = vFeatVecKF.lower_bound(Fit->first);
            
            /**
             * B.B
             * If the feature ID in the current frame is less than that in the keyframe, 
             * it means that the feature in the current frame is not found in the keyframe. 
             * The code advances the current frame feature iterator (Fit) to the next feature in the current frame.
            */
            } else {
                Fit = F.mFeatVec.lower_bound(KFit->first);
            }
        } // B.B feature matching while loop
#ifdef ENABLE_CHECK_ORIENTATION
        if(mbCheckOrientation) {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++) {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
#endif
        DBG_PRINTF("Returned nmatches: %d \n", nmatches);

        return nmatches;
    }

    int ORBmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3f &Scw, const vector<MapPoint*> &vpPoints,
                                       vector<MapPoint*> &vpMatched, int th, float ratioHamming)
    {
        //DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

        int nmatches=0;

        // For each Candidate MapPoint Project and Match
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0)
                continue;

            // Project into Image
            const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

            // Point must be inside the image
            if(!pKF->IsInImage(uv(0),uv(1)))
                continue;

            // Depth must be inside the scale invariance region of the point
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist = PO.norm();

            if(dist<minDistance || dist>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist)
                continue;

            int nPredictedLevel = pMP->PredictScale(dist,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            float bestDist = 256;
            int bestIdx = -1;
            // TIC
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;
                if(vpMatched[idx])
                    continue;

                const int &kpLevel= pKF->mvKeysUn[idx].octave;

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                cout << "B.B Line 608 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                // cin.get();
                const float dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }
            // TOC

            if(bestDist<=TH_LOW*ratioHamming)
            {
                vpMatched[bestIdx]=pMP;
                nmatches++;
            }

        }
        DBG_PRINTF("Returned nmatches: %d \n", nmatches);
        return nmatches;
    }

    int ORBmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs,
                                       std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
    {
        //DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

        int nmatches=0;

        // For each Candidate MapPoint Project and Match
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];
            KeyFrame* pKFi = vpPointsKFs[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0)
                continue;

            // Project into Image
            const float invz = 1/p3Dc(2);
            const float x = p3Dc(0)*invz;
            const float y = p3Dc(1)*invz;

            const float u = fx*x+cx;
            const float v = fy*y+cy;

            // Point must be inside the image
            if(!pKF->IsInImage(u,v))
                continue;

            // Depth must be inside the scale invariance region of the point
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist = PO.norm();

            if(dist<minDistance || dist>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist)
                continue;

            int nPredictedLevel = pMP->PredictScale(dist,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            float bestDist = 256;
            int bestIdx = -1;
            
            // TIC
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;
                if(vpMatched[idx])
                    continue;

                const int &kpLevel= pKF->mvKeysUn[idx].octave;

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                cout << "B.B Line 725 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                // cin.get();
                const float dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }
            // TOC

            if(bestDist<=TH_LOW*ratioHamming)
            {
                vpMatched[bestIdx] = pMP;
                vpMatchedKF[bestIdx] = pKFi;
                nmatches++;
            }

        }
        DBG_PRINTF("Returned nmatches: %d \n", nmatches);
        return nmatches;
    }

    int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
    {
        //DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);
        int nmatches=0;
        vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
        vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

        DBG_PRINTF("Num keypoints F1: %d, num keypoints F2: %d\n",F1.mvKeysUn.size(),F2.mvKeysUn.size());

        for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
        {
            cv::KeyPoint kp1 = F1.mvKeysUn[i1];
            int level1 = kp1.octave;
            if(level1>0)
                continue;

            vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

            if(vIndices2.empty())
                continue;

            cv::Mat d1 = F1.mDescriptors.row(i1);

            float bestDist = INT_MAX;
            float bestDist2 = INT_MAX;
            int bestIdx2 = -1;

            // TIC
            for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
            {
                size_t i2 = *vit;

                cv::Mat d2 = F2.mDescriptors.row(i2);

                cout << "B.B Line 788 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                // cin.get();
                float dist = DescriptorDistance(d1,d2);

                if(vMatchedDistance[i2]<=dist)
                    continue;

                if(dist<bestDist)
                {
                    bestDist2=bestDist;
                    bestDist=dist;
                    bestIdx2=i2;
                }
                else if(dist<bestDist2)
                {
                    bestDist2=dist;
                }
            }
            // TOC

            if(bestDist<=TH_LOW)
            {
                if(bestDist<(float)bestDist2*mfNNratio)
                {
                    if(vnMatches21[bestIdx2]>=0)
                    {
                        vnMatches12[vnMatches21[bestIdx2]]=-1;
                        nmatches--;
                    }
                    vnMatches12[i1]=bestIdx2;
                    vnMatches21[bestIdx2]=i1;
                    vMatchedDistance[bestIdx2]=bestDist;
                    nmatches++;
#ifdef ENABLE_CHECK_ORIENTATION
                    if(mbCheckOrientation)
                    {
                        float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(i1);
                    }
#endif
                }
            }

        }
#ifdef ENABLE_CHECK_ORIENTATION
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    int idx1 = rotHist[i][j];
                    if(vnMatches12[idx1]>=0)
                    {
                        vnMatches12[idx1]=-1;
                        nmatches--;
                    }
                }
            }

        }
#endif
        //Update prev matched
        for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
            if(vnMatches12[i1]>=0)
                vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

        DBG_PRINTF("Returned nmatches: %d \n", nmatches);
        return nmatches;
    }

    int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
    {
        //DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);
        const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
#ifdef USE_DBOW2
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
#else
        const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
#endif 
        const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
        const cv::Mat &Descriptors1 = pKF1->mDescriptors;

        const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
#ifdef USE_DBOW2
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
#else
        const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
#endif 

        const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
        const cv::Mat &Descriptors2 = pKF2->mDescriptors;

        vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
        vector<bool> vbMatched2(vpMapPoints2.size(),false);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;

        int nmatches = 0;

#ifdef USE_DBOW2
        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
#else
        DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();
#endif 
        while(f1it != f1end && f2it != f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];
                    if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
                        continue;
                    }

                    MapPoint* pMP1 = vpMapPoints1[idx1];
                    if(!pMP1)
                        continue;
                    if(pMP1->isBad())
                        continue;

                    const cv::Mat &d1 = Descriptors1.row(idx1);

                    float bestDist1=256;
                    int bestIdx2 =-1 ;
                    float bestDist2=256;

                    // TIC
                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        const size_t idx2 = f2it->second[i2];

                        if(pKF2 -> NLeft != -1 && idx2 >= pKF2 -> mvKeysUn.size()){
                            continue;
                        }

                        MapPoint* pMP2 = vpMapPoints2[idx2];

                        if(vbMatched2[idx2] || !pMP2)
                            continue;

                        if(pMP2->isBad())
                            continue;

                        const cv::Mat &d2 = Descriptors2.row(idx2);

                        cout << "B.B Line 957 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                        // cin.get();
                        float dist = DescriptorDistance(d1,d2);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }
                    // TOC

                    if(bestDist1<TH_LOW)
                    {
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                        {
                            vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2]=true;
#ifdef ENABLE_CHECK_ORIENTATION
                            if(mbCheckOrientation)
                            {
                                float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
#endif

                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }
#ifdef ENABLE_CHECK_ORIENTATION
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
#endif

        DBG_PRINTF("Returned nmatches: %d \n", nmatches);
        return nmatches;
    }

    int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                           vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
    {
        //DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);
        #ifdef USE_DBOW2
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        #else
        const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        #endif  

        //Compute epipole in second image
        Sophus::SE3f T1w = pKF1->GetPose();
        Sophus::SE3f T2w = pKF2->GetPose();
        Sophus::SE3f Tw2 = pKF2->GetPoseInverse(); // for convenience
        Eigen::Vector3f Cw = pKF1->GetCameraCenter();
        Eigen::Vector3f C2 = T2w * Cw;

        Eigen::Vector2f ep = pKF2->mpCamera->project(C2);
        Sophus::SE3f T12;
        Sophus::SE3f Tll, Tlr, Trl, Trr;
        Eigen::Matrix3f R12; // for fastest computation
        Eigen::Vector3f t12; // for fastest computation

        GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

        if(!pKF1->mpCamera2 && !pKF2->mpCamera2){
            T12 = T1w * Tw2;
            R12 = T12.rotationMatrix();
            t12 = T12.translation();
        }
        else{
            Sophus::SE3f Tr1w = pKF1->GetRightPose();
            Sophus::SE3f Twr2 = pKF2->GetRightPoseInverse();
            Tll = T1w * Tw2;
            Tlr = T1w * Twr2;
            Trl = Tr1w * Tw2;
            Trr = Tr1w * Twr2;
        }

        Eigen::Matrix3f Rll = Tll.rotationMatrix(), Rlr  = Tlr.rotationMatrix(), Rrl  = Trl.rotationMatrix(), Rrr  = Trr.rotationMatrix();
        Eigen::Vector3f tll = Tll.translation(), tlr = Tlr.translation(), trl = Trl.translation(), trr = Trr.translation();

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node
        int nmatches=0;
        vector<bool> vbMatched2(pKF2->N,false);
        vector<int> vMatches12(pKF1->N,-1);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;

        #ifdef USE_DBOW2
        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
        #else
        DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();
        #endif

        auto tic = std::chrono::system_clock::now();

        while(f1it!=f1end && f2it!=f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];

                    MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

                    // If there is already a MapPoint skip
                    if(pMP1)
                    {
                        continue;
                    }

                    const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1]>=0);

                    if(bOnlyStereo)
                        if(!bStereo1)
                            continue;

                    const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                    : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                             : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                    const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                       : true;

                    const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                    float bestDist = TH_LOW;
                    int bestIdx2 = -1;
                    // TIC
                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        size_t idx2 = f2it->second[i2];

                        MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                        // If we have already matched or there is a MapPoint skip
                        if(vbMatched2[idx2] || pMP2)
                            continue;

                        const bool bStereo2 = (!pKF2->mpCamera2 &&  pKF2->mvuRight[idx2]>=0);

                        if(bOnlyStereo)
                            if(!bStereo2)
                                continue;

                        const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                        cout << "B.B Line 1158 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                        // cin.get();
                        const float dist = DescriptorDistance(d1,d2);

                        if(dist>TH_LOW || dist>bestDist)
                            continue;

                        const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                        : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                                 : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                        const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                           : true;

                        if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
                        {
                            const float distex = ep(0)-kp2.pt.x;
                            const float distey = ep(1)-kp2.pt.y;
                            if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            {
                                continue;
                            }
                        }

                        if(pKF1->mpCamera2 && pKF2->mpCamera2){
                            if(bRight1 && bRight2){
                                R12 = Rrr;
                                t12 = trr;
                                T12 = Trr;

                                pCamera1 = pKF1->mpCamera2;
                                pCamera2 = pKF2->mpCamera2;
                            }
                            else if(bRight1 && !bRight2){
                                R12 = Rrl;
                                t12 = trl;
                                T12 = Trl;

                                pCamera1 = pKF1->mpCamera2;
                                pCamera2 = pKF2->mpCamera;
                            }
                            else if(!bRight1 && bRight2){
                                R12 = Rlr;
                                t12 = tlr;
                                T12 = Tlr;

                                pCamera1 = pKF1->mpCamera;
                                pCamera2 = pKF2->mpCamera2;
                            }
                            else{
                                R12 = Rll;
                                t12 = tll;
                                T12 = Tll;

                                pCamera1 = pKF1->mpCamera;
                                pCamera2 = pKF2->mpCamera;
                            }

                        }

                        if(bCoarse || pCamera1->epipolarConstrain(pCamera2,kp1,kp2,R12,t12,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave])) // MODIFICATION_2
                        {
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }

                    // TOC
                    if(bestIdx2>=0)
                    {
                        const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                        : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                     : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
                        vMatches12[idx1]=bestIdx2;
                        nmatches++;
#ifdef ENABLE_CHECK_ORIENTATION
                        if(mbCheckOrientation)
                        {
                            float rot = kp1.angle-kp2.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
#endif

                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        auto toc = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = tic-toc;

        // std::cout << "matching time" << elapsed_seconds.count() << std::endl;

#ifdef ENABLE_CHECK_ORIENTATION
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vMatches12[rotHist[i][j]]=-1;
                    nmatches--;
                }
            }

        }
#endif

        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);

        for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
        {
            if(vMatches12[i]<0)
                continue;
            vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
        }

        DBG_PRINTF("Returned nmatches: %d \n", nmatches);
        return nmatches;
    }

    int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
    {
        // DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);
        GeometricCamera* pCamera;
        Sophus::SE3f Tcw;
        Eigen::Vector3f Ow;

        if(bRight){
            Tcw = pKF->GetRightPose();
            Ow = pKF->GetRightCameraCenter();
            pCamera = pKF->mpCamera2;
        }
        else{
            Tcw = pKF->GetPose();
            Ow = pKF->GetCameraCenter();
            pCamera = pKF->mpCamera;
        }

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        int nFused=0;

        const int nMPs = vpMapPoints.size();

        // For debbuging
        int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;
        for(int i=0; i<nMPs; i++)
        {
            MapPoint* pMP = vpMapPoints[i];

            if(!pMP)
            {
                count_notMP++;
                continue;
            }

            if(pMP->isBad())
            {
                count_bad++;
                continue;
            }
            else if(pMP->IsInKeyFrame(pKF))
            {
                count_isinKF++;
                continue;
            }

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0f)
            {
                count_negdepth++;
                continue;
            }

            const float invz = 1/p3Dc(2);

            const Eigen::Vector2f uv = pCamera->project(p3Dc);

            // Point must be inside the image
            if(!pKF->IsInImage(uv(0),uv(1)))
            {
                count_notinim++;
                continue;
            }

            const float ur = uv(0)-bf*invz;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist3D = PO.norm();

            // Depth must be inside the scale pyramid of the image
            if(dist3D<minDistance || dist3D>maxDistance) {
                count_dist++;
                continue;
            }

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist3D)
            {
                count_normal++;
                continue;
            }

            int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius,bRight);

            if(vIndices.empty())
            {
                count_notidx++;
                continue;
            }

            // Match to the most similar keypoint in the radius

            const cv::Mat dMP = pMP->GetDescriptor();

            float bestDist = 256;
            int bestIdx = -1;
            // TIC
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                size_t idx = *vit;
                const cv::KeyPoint &kp = (pKF -> NLeft == -1) ? pKF->mvKeysUn[idx]
                                                              : (!bRight) ? pKF -> mvKeys[idx]
                                                                          : pKF -> mvKeysRight[idx];

                const int &kpLevel= kp.octave;

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                if(pKF->mvuRight[idx]>=0)
                {
                    // Check reprojection error in stereo
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;
                    const float &kpr = pKF->mvuRight[idx];
                    const float ex = uv(0)-kpx;
                    const float ey = uv(1)-kpy;
                    const float er = ur-kpr;
                    const float e2 = ex*ex+ey*ey+er*er;

                    if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                        continue;
                }
                else
                {
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;
                    const float ex = uv(0)-kpx;
                    const float ey = uv(1)-kpy;
                    const float e2 = ex*ex+ey*ey;

                    if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                        continue;
                }

                if(bRight) idx += pKF->NLeft;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                cout << "B.B Line 1458 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                // cin.get();
                const float dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }
            // TOC

            // If there is already a MapPoint replace otherwise add new measurement
            if(bestDist<=TH_LOW)
            {
                MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                if(pMPinKF)
                {
                    if(!pMPinKF->isBad())
                    {
                        if(pMPinKF->Observations()>pMP->Observations())
                            pMP->Replace(pMPinKF);
                        else
                            pMPinKF->Replace(pMP);
                    }
                }
                else
                {
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPoint(pMP,bestIdx);
                }
                nFused++;
            }
            else
                count_thcheck++;

        }

        return nFused;
    }

    int ORBmatcher::Fuse(KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
    {
        //DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

        int nFused=0;

        const int nPoints = vpPoints.size();

        // For each candidate MapPoint project and match
        for(int iMP=0; iMP<nPoints; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0f)
                continue;

            // Project into Image
            const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

            // Point must be inside the image
            if(!pKF->IsInImage(uv(0),uv(1)))
                continue;

            // Depth must be inside the scale pyramid of the image
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist3D = PO.norm();

            if(dist3D<minDistance || dist3D>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist3D)
                continue;

            // Compute predicted scale level
            const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius

            const cv::Mat dMP = pMP->GetDescriptor();

            float bestDist = INT_MAX;
            int bestIdx = -1;
            // TIC
            for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
            {
                const size_t idx = *vit;
                const int &kpLevel = pKF->mvKeysUn[idx].octave;

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                cout << "B.B Line 1586 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                // cin.get();
                float dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }
            // TOC

            // If there is already a MapPoint replace otherwise add new measurement
            if(bestDist<=TH_LOW)
            {
                MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                if(pMPinKF)
                {
                    if(!pMPinKF->isBad())
                        vpReplacePoint[iMP] = pMPinKF;
                }
                else
                {
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPoint(pMP,bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }

    int ORBmatcher::SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th)
    {
        
        // DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);
        const float &fx = pKF1->fx;
        const float &fy = pKF1->fy;
        const float &cx = pKF1->cx;
        const float &cy = pKF1->cy;

        // Camera 1 & 2 from world
        Sophus::SE3f T1w = pKF1->GetPose();
        Sophus::SE3f T2w = pKF2->GetPose();

        //Transformation between cameras
        Sophus::Sim3f S21 = S12.inverse();

        const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
        const int N1 = vpMapPoints1.size();

        const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
        const int N2 = vpMapPoints2.size();

        vector<bool> vbAlreadyMatched1(N1,false);
        vector<bool> vbAlreadyMatched2(N2,false);

        for(int i=0; i<N1; i++)
        {
            MapPoint* pMP = vpMatches12[i];
            if(pMP)
            {
                vbAlreadyMatched1[i]=true;
                int idx2 = get<0>(pMP->GetIndexInKeyFrame(pKF2));
                if(idx2>=0 && idx2<N2)
                    vbAlreadyMatched2[idx2]=true;
            }
        }

        vector<int> vnMatch1(N1,-1);
        vector<int> vnMatch2(N2,-1);

        // Transform from KF1 to KF2 and search
        for(int i1=0; i1<N1; i1++)
        {
            MapPoint* pMP = vpMapPoints1[i1];

            if(!pMP || vbAlreadyMatched1[i1])
                continue;

            if(pMP->isBad())
                continue;

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc1 = T1w * p3Dw;
            Eigen::Vector3f p3Dc2 = S21 * p3Dc1;

            // Depth must be positive
            if(p3Dc2(2)<0.0)
                continue;

            const float invz = 1.0/p3Dc2(2);
            const float x = p3Dc2(0)*invz;
            const float y = p3Dc2(1)*invz;

            const float u = fx*x+cx;
            const float v = fy*y+cy;

            // Point must be inside the image
            if(!pKF2->IsInImage(u,v))
                continue;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            const float dist3D = p3Dc2.norm();

            // Depth must be inside the scale invariance region
            if(dist3D<minDistance || dist3D>maxDistance )
                continue;

            // Compute predicted octave
            const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

            // Search in a radius
            const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            float bestDist = INT_MAX;
            int bestIdx = -1;
            // TIC
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;

                const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

                if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

                cout << "B.B Line 1723 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                // cin.get();
                const float dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }
            // TOC

            if(bestDist<=TH_HIGH)
            {
                vnMatch1[i1]=bestIdx;
            }
        }

        // Transform from KF2 to KF2 and search
        for(int i2=0; i2<N2; i2++)
        {
            MapPoint* pMP = vpMapPoints2[i2];

            if(!pMP || vbAlreadyMatched2[i2])
                continue;

            if(pMP->isBad())
                continue;

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc2 = T2w * p3Dw;
            Eigen::Vector3f p3Dc1 = S12 * p3Dc2;

            // Depth must be positive
            if(p3Dc1(2)<0.0)
                continue;

            const float invz = 1.0/p3Dc1(2);
            const float x = p3Dc1(0)*invz;
            const float y = p3Dc1(1)*invz;

            const float u = fx*x+cx;
            const float v = fy*y+cy;

            // Point must be inside the image
            if(!pKF1->IsInImage(u,v))
                continue;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            const float dist3D = p3Dc1.norm();

            // Depth must be inside the scale pyramid of the image
            if(dist3D<minDistance || dist3D>maxDistance)
                continue;

            // Compute predicted octave
            const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

            // Search in a radius of 2.5*sigma(ScaleLevel)
            const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            float bestDist = INT_MAX;
            int bestIdx = -1;
            // TIC
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;

                const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

                if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

                cout << "B.B Line 1805 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                // cin.get();
                const float dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_HIGH)
            {
                vnMatch2[i2]=bestIdx;
            }
            // TOC
        }

        // Check agreement
        int nFound = 0;

        for(int i1=0; i1<N1; i1++)
        {
            int idx2 = vnMatch1[i1];

            if(idx2>=0)
            {
                int idx1 = vnMatch2[idx2];
                if(idx1==i1)
                {
                    vpMatches12[i1] = vpMapPoints2[idx2];
                    nFound++;
                }
            }
        }

        DBG_PRINTF("Returned nFound: %d \n", nFound);

        return nFound;
    }

    /**
     * B.B
     * B.B To predict the coordinate of featres in the current frame, 
     * by projecting 3D points from the last frame into the current frame's image.
     * ensures that matches are not only geometrically accurate but also rotationally consistent.
     * It also checks for rotation consistency and applies filters based on matching conditions.
    */
    int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono) {
        //DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);
        int nmatches = 0;

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for(int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f / HISTO_LENGTH;

        const Sophus::SE3f Tcw = CurrentFrame.GetPose();
        const Eigen::Vector3f twc = Tcw.inverse().translation();

        const Sophus::SE3f Tlw = LastFrame.GetPose();

        // B.B calculates the translation vector (tlc) by transforming the current camera position from the world coordinate system 
        // B.B to the last frame's camera coordinate system.
        const Eigen::Vector3f tlc = Tlw * twc;

        // B.B bForward is true if the transformed point is in front of the camera (positive Z) 
        const bool bForward = tlc(2) > CurrentFrame.mb && !bMono;

        // B.B true if the transformed point is behind the camera (negative Z) 
        const bool bBackward = -tlc(2) > CurrentFrame.mb && !bMono;

        // B.B iterates through the features in the last frame
        for(int i = 0; i < LastFrame.N; i++) {
            MapPoint* pMP = LastFrame.mvpMapPoints[i];
            if(pMP) {
                if(!LastFrame.mvbOutlier[i]) {
                    // Project
                    Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                    // B.B projects the 3D point x3Dw from the last frame into the current frame
                    Eigen::Vector3f x3Dc = Tcw * x3Dw;

                    const float xc = x3Dc(0);
                    const float yc = x3Dc(1);
                    const float invzc = 1.0 / x3Dc(2);

                    if(invzc < 0)
                        continue;

                    // B.B computes the 2D pixel coordinates uv in the current frame corresponding to x3Dc.
                    // B.B calculates the 2D coordinates on the current frame plane equivalent to a 3D in the current frame's coordinate system.
                    Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    // B.B checks whether uv is within the image bounds of the current frame.
                    if(uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX)
                        continue;
                    if(uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY)
                        continue;

                    // B.B calculates the scale level (nLastOctave) of the feature in the last frame.
                    int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                     : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                    // Search in a window. Size depends on scale
                    float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

                    vector<size_t> vIndices2;

                    // B.B Based on the direction (bForward, bBackward, or both), 
                    // B.b it retrieves feature indices (vIndices2) in the current frame within a certain search radius.
                    if(bForward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave);
                    else if(bBackward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, 0, nLastOctave);
                    else
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave-1, nLastOctave+1);

                    // printf("%s num points to match: %lu \n", __PRETTY_FUNCTION__, vIndices2.size());

                    // B.B no features are found in the search area, it continues to the next feature.
                    if(vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    float bestDist = 256;
                    int bestIdx2 = -1;
                    
                    // TIC
                    for(vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++) {
                        const size_t i2 = *vit;

                        if(CurrentFrame.mvpMapPoints[i2])
                            if(CurrentFrame.mvpMapPoints[i2]->Observations() > 0)
                                continue;

                        if(CurrentFrame.Nleft == -1 && CurrentFrame.mvuRight[i2] > 0) {
                            const float ur = uv(0) - CurrentFrame.mbf * invzc;
                            const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                            if(er > radius)
                                continue;
                        }

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        // B.B calculate the distance between the descriptor of the MapPoint (dMP) and 
                        // B.B the descriptors of potential matches in the current frame.
                        cout << "B.B Line 1951 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                        // cin.get();
                        const float dist = DescriptorDistance(dMP, d);

                        if(dist < bestDist) {
                            bestDist = dist;
                            bestIdx2 = i2;
                        }
                    }
                    // TOC

                    // B.B iterates through the potential matching feature indices and keeps track of the best match based on the smallest distance.
                    // B.B If a match with a small enough distance is found (bestDist is below TH_HIGH), 
                    // B.B the current feature in the current frame is associated with the MapPoint
                    if(bestDist <= TH_HIGH){
                        CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
                        nmatches++;
// B.B If orientation checking is enabled (mbCheckOrientation), additional checks are performed to ensure consistent feature orientation between frames.
#ifdef ENABLE_CHECK_ORIENTATION
                        if(mbCheckOrientation)
                        {
                            cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                        : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                            cv::KeyPoint kpCF = (CurrentFrame.Nleft == -1) ? CurrentFrame.mvKeysUn[bestIdx2]
                                                                           : (bestIdx2 < CurrentFrame.Nleft) ? CurrentFrame.mvKeys[bestIdx2]
                                                                                                             : CurrentFrame.mvKeysRight[bestIdx2 - CurrentFrame.Nleft];
                            float rot = kpLF.angle-kpCF.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
#endif 
                    }
                    // B.B handles stereo cameras, checking the right image if available and performing a similar process.
                    if(CurrentFrame.Nleft != -1){
                        Eigen::Vector3f x3Dr = CurrentFrame.GetRelativePoseTrl() * x3Dc;
                        Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dr);

                        int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                             : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                        // Search in a window. Size depends on scale
                        float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

                        vector<size_t> vIndices2;

                        if(bForward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave, -1,true);
                        else if(bBackward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, 0, nLastOctave, true);
                        else
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave-1, nLastOctave+1, true);

                        const cv::Mat dMP = pMP->GetDescriptor();

                        float bestDist = 256;
                        int bestIdx2 = -1;
                        // TIC
                        for(vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++) {
                            const size_t i2 = *vit;
                            if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft])
                                if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft]->Observations() > 0)
                                    continue;

                            const cv::Mat &d = CurrentFrame.mDescriptors.row(i2 + CurrentFrame.Nleft);

                            cout << "B.B Line 2021 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                            // cin.get();

                            const float dist = DescriptorDistance(dMP,d);

                            if(dist < bestDist) {
                                bestDist = dist;
                                bestIdx2 = i2;
                            }
                        }
                        // TOC

                        if(bestDist <= TH_HIGH) {
                            CurrentFrame.mvpMapPoints[bestIdx2 + CurrentFrame.Nleft] = pMP;
                            nmatches++;
#ifdef ENABLE_CHECK_ORIENTATION
                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                            : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                    : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                                cv::KeyPoint kpCF = CurrentFrame.mvKeysRight[bestIdx2];

                                float rot = kpLF.angle-kpCF.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdx2  + CurrentFrame.Nleft);
                            }
#endif
                        }

                    }
                    // TOC
                
                }
            }
        }

// B.B After processing all features, the code applies a rotation consistency check 
// B.B to ensure that matches are consistent in terms of orientation.
        //Apply rotation consistency
#ifdef ENABLE_CHECK_ORIENTATION
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                        nmatches--;
                    }
                }
            }
        }
#endif
        DBG_PRINTF("Returned nmatches: %d \n", nmatches);
        return nmatches;
    }

    int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
    {
        //DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);
        int nmatches = 0;

        const Sophus::SE3f Tcw = CurrentFrame.GetPose();
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        // TIC
        for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMPs[i];

            if(pMP)
            {
                if(!pMP->isBad() && !sAlreadyFound.count(pMP))
                {
                    //Project
                    Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                    Eigen::Vector3f x3Dc = Tcw * x3Dw;

                    const Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    if(uv(0)<CurrentFrame.mnMinX || uv(0)>CurrentFrame.mnMaxX)
                        continue;
                    if(uv(1)<CurrentFrame.mnMinY || uv(1)>CurrentFrame.mnMaxY)
                        continue;

                    // Compute predicted scale level
                    Eigen::Vector3f PO = x3Dw-Ow;
                    float dist3D = PO.norm();

                    const float maxDistance = pMP->GetMaxDistanceInvariance();
                    const float minDistance = pMP->GetMinDistanceInvariance();

                    // Depth must be inside the scale pyramid of the image
                    if(dist3D<minDistance || dist3D>maxDistance)
                        continue;

                    int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                    // Search in a window
                    const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                    const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius, nPredictedLevel-1, nPredictedLevel+1);

                    if(vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    float bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                    {
                        const size_t i2 = *vit;
                        if(CurrentFrame.mvpMapPoints[i2])
                            continue;

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        cout << "B.B Line 2160 ORBMatcher.cc. before calling ORBmatcher::DescriptorDistance. Press Enter..."; 
                        // cin.get();

                        const float dist = DescriptorDistance(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=ORBdist)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                        nmatches++;
#ifdef ENABLE_CHECK_ORIENTATION
                        if(mbCheckOrientation)
                        {
                            float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
#endif
                    }

                }
            }
        }
#ifdef ENABLE_CHECK_ORIENTATION
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                        nmatches--;
                    }
                }
            }
        }
#endif
        // TOC
        DBG_PRINTF("Returned nmatches: %d \n", nmatches);
        return nmatches;
    }

    void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1=0;
        int max2=0;
        int max3=0;

        for(int i=0; i<L; i++)
        {
            const int s = histo[i].size();
            if(s>max1)
            {
                max3=max2;
                max2=max1;
                max1=s;
                ind3=ind2;
                ind2=ind1;
                ind1=i;
            }
            else if(s>max2)
            {
                max3=max2;
                max2=s;
                ind3=ind2;
                ind2=i;
            }
            else if(s>max3)
            {
                max3=s;
                ind3=i;
            }
        }

        if(max2<0.1f*(float)max1)
        {
            ind2=-1;
            ind3=-1;
        }
        else if(max3<0.1f*(float)max1)
        {
            ind3=-1;
        }
    }


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    float ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
        
        //DBG_PRINTF("%s \n", __PRETTY_FUNCTION__);
#ifdef USE_BINARY_DESCRIPTORS 
    
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist=0;

        for(int i=0; i<8; i++, pa++, pb++)
        {
            unsigned  int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }
        return (float)dist;
#else
        // TIC
        cout << endl << "B.B in ORBmatcher::DescriptorDistance. Before computing the NORM_L2 of two descriptors.";
        // cin.get();

            float dist = (float)cv::norm(a, b, cv::NORM_L2);
        
        cout << endl << "B.B in ORBmatcher::DescriptorDistance. After computing the NORM_L2 of two descriptors. dist=" << dist;
        // cin.get();

            // DBG_PRINTF("%s, distance evaluated: %f \n", __PRETTY_FUNCTION__, dist);
        // TOC
            return dist;

#endif
        
    }

} //namespace ORB_SLAM
