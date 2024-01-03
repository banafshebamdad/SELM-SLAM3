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

#include "BBLGMatcher.hpp"

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

using namespace SELMSLAM;

namespace ORB_SLAM3
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint():
    mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL))
{
    mpReplaced = static_cast<MapPoint*>(NULL);
}

MapPoint::MapPoint(const Eigen::Vector3f &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId())
{
    SetWorldPos(Pos);

    mNormalVector.setZero();

    mbTrackInViewR = false;
    mbTrackInView = false;

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

MapPoint::MapPoint(const double invDepth, cv::Point2f uv_init, KeyFrame* pRefKF, KeyFrame* pHostKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId())
{
    mInvDepth=invDepth;
    mInitU=(double)uv_init.x;
    mInitV=(double)uv_init.y;
    mpHostKF = pHostKF;

    mNormalVector.setZero();

    // Worldpos is not set
    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const Eigen::Vector3f &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap), mnOriginMapId(pMap->GetId())
{
    SetWorldPos(Pos);

    Eigen::Vector3f Ow;
    if(pFrame -> Nleft == -1 || idxF < pFrame -> Nleft){
        Ow = pFrame->GetCameraCenter();
    }
    else{
        Eigen::Matrix3f Rwl = pFrame->GetRwc();
        Eigen::Vector3f tlr = pFrame->GetRelativePoseTlr().translation();
        Eigen::Vector3f twl = pFrame->GetOw();

        Ow = Rwl * tlr + twl;
    }
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector / mNormalVector.norm();

    Eigen::Vector3f PC = mWorldPos - Ow;
    const float dist = PC.norm();
    const int level = (pFrame -> Nleft == -1) ? pFrame->mvKeysUn[idxF].octave
                                              : (idxF < pFrame -> Nleft) ? pFrame->mvKeys[idxF].octave
                                                                         : pFrame -> mvKeysRight[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist * levelScaleFactor;
    mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapPoint::SetWorldPos(const Eigen::Vector3f &Pos) {
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    mWorldPos = Pos;
}

Eigen::Vector3f MapPoint::GetWorldPos() {
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos;
}

Eigen::Vector3f MapPoint::GetNormal() {
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector;
}


KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void MapPoint::AddObservation(KeyFrame* pKF, int idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    tuple<int,int> indexes;

    if(mObservations.count(pKF)){
        indexes = mObservations[pKF];
    }
    else{
        indexes = tuple<int,int>(-1,-1);
    }

    if(pKF -> NLeft != -1 && idx >= pKF -> NLeft){
        get<1>(indexes) = idx;
    }
    else{
        get<0>(indexes) = idx;
    }

    mObservations[pKF]=indexes;

    if(!pKF->mpCamera2 && pKF->mvuRight[idx]>=0)
        nObs+=2;
    else
        nObs++;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            tuple<int,int> indexes = mObservations[pKF];
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if(leftIndex != -1){
                if(!pKF->mpCamera2 && pKF->mvuRight[leftIndex]>=0)
                    nObs-=2;
                else
                    nObs--;
            }
            if(rightIndex != -1){
                nObs--;
            }

            mObservations.erase(pKF);

            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}


std::map<KeyFrame*, std::tuple<int,int>>  MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

void MapPoint::SetBadFlag()
{
    map<KeyFrame*, tuple<int,int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        mObservations.clear();
    }
    for(map<KeyFrame*, tuple<int,int>>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        int leftIndex = get<0>(mit -> second), rightIndex = get<1>(mit -> second);
        if(leftIndex != -1){
            pKF->EraseMapPointMatch(leftIndex);
        }
        if(rightIndex != -1){
            pKF->EraseMapPointMatch(rightIndex);
        }
    }

    mpMap->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,tuple<int,int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    for(map<KeyFrame*,tuple<int,int>>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        tuple<int,int> indexes = mit -> second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        if(!pMP->IsInKeyFrame(pKF))
        {
            if(leftIndex != -1){
                pKF->ReplaceMapPointMatch(leftIndex, pMP);
                pMP->AddObservation(pKF,leftIndex);
            }
            if(rightIndex != -1){
                pKF->ReplaceMapPointMatch(rightIndex, pMP);
                pMP->AddObservation(pKF,rightIndex);
            }
        }
        else
        {
            if(leftIndex != -1){
                pKF->EraseMapPointMatch(leftIndex);
            }
            if(rightIndex != -1){
                pKF->EraseMapPointMatch(rightIndex);
            }
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad()
{
    unique_lock<mutex> lock1(mMutexFeatures,std::defer_lock);
    unique_lock<mutex> lock2(mMutexPos,std::defer_lock);
    lock(lock1, lock2);

    return mbBad;
}

void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

/**
 * B.B
 * Computes a distinctive descriptor for a map point by analyzing the descriptors of its observations 
 * and selecting the one that is least similar to the others.
*/
void MapPoint::ComputeDistinctiveDescriptors() {
    // Retrieve all observed descriptors
    // B.B an empty vector to store the descriptors of all observations of this map point.
    vector<cv::Mat> vDescriptors;

    /**
     * B.B
     * A std::map is created to store information about which keyframes have observed this map point. 
     * The map associates each keyframe with a tuple containing two integers: (leftIndex, rightIndex). 
     * These indexes represent the indices of the feature points in the left and right camera images (for stereo vision).
    */
    map<KeyFrame*, tuple<int,int>> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations = mObservations;
    }

    // B.B If there are no observations of this map point, the function returns early because there are no descriptors to compute.
    if(observations.empty())
        return;

    // B.B to optimize memory allocation.
    vDescriptors.reserve(observations.size());

    // B.B iterates through the observations.
    for(map<KeyFrame*, tuple<int,int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {

        KeyFrame* pKF = mit->first;

        // B.B checks if the keyframe associated with the observation is not marked as bad
        if(!pKF->isBad()) {
            // B.B retrieves the left and right indexes from the observation tuple. 
            tuple<int, int> indexes = mit -> second;
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            // B.B collects descriptors from both left and right observations of the map point.
            // B.B If a valid left index exists, it adds the descriptor associated with the left index from the keyframe's descriptor matrix 
            // B.B to the vDescriptors vector.

            if(leftIndex != -1) {
                vDescriptors.push_back(pKF->mDescriptors.row(leftIndex));
            }
            if(rightIndex != -1) {
                vDescriptors.push_back(pKF->mDescriptors.row(rightIndex));
            }
        }
    }

    // B.B no valid descriptors to compute
    if(vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    // B.B a 2D array to store pairwise distances between descriptors. 
    float Distances[N][N];
    for(size_t i = 0; i < N; i++) {
        Distances[i][i] = 0;
        for(size_t j = i + 1; j < N; j++) {

            // B.B computes the distances between all pairs of descriptors
            // Added by Banafshe Bamdad
            #ifdef USE_SELM_EXTRACTOR
                int distij = BBLGMatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            #else
                int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]); // B.B !!! ACHTUNG ACHTUNG !!!
            #endif
            
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    // B.B to find a distinctive descriptor that is different from the others.
    // B.B The distance represents how similar or dissimilar two descriptors are.
    int BestMedian = INT_MAX;
    int BestIdx = 0;

    for(size_t i = 0; i < N; i++) {
        vector<int> vDists(Distances[i], Distances[i] + N);
        sort(vDists.begin(), vDists.end());

        int median = vDists[0.5 * (N - 1)];

        if(median < BestMedian) {
            BestMedian = median;
            BestIdx = i; // B.B The selected descriptor index 
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);

        // B.B is updated with a clone of the selected descriptor.

        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

tuple<int,int> MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return tuple<int,int>(-1,-1);
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

/**
 * B.B
 * computes the normal vector and depth information (mfMinDistance) and mfMaxDistance) distances from the map point to the camera center) for a map point based on its observations in various keyframes
 * Octave levels are often used in scale-invariant feature detectors and descriptors to represent the scale at which a feature is detected.
*/
void MapPoint::UpdateNormalAndDepth() {

    /**
     * B.B
     * to store keyframes and corresponding indexes. 
     * The keyframes are associated with the map point, and the indexes refer to feature indexes in the keyframes.
    */
    std::map<KeyFrame*, tuple<int,int>> observations;

    // B.B reference keyframe.
    KeyFrame* pRefKF;

    // to store the world position of the map point
    Eigen::Vector3f Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);

        // B.B no further updates will be performed for this point if it's considered bad
        if(mbBad) {
            return;
        }

        observations = mObservations;
        pRefKF = mpRefKF;
        Pos = mWorldPos;

    }

    // B.B not compute normals and depths for map points with no observations.
    if(observations.empty()) {
        return;
    }

    Eigen::Vector3f normal;
    normal.setZero();
    int n = 0;

    // B.B iterates over the observations map, which contains keyframes and their associated indexes for the map point.
    for(map<KeyFrame*, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {

        // B.B retrieves the keyframe from the current observation.
        KeyFrame* pKF = mit->first;

        // B.B retrieves the indexes associated with the keyframe observation.
        tuple<int,int> indexes = mit -> second;

        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        // B.B check if valid left camera indexes are available in the observations
        if(leftIndex != -1) {

            Eigen::Vector3f Owi = pKF->GetCameraCenter();

            // B.B calculates the normal vector based on the difference between the world position (Pos) and the camera centers.
            /**
             * B.B Pos: the world position of the map point
             * Owi: the camera center of the keyframe
             * normali: the normal vector from the map point to the camera center.
            */
            Eigen::Vector3f normali = Pos - Owi;

            // B.B computed normals are accumulated
            // B.B normal vector is normalized (divided by its magnitude) before being added to the accumulated normal
            // B.B Vector normalization (unit vector creation): the process of changing a vector's magnitude (length) to a value of 1 while preserving its direction. 
            // B.B ∥v∥: Euclidean norm or magnitude of the vector, the square root of the sum of the squares of vector's components.
            normal = normal + normali / normali.norm();
            n++;
        }

        // B.B check if valid right camera indexes are available in the observations
        if(rightIndex != -1){
            Eigen::Vector3f Owi = pKF->GetRightCameraCenter();
            Eigen::Vector3f normali = Pos - Owi;
            normal = normal + normali / normali.norm();
            n++;
        }
    }

    // B.B calculates the distance between the map point's world position and the camera center of the reference keyframe
    Eigen::Vector3f PC = Pos - pRefKF->GetCameraCenter();
    const float dist = PC.norm();

    // B.B calculates the octave level and levelScaleFactor based on the reference keyframe's attributes and observation indexes. 
    // B.B The octave level is used to compute the scale factor.
    tuple<int ,int> indexes = observations[pRefKF];
    int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    int level;
    if(pRefKF -> NLeft == -1) {
        level = pRefKF->mvKeysUn[leftIndex].octave;
    } else if(leftIndex != -1) {
        level = pRefKF -> mvKeys[leftIndex].octave;
    } else {
        level = pRefKF -> mvKeysRight[rightIndex - pRefKF -> NLeft].octave;
    }

    //const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    // B.B used to relate the scale of the MapPoint with the scale of the reference keyframe.

    const float levelScaleFactor =  pRefKF->mvScaleFactors[level]; // B.B the !!! ACHTUNG ACHTUNG !!! WICKED CUPLRIT !!!
    const int nLevels = pRefKF->mnScaleLevels;

    // B.B to ensure thread safety while updating the map point's attributes.
    {
        unique_lock<mutex> lock3(mMutexPos);

        /**
         * B.B 
         * The mfMaxDistance and mfMinDistance are computed based on the calculated distance and scale factors. 
         * These values represent the maximum and minimum distances from the MapPoint to the camera center, taking into account the scale level.
        */
        // B.B computes the maximum distance from the map point to the reference keyframe's camera center
        mfMaxDistance = dist * levelScaleFactor;

        //  B.B computes the minimum distance based on the maximum distance and the scale factor of the lowest scale level.
        mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1]; // B.B the scale factor of the lowest scale level. nLevels - 1: the last level in pyramid

        // B.B computes the final normal vector by dividing the accumulated normal vector by the number of valid observations n
        mNormalVector = normal / n;
    }
}

void MapPoint::SetNormalVector(const Eigen::Vector3f& normal)
{
    unique_lock<mutex> lock3(mMutexPos);
    mNormalVector = normal;
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f * mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f * mfMaxDistance;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame* pF) {
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}

void MapPoint::PrintObservations()
{
    cout << "MP_OBS: MP " << mnId << endl;
    for(map<KeyFrame*,tuple<int,int>>::iterator mit=mObservations.begin(), mend=mObservations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKFi = mit->first;
        tuple<int,int> indexes = mit->second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
        cout << "--OBS in KF " << pKFi->mnId << " in map " << pKFi->GetMap()->GetId() << endl;
    }
}

Map* MapPoint::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void MapPoint::UpdateMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void MapPoint::PreSave(set<KeyFrame*>& spKF,set<MapPoint*>& spMP)
{
    mBackupReplacedId = -1;
    if(mpReplaced && spMP.find(mpReplaced) != spMP.end())
        mBackupReplacedId = mpReplaced->mnId;

    mBackupObservationsId1.clear();
    mBackupObservationsId2.clear();
    // Save the id and position in each KF who view it
    for(std::map<KeyFrame*,std::tuple<int,int> >::const_iterator it = mObservations.begin(), end = mObservations.end(); it != end; ++it)
    {
        KeyFrame* pKFi = it->first;
        if(spKF.find(pKFi) != spKF.end())
        {
            mBackupObservationsId1[it->first->mnId] = get<0>(it->second);
            mBackupObservationsId2[it->first->mnId] = get<1>(it->second);
        }
        else
        {
            EraseObservation(pKFi);
        }
    }

    // Save the id of the reference KF
    if(spKF.find(mpRefKF) != spKF.end())
    {
        mBackupRefKFId = mpRefKF->mnId;
    }
}

void MapPoint::PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid)
{
    mpRefKF = mpKFid[mBackupRefKFId];
    if(!mpRefKF)
    {
        cout << "ERROR: MP without KF reference " << mBackupRefKFId << "; Num obs: " << nObs << endl;
    }
    mpReplaced = static_cast<MapPoint*>(NULL);
    if(mBackupReplacedId>=0)
    {
        map<long unsigned int, MapPoint*>::iterator it = mpMPid.find(mBackupReplacedId);
        if (it != mpMPid.end())
            mpReplaced = it->second;
    }

    mObservations.clear();

    for(map<long unsigned int, int>::const_iterator it = mBackupObservationsId1.begin(), end = mBackupObservationsId1.end(); it != end; ++it)
    {
        KeyFrame* pKFi = mpKFid[it->first];
        map<long unsigned int, int>::const_iterator it2 = mBackupObservationsId2.find(it->first);
        std::tuple<int, int> indexes = tuple<int,int>(it->second,it2->second);
        if(pKFi)
        {
           mObservations[pKFi] = indexes;
        }
    }

    mBackupObservationsId1.clear();
    mBackupObservationsId2.clear();
}

} //namespace ORB_SLAM
