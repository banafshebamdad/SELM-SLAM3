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

#include "Frame.h"
#include "Defs.h"
#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"

#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>

// B.B
using namespace SELMSLAM;

namespace ORB_SLAM3
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
{
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
}


//Copy Constructor
// B.B This constructor used in Tracking::StereoInitialization(), when mLastFrame = Frame(mCurrentFrame)
// B.B the two arguments added by Banafshe Bamdad
// B.B mpBBSPExtractorLeft(frame.mpBBSPExtractorLeft), mpBBSPExtractorRight(frame.mpBBSPExtractorRight),
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi), mpORBvocabulary(frame.mpORBvocabulary), 
     mpBBSPExtractorLeft(frame.mpBBSPExtractorLeft), mpBBSPExtractorRight(frame.mpBBSPExtractorRight),
     mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame),
     mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl),
     mTcw(frame.mTcw), mbHasPose(false), mbHasVelocity(false)
{

    // cout << endl << "B.B In Frame copy constructor. press Enter...";
    // cin.get();

    for(int i = 0; i < FRAME_GRID_COLS; i++)
        for(int j = 0; j < FRAME_GRID_ROWS; j++){
            mGrid[i][j] = frame.mGrid[i][j];
            if(frame.Nleft > 0){
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if(frame.mbHasPose)
        SetPose(frame.GetPose());

    if(frame.HasVelocity()) {
        SetVelocity(frame.GetVelocity());
    }

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;

#ifdef REGISTER_TIMES
    mTimeStereoMatch = frame.mTimeStereoMatch;
    mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,0,0);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mmProjectPoints.clear();
    mmMatchedInImage.clear();


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);



        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}

// B.B Achtung
// B.B Verifie-toi pourquoi la valeur de mCurrentFrame.N=0.
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId = nNextId++;

    cout << endl << "B.B in Frame::Frame, Frame ID=" << mnId;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // cout << endl << "\tmnScaleLevels=" << mnScaleLevels;
    // cout << endl << "\tmfScaleFactor=" << mfScaleFactor;
    // cout << endl << "\tmfLogScaleFactor=" << mfLogScaleFactor;
    // cout << endl << "\tmvScaleFactors=" << mvScaleFactors;
    // cout << endl << "\tmvInvScaleFactors=" << mvInvScaleFactors;
    // cout << endl << "\tmvLevelSigma2=" << mvLevelSigma2;
    // cout << endl << "\tmvInvLevelSigma2=" << mvInvLevelSigma2;
    // cin.get();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0, imGray , 0 ,0);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();

    // cout << endl << "B.B In Frame::Frame, after extracting ORB features. mvKeys=" << N << endl;

    if(mvKeys.empty()) {
        // cout << "B.B In Frame::Frame, mvKeys is empty..." << endl; 
        return;
    }

    // cout << endl << "B.B In Frame::Frame, before (1) UndistortKeyPoints() " << endl;
    // B.B to undistort the keypoints using the camera distortion model.
    UndistortKeyPoints();

    // cout << endl << "B.B In Frame::Frame, before (2) ComputeStereoFromRGBD() " << endl;
    ComputeStereoFromRGBD(imDepth);

    // B.B initializes a vector named mvpMapPoints with a size of N, and each element of this vector is set to NULL, a null pointer of type MapPoint*.
    mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N, false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations) {
        ComputeImageBounds(imGray);

        /**
         * B.B
         * FRAME_GRID_COLS: the number of columns in a grid structure.
         * mnMaxX and mnMinX: the maximum and minimum x-coordinates of keypoints in the current frame.
         * calculates the inverse of the width of a grid element along the x-axis. 
         * It's used to map keypoints to grid cells efficiently. 
         * The smaller this value, the larger the grid element size, which means keypoints will be grouped into larger grid cells.
        */
        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

        // B.B calculates the inverse of the focal length
        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    mb = mbf / fx;

    if(pPrevF) {
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    } else {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    // cout << endl << "B.B In Frame::Frame, before (3) AssignFeaturesToGrid() " << endl;
    // B.B organizes keypoint features into a grid structure for efficient feature matching and tracking
    // B.B !!! ACHTUNG ACHTUNG !!!
    AssignFeaturesToGrid();
}

// Added by Banafshe Bamdad

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, SELMSLAM::BBSPExtractor* extractor, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc), mpBBSPExtractorLeft(extractor), mpBBSPExtractorRight(static_cast<BBSPExtractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera), mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId = nNextId++;

    // Scale Level Info
    // B.B Since I want to remove the following information, I initialized them with constant values.
    std::vector<float> myVector(1, 1.0f);
    mnScaleLevels = 1;
    mfScaleFactor = 1.0;
    mfLogScaleFactor = 0.0;
    mvScaleFactors = myVector;
    mvInvScaleFactors = myVector;
    mvLevelSigma2 = myVector;
    mvInvLevelSigma2 = myVector;

    // SuperPoint extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif

    // B.B
    BBExtractSP(0, imGray, 0, 0);

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();

    cout << endl << "B.B In Frame::Frame, after extracting ORB/SP features. mvKeys size=" << N << endl;

    if(mvKeys.empty()) {
        return;
    }

    // cout << endl << "B.B In Frame::Frame, before (1) UndistortKeyPoints() " << endl;
    // B.B to undistort the keypoints using the camera distortion model.
    UndistortKeyPoints();

    // cout << endl << "B.B In Frame::Frame, before (2) ComputeStereoFromRGBD() " << endl;
    ComputeStereoFromRGBD(imDepth);

    // B.B initializes a vector named mvpMapPoints with a size of N, and each element of this vector is set to NULL, a null pointer of type MapPoint*.
    mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N, false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations) {
        ComputeImageBounds(imGray);

        /**
         * B.B
         * FRAME_GRID_COLS: the number of columns in a grid structure.
         * mnMaxX and mnMinX: the maximum and minimum x-coordinates of keypoints in the current frame.
         * calculates the inverse of the width of a grid element along the x-axis. 
         * It's used to map keypoints to grid cells efficiently. 
         * The smaller this value, the larger the grid element size, which means keypoints will be grouped into larger grid cells.
        */
        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

        // B.B calculates the inverse of the focal length
        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    mb = mbf / fx;

    if(pPrevF) {
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    } else {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    // cout << endl << "B.B In Frame::Frame, before (3) AssignFeaturesToGrid() " << endl;
    // B.B organizes keypoint features into a grid structure for efficient feature matching and tracking
    // B.B !!! ACHTUNG ACHTUNG !!!
    AssignFeaturesToGrid();
}
// B.B jusqu'ici

Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,1000);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }


    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();
}


/**
 * B.B
 * organizes keypoints into a grid structure based on their positions in the image. 
 * This grid-based organization is used to accelerate feature matching and tracking by limiting the search for matching features to specific grid cells, 
 * reducing the computational complexity of these operations.
*/
void Frame::AssignFeaturesToGrid() {
    // Fill matrix with points

    // B.B Calculates the total number of cells in the grid
    const int nCells = FRAME_GRID_COLS * FRAME_GRID_ROWS;

    // B.B Calculates the number of features to reserve for each grid cell. 
    // B.B It reserves space to accommodate about half of the total features (N) evenly distributed across the grid cells.
    int nReserve = 0.5f * N / (nCells);

    for(unsigned int i = 0; i < FRAME_GRID_COLS; i++)
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++){

            // B.B Reserves space for the specified number of features (nReserve) in the current grid cell at position (i, j) in the left image.
            mGrid[i][j].reserve(nReserve);

            // B.B checks if there is a right image
            if(Nleft != -1){

                // B.B reserves space in the right image grid
                mGridRight[i][j].reserve(nReserve);
            }
        }


    // B.B loops over all the keypoints (N) and determines which grid cell each keypoint belongs to.
    for(int i = 0; i< N; i++) {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];

        // B.B Calculates the grid position (nGridPosX, nGridPosY) of the current keypoint based on its location in the image.
        int nGridPosX, nGridPosY;

        // B.B Checks if the keypoint's position is within the defined grid. 
        // B.B determines whether the keypoint's image coordinates fall within the boundaries of the grid.
        if(PosInGrid(kp, nGridPosX, nGridPosY)) {

            // B.B If the keypoint belongs to the left image, adds the index i (the feature) to the corresponding cell in the left image grid (mGrid).
            if(Nleft == -1 || i < Nleft)
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1) {
    vector<int> vLapping = {x0,x1};
    if(flag == 0) {
        // B.B mpORBextractorLeft: a pointer to an object
        // B.B (*mpORBextractorLeft): accessing the object that the pointer is pointing to call a method or access a member of that object.
        // B.B ORBextractor* mpORBextractorLeft
        /**
         * B.B
         * In C++, objects of a class can overload certain operators to define custom behaviors for those operators when they are used with instances of that class. 
         * One of the operators that can be overloaded is the () operator (parentheses).
         * When an object overloads the () operator, it essentially allows you to treat instances of that class as if they were functions. 
         * In other words, you can call an object like a function by using the parentheses () syntax, passing arguments inside those parentheses.
        */
        monoLeft = (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors, vLapping);
        cout << "SPT: desc size: " << mDescriptors.rows << ' ' << mDescriptors.cols << endl; // B.B = 0
        cout << "SPT: ExtractORB mvKeys size: " << mvKeys.size() << endl; // B.B = 0

        // B.B Print two elements from the vector of mvKeys
        for (int i = 0; i < std::min(2, static_cast<int>(mvKeys.size())); i++) {
            cv::KeyPoint kp = mvKeys[i];
            // std::cout << "B.B ORB KeyPoint " << i << ": (" << kp.pt.x << ", " << kp.pt.y << "), size: " << kp.size << std::endl;
        }
        
        // cout << "B.B Press Enter ...";
        // cin.get();

        // int type = mDescriptors.type();
        // if (type == CV_8U) {
        //     std::cout << "B.B The elements in the ORB mDescriptors Mat are 8-bit unsigned integers (binary data)." << std::endl;
        // } else {
        //     std::cout << "The elements in the ORB mDescriptors Mat ARE NOT binary." << std::endl;
        // }

        // for (int i = 0; i < std::min(2, mDescriptors.rows); i++) {
        //     for (int j = 0; j < std::min(2, mDescriptors.cols); j++) {
        //         std::cout << "B.B ORB: Value at (" << i << ", " << j << "): " << mDescriptors.at<float>(i, j) << std::endl;
        //     }
        // }

        // cout << "B.B Press Enter ...";
        // cin.get();
    } else
        monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
}

// Added by Banafshe bamdad
void Frame::BBExtractSP(int flag, const cv::Mat &im, const int x0, const int x1) {

    vector<int> vLapping = {x0, x1};
    if(flag == 0) {

        monoLeft = (*mpBBSPExtractorLeft)(im, cv::Mat(), mvKeys, mDescriptors, vLapping);

        // cout << "B.B Desc size: (" << mDescriptors.rows << ", " << mDescriptors.cols << ")" << endl; 
        // cout << "B.B BBExtractSP mvKeys size: " << mvKeys.size() << endl;

        // B.B Print two elements from the vector of mvKeys
        // for (int i = 0; i < std::min(5, static_cast<int>(mvKeys.size())); i++) {
        //     cv::KeyPoint kp = mvKeys[i];
        //     std::cout << "B.B SP KeyPoint " << i << ": (" << kp.pt.x << ", " << kp.pt.y << "), size: " << kp.size << std::endl;
        // }
        
        // int type = mDescriptors.type();
        // if (type == CV_8U) {
        //     std::cout << "B.B The elements in the SP mDescriptors Mat are 8-bit unsigned integers (binary data)." << std::endl;
        // } else {
        //     std::cout << "The elements in the SP mDescriptors Mat ARE NOT binary." << std::endl;
        // }

        // for (int i = 0; i < std::min(5, mDescriptors.rows); i++) {
        //     for (int j = 0; j < std::min(5, mDescriptors.cols); j++) {
        //         std::cout << "B.B SP: Value at (" << i << ", " << j << "): " << mDescriptors.at<float>(i, j) << std::endl;
        //     }
        // }

    } else
        monoRight = (*mpBBSPExtractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight, vLapping);
}
// B.B

bool Frame::isSet() const {
    return mbIsSet;
}

void Frame::SetPose(const Sophus::SE3<float> &Tcw) {
    mTcw = Tcw;

    // std::cout << "------------" << "\n"  << Tcw.matrix() << "\n" << "------------" << std::endl;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void Frame::SetVelocity(Eigen::Vector3f Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const
{
    return mVw;
}

void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::UpdatePoseMatrices()
{
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc = Twc.rotationMatrix();
    mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix();
    mtcw = mTcw.translation();
}

Eigen::Matrix<float,3,1> Frame::GetImuPosition() const {
    return mRwc * mImuCalib.mTcb.translation() + mOw;
}

Eigen::Matrix<float,3,3> Frame::GetImuRotation() {
    return mRwc * mImuCalib.mTcb.rotationMatrix();
}

Sophus::SE3<float> Frame::GetImuPose() {
    return mTcw.inverse() * mImuCalib.mTcb;
}

Sophus::SE3f Frame::GetRelativePoseTrl()
{
    return mTrl;
}

Sophus::SE3f Frame::GetRelativePoseTlr()
{
    return mTlr;
}

Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation(){
    return mTlr.rotationMatrix();
}

Eigen::Vector3f Frame::GetRelativePoseTlr_translation() {
    return mTlr.translation();
}


/**
 * B.B
 * The frustum is the portion of space in front of the camera that is captured in the image.
*/
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
    if(Nleft == -1) { // B.B the frame is not part of a stereo pair 
        pMP->mbTrackInView = false; // B.B the MapPoint is not visible in the frame.
        pMP->mTrackProjX = -1; // B.B sets the x-coordinate of the MapPoint's projection in the image
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        Eigen::Matrix<float,3,1> P = pMP->GetWorldPos(); // B.B the 3D position of the MapPoint in the world

        // 3D in camera coordinates
        // B.B converts the MapPoint's 3D position from the world coordinate system to the camera coordinate system using the current camera pose (mRcw and mtcw)
        const Eigen::Matrix<float,3,1> Pc = mRcw * P + mtcw;
        const float Pc_dist = Pc.norm(); // B.B the distance of the MapPoint from the camera center

        // Check positive depth
        const float &PcZ = Pc(2);
        const float invz = 1.0f / PcZ;
        if(PcZ < 0.0f) { // B.B the MapPoint is behind the camera and is not visible
            return false;
        }

        // B.B projects the MapPoint's 3D position in the camera coordinate system onto the image plane using the camera's intrinsic parameters
        const Eigen::Vector2f uv = mpCamera->project(Pc);

        if(uv(0) < mnMinX || uv(0) > mnMaxX) {
            return false;
        }
        if(uv(1) < mnMinY || uv(1) > mnMaxY) {
            return false;
        }

        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);

        // Check distance is in the scale invariance region of the MapPoint
        /**
         * B.B
         * The GetMaxDistanceInvariance() helps to ensure that only visible MapPoints are used in the tracking process. 
         * This helps to improve the accuracy and robustness of the tracker, particularly in challenging environments 
         * where the visibility of MapPoints can vary significantly.
        */
        // B.B the maximum distance at which the MapPoint can still be reliably detected in the image under certain conditions.
        // B.B A higher maximum distance invariance indicates that the MapPoint can be detected at greater distances, while a lower value means that it is only visible at shorter distances.
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();

        // B.B the vector from the MapPoint's position to the camera center
        const Eigen::Vector3f PO = P - mOw; // B.B mOw is the coordinate of camera center in the World coordinate system

        // B.B the distance between the MapPoint and the camera center (the diatance between P and camera center in world coordinate system)
        const float dist = PO.norm();

        // B.B If the distance is less than the maximum distance invariance, then the MapPoint is considered to be visible in the frame.
        if(dist < minDistance || dist > maxDistance)
            return false;

        // Check viewing angle
        Eigen::Vector3f Pn = pMP->GetNormal();

        /**
         * B.B
         * ??? the cosinus of the angle between the vector from the MapPoint's position to the camera center (PO) and the MapPoint's normal vector (Pn). 
         * The result is divided by the distance between the MapPoint and the camera center to normalize the angle.
        */
        const float viewCos = PO.dot(Pn) / dist;

        if(viewCos < viewingCosLimit)
            return false;

        // B.B the scale level of a MapPoint in a frame is a measure of how close the MapPoint is to the camera. 
        // B.B the scale level of the MapPoint in the current frame based on its distance from the camera center and the camera's intrinsic parameters (this).
        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pMP->mbTrackInView = true; // B.B the MapPoint is visible in the frame
        pMP->mTrackProjX = uv(0); // B.B x-coordinate of the MapPoints projection in the image

        /**
         * B.B
         * used by the tracker to refine the camera pose and improve the accuracy of the tracking results.
         * the inverse depth of a MapPoint can provide additional information about the relative depth of the MapPoint and the camera.
         * the mTrackProjXR variable can provide a more accurate representation of the MapPoint's location in the image plane. 
         * This can help to improve the accuracy of the camera pose estimation and the tracking results overall.
        */
        pMP->mTrackProjXR = uv(0) - mbf * invz;

        pMP->mTrackDepth = Pc_dist; // B.B the distance of the MapPoint from the camera center

        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel = nPredictedLevel;

        // B.B the cosinus of the angle between the vector from the MapPoint's position to the camera center (PO) and the MapPoint's normal vector (Pn), normalized by the distance between the MapPoint and the camera center
        pMP->mTrackViewCos = viewCos;

        return true;
    } else { // B.B the frame is part of a stereo pair 
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP -> mnTrackScaleLevel = -1;
        pMP -> mnTrackScaleLevelR = -1;

        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mRcw * P + mtcw;
    const float &PcX = Pc(0);
    const float &PcY= Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw)
{
    return mRcw * pCw + mtcw;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW() {
    // B.B prevents redundant computations, ensuring that the BoW representation is computed only if it hasn't been computed already for this frame.
    if(mBowVec.empty()) {

        // cout << endl << "B.B BoWVec of the frame is empty. Please go agead and compute them...." << endl;

        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);

        // B.B e.g. vCurrentDesc size: 1009, mDescriptors size: [256 x 1009]
        // cout << endl << "B.B In Frame::ComputeBoW, vCurrentDesc size: " << vCurrentDesc.size() << ", mDescriptors size: " << mDescriptors.size() << endl;

        // B.B computes the Bag of Words (BoW) representation for the frame.
        // B.B DBOW_LEVELS: likely specifies the depth or level of the vocabulary tree to use during the BoW transformation
        cout << endl << "B.B in Frame.cc. Line 1035. Before calling mpORBvocabulary->transform. Press Enter..."; 
        // cin.get();
        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, DBOW_LEVELS); // B.B ACHTUNG ACHTUNG !!!
        // cout << endl << "B.B In Frame::ComputeBoW, vCurrentDesc size: " << vCurrentDesc.size() << ", DBOW_LEVELS: " << DBOW_LEVELS << endl;
        
        // mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,3);
        // printf("%s Frame - Extracted %d BOW FEATURES \n", __PRETTY_FUNCTION__, mBowVec.size());
        // printf("vCurrentDesc %d, mBowVec %d, mFeatVec %d\n",vCurrentDesc.size() ,mBowVec.size() ,mFeatVec.size() );
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);

    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);


    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }

}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches() {

    // cout << endl << "B.B In Frame::ComputeStereoMatches. Press Enter ...";
    // cin.get();
    
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);

                cout << "B.B Line 1187 Frame.cc. before calling ORBmatcher::DescriptorDistance. Press Enter...";
                // cin.get();

                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


/**
 * B.B
 * to compute stereo information (depth and right disparity) for keypoints in a frame using depth data from an RGB-D sensor.
*/
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth) {

    // B.B  initialized with a size of N
    mvuRight = vector<float>(N, -1);
    mvDepth = vector<float>(N, -1);

    for(int i = 0; i < N; i++) {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        // B.B retrieves the v and u coordinates of the current keypoint from the frame.
        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        // B.B the depth of the scene at the location of the keypoint.
        const float d = imDepth.at<float>(v, u);

        // B.B a valid depth measurement was obtained from the RGB-D sensor at that point.
        if(d > 0) {
            // B.B stores the depth d in the mvDepth array
            mvDepth[i] = d;
            // B.B computes the right disparity for the keypoint using the provided baseline mbf (baseline between the stereo cameras) and the depth d.
            mvuRight[i] = kpU.pt.x - mbf / d;
        }
    }
}

/**
 * B.B
 * i: the index of a feature (keypoint) in the frame.
 * x3D: the 3D coordinates of the unprojected point.
*/
bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D) {

    // B.B the distance from the camera to the feature.
    const float z = mvDepth[i];

    // B.B unproject the 2D pixel coordinates (u, v) to 3D world coordinates using stereo vision geometry. 
    if(z > 0) {
        // B.B retrieve the pixel coordinates of the feature
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;

        // B.B calculate the 3D coordinates using the camera intrinsics and the pixel coordinates.
        const float x = (u - cx) * z * invfx;
        const float y = (v - cy) * z * invfy;

        // cout << "B.B In Frame::UnprojectStereo. (u, v)=(" << u << ", " << v << ") , (x, y, z)=(" << x << ", " << y << ", " << z << ")" << endl;

        // B.B the feature's 3D coordinates in the camera coordinate frame.
        Eigen::Vector3f x3Dc(x, y, z);

        // cout << "B.B In Frame::UnprojectStereo after creating x3Dc object in camera frame.";
        // cin.get();

        // B.B transforms the 3D coordinates from the camera frame to the world frame using a rotation matrix mRwc and a translation vector mOw
        x3D = mRwc * x3Dc + mOw;

        // cout << "B.B In Frame::UnprojectStereo after creating x3D object world frame.";
        // cin.get();

        return true;
    } else {
        // cout << "B.B In Frame::UnprojectStereo depth <= 0";
        // cin.get();

        return false;
    }
}

bool Frame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
        :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),  mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2),
         mbHasPose(false), mbHasVelocity(false)

{
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    N = Nleft + Nright;

    if(N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf / fx;

    // Sophus/Eigen
    mTlr = Tlr;
    mTrl = mTlr.inverse();
    mRlr = mTlr.rotationMatrix();
    mtlr = mTlr.translation();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoFishEyeMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    //Put all descriptors in the same matrix
    cv::vconcat(mDescriptors, mDescriptorsRight, mDescriptors);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();

}

void Frame::ComputeStereoFishEyeMatches() {
    //Speed it up by matching keypoints in the lapping area
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch = vector<int>(Nleft,-1);
    mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f);
    mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
    mnCloseMPs = 0;

    //Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

    int nMatches = 0;
    int descMatches = 0;

    //Check matches using Lowe's ratio
    for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
        if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
            //For every good match, check parallax and reprojection error to discard spurious matches
            Eigen::Vector3f p3D;
            descMatches++;
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,mvKeys[(*it)[0].queryIdx + monoLeft],mvKeysRight[(*it)[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
            if(depth > 0.0001f){
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D;
                mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                nMatches++;
            }
        }
    }
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if(bRight){
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
        Eigen::Vector3f trl = mTrl.translation();
        mR = Rrl * mRcw;
        mt = Rrl * mtcw + trl;
        twc = mRwc * mTlr.translation() + mOw;
    }
    else{
        mR = mRcw;
        mt = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mR * P + mt;
    const float Pc_dist = Pc.norm();
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    if(bRight) uv = mpCamera2->project(Pc);
    else uv = mpCamera->project(Pc);

    if(uv(0)<mnMinX || uv(0)>mnMaxX)
        return false;
    if(uv(1)<mnMinY || uv(1)>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const Eigen::Vector3f PO = P - twc;
    const float dist = PO.norm();

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    Eigen::Vector3f Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    if(bRight){
        pMP->mTrackProjXR = uv(0);
        pMP->mTrackProjYR = uv(1);
        pMP->mnTrackScaleLevelR= nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    }
    else{
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    return true;
}

Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i){
    return mRwc * mvStereo3Dpoints[i] + mOw;
}

} //namespace ORB_SLAM
