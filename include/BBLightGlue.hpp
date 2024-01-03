/*
 * Author: Banafshe Bamdad
 * Created on Tue Oct 10 2023 08:41:16 CET
 *
 */

// "include/header guards" technique to prevent multiple inclusions of the same header file
// If Not Defined preprocessor directive checks if BBLIGHTGLUE_HPP identifier has not been defined previously in the code.
#ifndef BBLIGHTGLUE_HPP
#define BBLIGHTGLUE_HPP

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include <string>

using namespace ORB_SLAM3;

namespace SELMSLAM {

    struct CudaMemoryDeleter {
        explicit CudaMemoryDeleter(Ort::Allocator* alloc) {
            alloc_ = alloc;
        }

        void operator()(void* ptr) const {
            alloc_->Free(ptr);
        }
            
        Ort::Allocator* alloc_;
    };

    /** Structure containing image keypoints and descriptors. */
    struct ImageFeatures {
        long unsigned int img_idx; // frame.nmId
        cv::Size img_size;
        std::vector<cv::KeyPoint> keypoints;
        cv::UMat descriptors;
    };

    struct MatchesInfo {
        // Default constructor
        MatchesInfo();

        // // copy constructor
        // MatchesInfo(const MatchesInfo &other);

        // // assignment operator overloading allows to assign the values from one MatchesInfo object to another using the assignment operator (=).
        // const MatchesInfo& operator =(const MatchesInfo &other);

        long unsigned int src_img_idx = 0; // keyframe.nmId
        long unsigned int dst_img_idx = 0;// frame.nmId

        int64_t* match1 = nullptr;
        int64_t* match2 = nullptr;

        float* mscore1 = nullptr;
        float* mscore2 = nullptr;

        std::vector<int64_t> match1shape;
        std::vector<int64_t> match2shape;

        int match1counts = 0;
        int match2counts = 0;

        std::vector<int64_t> mscoreshape1;
        std::vector<int64_t> mscoreshape2;

        int mscore1count = 0;
        int mscore2count = 0;

        std::vector<cv::DMatch> matches; // stores information about matched keypoint descriptors, such as their indices in the source and destination images.
        // std::vector<uchar> inliers_mask; // mask to identify geometrically consistent matches. It can be used to filter out unreliable matches based on some criteria.
        // int num_inliers; // the number of geometrically consistent matches. This is likely computed based on the inliers mask.
        // cv::Mat H; // an estimated transformation, such as a homography matrix, between the source and destination images.
    };

    // I cannot inherit from any class (e.g. ORBMatcher class) in ORB-SLAM3 because it is not designed to support class inheritence.
    class BBLightGlue { // : inherit from ORBMatcher??? !!! ACHTUNG !!!
        public:
            BBLightGlue();
            BBLightGlue(std::string modelPath, float matchThresh);
            static void CallbackCheckMapPoint(MapPoint* pMP, Frame& F, const int i);
            static void CallbackCheckObservations(MapPoint* pMP, Frame& F, const int i);
            int match(KeyFrame* pKF, Frame &F, vector<MapPoint*> &vpMapPointMatches);
            int match(Frame lastFrame, Frame &F);
            int match(const vector<MapPoint*> &vpMapPoints, cv::Mat MPDescriptors, std::vector<cv::KeyPoint> vMPKeypoint, Frame &F, std::function<void(MapPoint*, Frame&, const int i)> callback);
            void perform_match(ImageFeatures features1, ImageFeatures features2, SELMSLAM::MatchesInfo &matches_info);
        protected:
            std::string m_modelPath;
            float m_matchThresh = 0.0;
    };
} // namespace

#endif