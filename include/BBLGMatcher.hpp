/*
 * Author: Banafshe Bamdad
 * Created on Sat Oct 28 2023 16:47:06 CET
 *
 */

#ifndef BBLGMATCHER_HPP
#define BBLGMATCHER_HPP

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

using namespace ORB_SLAM3;

namespace SELMSLAM {
    class BBLGMatcher {
    public:
        BBLGMatcher();
        ~BBLGMatcher();
        static float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
        int SearchByLG(ORB_SLAM3::KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
        int MatchLastAndCurrentFrame(Frame lastFrame, Frame &F);
        int SearchByProjection(ORB_SLAM3::Frame &F, const vector<MapPoint*> &vpMapPoints, const bool bFarPoints, const float thFarPoints);
        int TrackLastFrameMapPoints(ORB_SLAM3::Frame &CurrentFrame, const Frame &LastFrame);
    };
    
}

#endif