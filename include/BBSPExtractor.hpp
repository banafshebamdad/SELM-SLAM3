/*
 * Author: Banafshe Bamdad
 * Created on Mon Oct 16 2023 10:51:46 CET
 *
 */

#ifndef BBSPEXTRACTOR_HPP
#define BBSPEXTRACTOR_HPP

#include "opencv2/core.hpp"

#include <string>

namespace SELMSLAM {

    class BBSPExtractor {
    public:
        BBSPExtractor();
        // @todo Fr Okt 20, do I need these parameters?
        // BBSPExtractor(int nfeatures, float scaleFactor, int nlevels, float iniThFAST, float minThFAST);
        // ~BBSPExtractor();

        int operator()( cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& _keypoints, cv::OutputArray _descriptors, std::vector<int> &vLappingArea);
    }; // class
} // namespace

#endif