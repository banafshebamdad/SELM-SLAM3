/*
 * Author: Banafshe Bamdad
 * Created on Mon Oct 16 2023 11:25:15 CET
 *
 */
#ifndef BBSUPERPOINT_HPP
#define BBSUPERPOINT_HPP

#include "opencv2/core.hpp"

#include <vector>
#include <string>

namespace SELMSLAM {

    class BBSuperPoint {
        public:
            BBSuperPoint(std::string modelPath);
            void featureExtractor(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray& descriptors);
        protected:
            std::string m_modelPath;
            std::vector<float> preprocessImage(const cv::Mat& image, float& mean, float& std);
    };
} // namespace

#endif