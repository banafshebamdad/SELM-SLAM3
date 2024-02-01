/*
 * Author: Banafshe Bamdad
 * Created on Mon Oct 16 2023 10:52:26 CET
 *
 */

#include <iostream>

#include "BBSPExtractor.hpp"
#include "BBSuperPoint.hpp"

#ifndef BBSUPERPOINT_WEIGHT_PATH
    #define BBSUPERPOINT_WEIGHT_PATH "/Weights/BBPretrained_Models/superpoint.onnx"
#endif

namespace SELMSLAM {

    using namespace cv;
    using namespace std;

    BBSPExtractor::BBSPExtractor() {
        cout << endl << "B.B is in BBSPExtractor constructor. Press Enter ...";
        // cin.get();
    }

    int BBSPExtractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                                  OutputArray _descriptors, std::vector<int> &vLappingArea) {

        int monoIndex = 0;

        if(_image.empty()) {
            cout << "B.B In BBSPExtractor::operator. _image is empty. Press Enter ...";
            // cin.get();
            return -1; 
        }

        Mat image = _image.getMat();

        // checks whether the image is of type 8-bit unsigned single-channel. 
        // If it's not, it will raise an exception, indicating that the image doesn't have the expected data type.
        assert(image.type() == CV_8UC1);

        BBSuperPoint sp(BBSUPERPOINT_WEIGHT_PATH); // Culprit
        sp.featureExtractor(_image, cv::Mat(), _keypoints, _descriptors);

        return monoIndex; // @todo
    }
}