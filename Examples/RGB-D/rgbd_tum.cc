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

/**
 * $ cd /media/banafshe/Banafshe_2TB/Datasets/TUM
 * $ python associate.py /path/to/rgb.txt /path/to/depth.txt > /path/to/dataset/sequence/associations.txt
*/

// B.B
// Testing and Debugging
//      $ ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM1.yaml /media/banafshe/Banafshe_2TB/Datasets/TUM/Testing_and_Debugging/xyz/rgbd_dataset_freiburg1_xyz /media/banafshe/Banafshe_2TB/Datasets/TUM/Testing_and_Debugging/xyz/rgbd_dataset_freiburg1_xyz/associations.txt
//      $ ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM2.yaml /media/banafshe/Banafshe_2TB/Datasets/TUM/Testing_and_Debugging/xyz/rgbd_dataset_freiburg2_xyz /media/banafshe/Banafshe_2TB/Datasets/TUM/Testing_and_Debugging/xyz/rgbd_dataset_freiburg2_xyz/associations.txt

// Dynamic objects
//  $ ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM2.yaml /media/banafshe/Banafshe_2TB/Datasets/TUM/Dynamic_Objects/rgbd_dataset_freiburg2_desk_with_person /media/banafshe/Banafshe_2TB/Datasets/TUM/Dynamic_Objects/rgbd_dataset_freiburg2_desk_with_person/associations.txt
//  $ ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /media/banafshe/Banafshe_2TB/Datasets/TUM/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz /media/banafshe/Banafshe_2TB/Datasets/TUM/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/associations.txt

// Structure vs. Texture
//      $ ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /media/banafshe/Banafshe_2TB/Datasets/TUM/Structure_vs_Texture/rgbd_dataset_freiburg3_nostructure_notexture_far /media/banafshe/Banafshe_2TB/Datasets/TUM/Structure_vs_Texture/rgbd_dataset_freiburg3_nostructure_notexture_far/associations.txt
//      $ ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /media/banafshe/Banafshe_2TB/Datasets/TUM/Structure_vs_Texture/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop /media/banafshe/Banafshe_2TB/Datasets/TUM/Structure_vs_Texture/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop/associations.txt
//      $ ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /media/banafshe/Banafshe_2TB/Datasets/TUM/Structure_vs_Texture/rgbd_dataset_freiburg3_nostructure_texture_far /media/banafshe/Banafshe_2TB/Datasets/TUM/Structure_vs_Texture/rgbd_dataset_freiburg3_nostructure_texture_far/associations.txt
//      $ ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /media/banafshe/Banafshe_2TB/Datasets/TUM/Structure_vs_Texture/rgbd_dataset_freiburg3_structure_texture_far /media/banafshe/Banafshe_2TB/Datasets/TUM/Structure_vs_Texture/rgbd_dataset_freiburg3_structure_texture_far/associations.txt

// Handheld SLAM
// ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM2.yaml /media/banafshe/Banafshe_2TB/Datasets/TUM/Handheld_SLAM/rgbd_dataset_freiburg2_360_kidnap /media/banafshe/Banafshe_2TB/Datasets/TUM/Handheld_SLAM/rgbd_dataset_freiburg2_360_kidnap/associations.txt

// B.B to test embedding Python to C++
// $ ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /media/banafshe/Banafshe_2TB/Datasets/TUM/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz /media/banafshe/Banafshe_2TB/Datasets/TUM/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/associations.txt python_for_c multiply 2 3

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

// B.B
// #include <Python.h>
// #define PY_SSIZE_T_CLEAN
#include <codecvt>
#include "BBLightGlue.hpp"
#include "BBSuperPoint.hpp"

// Added by banafshe Bamdad
// #define USE_ORBFEATURES 1
using namespace std;
using namespace cv;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

// B.B
// void run_function_defined_in_Python_script(int argc, const char* python_for_c, const char* py_func_name);

// B.B
// std::string onnxSPModelsPath = "/home/banafshe/SUPERSLAM3/Weights/BBPretrained_Models/superpoint.onnx";
// std::string onnxLGModelsPath = "/home/banafshe/SUPERSLAM3/Weights/BBPretrained_Models/superpoint_lightglue.onnx";

float matchThresh = 0.0f;

int main(int argc, char **argv) {
    if(argc < 5) {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    // B.B
    // SELMSLAM::BBLightGlue lg;
    // B.B converts the provided LightGlue model paths from ASCII to wide strings
    // B.B Wide strings use a multi-byte character encoding, often based on the UTF-16 or UTF-32 encoding standards, 
    // B.B which can represent a broader range of characters, including non-ASCII characters from various languages and symbols.
    // std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    // std::wstring lh = converter.from_bytes(lightGluePath);
    // Ptr<BBLightGlue> lightglue = makePtr<BBLightGlue>(lh, matchThresh);
    // SELMSLAM::BBLightGlue lg(onnxLGModelsPath, matchThresh);
    // lg.match();

    // B.B ::: START Pure Python Embedding :::

    // PyObject *pName, *pModule, *pFunc;
    // PyObject *pArgs, *pValue;
    // int i;

    // // initializes the Python interpreter
    // Py_Initialize();

    // // decodes the script name (argv[5]) into a Python Unicode object
    // pName = PyUnicode_DecodeFSDefault(argv[5]);

    // // imports python_for_c
    // pModule = PyImport_Import(pName);

    // // After importing, it decrements the reference count of pName.
    // // it tells the Python interpreter that the C/C++ code is done using pName, and the reference to it can be released. 
    // // n Python, objects are automatically managed by a system called reference counting. Each object has a reference count associated with it, which keeps track of how many references (pointers) exist to that object. When the reference count of an object drops to zero, it means that there are no more references to that object, and it can be safely deallocated
    // Py_DECREF(pName);

    // if (pModule != NULL) {
    //     // retrieves a reference to the Python function from the imported module
    //    pFunc = PyObject_GetAttrString(pModule, argv[6]); 

    //     // If the function exists and is callable, it prepares the function arguments 
    //    if (pFunc && PyCallable_Check(pFunc)) {

    //         int num_of_c_arg = 7;
    //         int num_of_py_func_args = argc - num_of_c_arg;
    //         pArgs = PyTuple_New(num_of_py_func_args);
    //         for (i = 0; i < num_of_py_func_args; ++i) {
    //             pValue = PyLong_FromLong(atoi(argv[i + num_of_c_arg]));
    //             if (!pValue) {
    //                 Py_DECREF(pArgs);
    //                 Py_DECREF(pModule);
    //                 fprintf(stderr, "Cannot convert argument\n");
    //             }
    //             PyTuple_SetItem(pArgs, i, pValue);
    //         }
    //         // invokes the Python function (pFunc) with the prepared arguments (pArgs)
    //         pValue = PyObject_CallObject(pFunc, pArgs);
    //         Py_DECREF(pArgs);

    //         if (pValue != NULL) {
    //                 printf("Result of call: %ld\n", PyLong_AsLong(pValue));
    //                 Py_DECREF(pValue);
    //         } else {
    //             Py_DECREF(pFunc);
    //             Py_DECREF(pModule);
    //             PyErr_Print();
    //             fprintf(stderr,"Call failed\n");
    //         }
    //    } else {
    //         if (PyErr_Occurred())
    //             PyErr_Print();
    //         fprintf(stderr, "Cannot find function \"%s\"\n", argv[6]);
    //     }
    //     Py_XDECREF(pFunc);
    //     Py_DECREF(pModule);
    // } else {
    //     PyErr_Print();
    //     fprintf(stderr, "Failed to load \"%s\"\n", argv[5]);
    // }
    // Py_FinalizeEx();

    // B.B ::: END Pure Python Embedding  :::

    // B.B test test
    // run_function_defined_in_Python_script(argc, argv[5].c_str(), argv[6].c_str());

    // B.B added the following code snippet for testing purpose
    // B.B Very High Level Embedding
    // wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    // if (program == NULL) {
    //     fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
    //     exit(1);
    // }
    // Py_SetProgramName(program);  /* optional but recommended */
    // Py_Initialize();
    // PyRun_SimpleString("from time import time,ctime\n"
    //                    "print('Today is', ctime(time()))\n");
    // if (Py_FinalizeEx() < 0) {
    //     exit(120);
    // }
    // PyMem_RawFree(program);
    // cout << endl << "Press Enter..." << endl;
    // cin.get();
    // B.B jusqu'ici

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty()) {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    } else if(vstrImageFilenamesD.size()!= vstrImageFilenamesRGB.size()) {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "B.B In rgbd_tum, Start processing sequence ..." << endl;
    cout << "B.B In rgbd_tum, Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni = 0; ni < nImages; ni++) {

        cout << endl << "B.B in main class. Image number: " << ni;
        
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f) {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        // B.B @todo Fr. Okt. 20 08:26 process the output of this method which is Sophus::SE3f Tcw
        SLAM.TrackRGBD(imRGB, imD, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
        double T = 0;
        if(ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if(ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if(ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for(int ni = 0; ni < nImages; ni++) {
        totaltime += vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages  /2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");   

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof()) {
        string s;
        getline(fAssociation, s);
        if(!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}

// B.B run a function defined in a Python script
// void run_function_defined_in_Python_script(int argc, const char* python_for_c, const char* py_func_name) {
//     PyObject *pName, *pModule, *pFunc;
//     PyObject *pArgs, *pValue;
//     int i;

//     // initializes the Python interpreter
//     Py_Initialize();

//     // decodes the script name (argv[5]) into a Python Unicode object
//     pName = PyUnicode_DecodeFSDefault(python_for_c);

//     // imports python_for_c
//     pModule = PyImport_Import(pName);

//     // After importing, it decrements the reference count of pName.
//     // it tells the Python interpreter that the C/C++ code is done using pName, and the reference to it can be released. 
//     // n Python, objects are automatically managed by a system called reference counting. Each object has a reference count associated with it, which keeps track of how many references (pointers) exist to that object. When the reference count of an object drops to zero, it means that there are no more references to that object, and it can be safely deallocated
//     Py_DECREF(pName);

//     if (pModule != NULL) {
//         // retrieves a reference to the Python function from the imported module
//        pFunc = PyObject_GetAttrString(pModule, py_func_name); 

//         // If the function exists and is callable, it prepares the function arguments 
//        if (pFunc && PyCallable_Check(pFunc)) {

//             int num_of_c_arg = 7;
//             int num_of_py_func_args = argc - num_of_c_arg;
//             pArgs = PyTuple_New(num_of_py_func_args);
//             for (i = 0; i < num_of_py_func_args; ++i) {
//                 pValue = PyLong_FromLong(atoi(argv[i + num_of_c_arg]));
//                 if (!pValue) {
//                     Py_DECREF(pArgs);
//                     Py_DECREF(pModule);
//                     fprintf(stderr, "Cannot convert argument\n");
//                     return 1;
//                 }
//                 PyTuple_SetItem(pArgs, i, pValue);
//             }
//             // invokes the Python function (pFunc) with the prepared arguments (pArgs)
//             pValue = PyObject_CallObject(pFunc, pArgs);
//             Py_DECREF(pArgs);

//             if (pValue != NULL) {
//                     printf("Result of call: %ld\n", PyLong_AsLong(pValue));
//                     Py_DECREF(pValue);
//             } else {
//                 Py_DECREF(pFunc);
//                 Py_DECREF(pModule);
//                 PyErr_Print();
//                 fprintf(stderr,"Call failed\n");
//                 return 1;
//             }
//        } else {
//             if (PyErr_Occurred())
//                 PyErr_Print();
//             fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
//         }
//         Py_XDECREF(pFunc);
//         Py_DECREF(pModule);
//     } else {
//         PyErr_Print();
//         fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
//         return 1;
//     }
//     if (Py_FinalizeEx() < 0) {
//         return 120;
//     }
// }