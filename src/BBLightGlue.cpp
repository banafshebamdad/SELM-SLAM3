/*
 * Author: Banafshe Bamdad
 * Created on Tue Oct 10 2023 08:59:53 CET
 *
 */

/**
 * "Quotation Marks": for including headers that are part of your project or in the same directory as your source files.
 * the preprocessor first looks for the file in the current directory or the directory where the source file containing the #include directive is located.
 * If the file is not found in the current directory, the preprocessor will search in the include paths specified by the compiler.
 * 
 * <Angle Brackets>: for including standard library headers or headers from external libraries.
 * the preprocessor searches for the file in standard system directories or directories specified as part of the compiler's standard include
 * path. Typically, you use angle brackets for including standard library headers or headers that are part of a library or framework.
*/

#include "BBLightGlue.hpp"

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/types.hpp>

#include <iostream>
#include <set>
#include <string>

#define WITH_TICTOC
    #include <tictoc.hpp>
#define WITH_TICTOC

using namespace ORB_SLAM3;
using namespace std;

namespace SELMSLAM {

    BBLightGlue::BBLightGlue() {
    }

    BBLightGlue::BBLightGlue(std::string modelPath, float matchThresh) {
        this->m_matchThresh = matchThresh;
        this->m_modelPath = modelPath;
    }

    MatchesInfo::MatchesInfo() {
        this->src_img_idx = 0; // keyframe.nmId
        this->dst_img_idx = 0;// frame.nmId
    
        this->match1counts = 0;
        this->match2counts = 0;

        this->mscore1count = 0;
        this->mscore2count = 0;
    }

    void BBLightGlue::CallbackCheckObservations(MapPoint* pMP, Frame& F, const int i) {
        if (F.mvpMapPoints[i]) {
            if(F.mvpMapPoints[i]->Observations() <= 0){ // !!! ACHTUNG ACHTUNG  !!!
                F.mvpMapPoints[i] = pMP;
            }
        } else {
            F.mvpMapPoints[i] = pMP;
        }
    }

    void BBLightGlue::CallbackCheckMapPoint(MapPoint* pMP, Frame& F, const int i) {
        if (pMP && !pMP->isBad()) {
            F.mvpMapPoints[i] = pMP;
        }
    }

    int BBLightGlue::match(KeyFrame* pKF, Frame &F, vector<MapPoint*> &vpMapPointMatches) {

        /**
         * 
         * Step 1. 
         * Initialization
         * 
        */
        // retrieves a vector of MapPoint pointers from the keyframe
        const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

        // will store matched MapPoints for the current frame
        vpMapPointMatches = vector<MapPoint*>(F.N, static_cast<MapPoint*>(NULL));

        // keyframe
        SELMSLAM::ImageFeatures features1;

        long unsigned int kfId = pKF->mnId;
        cv::Mat kfDescriptors = pKF->mDescriptors;
        vector<cv::KeyPoint> kfKeypoints = pKF->mvKeys;

        features1.img_idx = kfId;
        features1.img_size = cv::Size(640, 480); // @todo Di Okt. 31 07:45 read from settings
        features1.keypoints = kfKeypoints;
        features1.descriptors = kfDescriptors.getUMat(cv::ACCESS_FAST);

        // current frame
        SELMSLAM::ImageFeatures features2;

        long unsigned int fId = F.mnId;
        cv::Mat fDescriptors = F.mDescriptors;
        vector<cv::KeyPoint> fKeypoints = F.mvKeys;

        features2.img_idx = fId;
        features2.img_size = cv::Size(640, 480);
        features2.keypoints = fKeypoints;
        features2.descriptors = fDescriptors.getUMat(cv::ACCESS_FAST);

        SELMSLAM::MatchesInfo matches_info;

        // TIC
        // BBLightGlue::perform_match(features1, features2, matches_info);
        // TOC

        // 
        // !!!
        // !!! Awful coding. improve it !!!
        // !!!
        // 
        vector<float> kp1;
        vector<float> kp2;

        // determine the size of the vectors
        kp1.resize(features1.keypoints.size() * 2);
        kp2.resize(features2.keypoints.size() * 2);

        float f1wid = features1.img_size.width / 2.0f;
        float f1hei = features1.img_size.height / 2.0f;

        // initialize the value of the vectors
        for (int i = 0; i < features1.keypoints.size(); i++) {
            kp1[2 * i] = (features1.keypoints[i].pt.x - f1wid) / f1wid;
            kp1[2 * i + 1] = (features1.keypoints[i].pt.y - f1hei) / f1hei;
        }

        float f2wid = features2.img_size.width / 2.0f;
        float f2hei = features2.img_size.height / 2.0f;

        for (int i = 0; i < features2.keypoints.size(); i++) {
            kp2[2 * i] = (features2.keypoints[i].pt.x - f2wid) / f2wid;
            kp2[2 * i + 1] = (features2.keypoints[i].pt.y - f2hei) / f2hei;
        }

        vector<float> des1;
        vector<float> des2;

        des1.resize(features1.keypoints.size() * 256);
        des2.resize(features2.keypoints.size() * 256);

        cv::Mat des1mat = features1.descriptors.getMat(cv::ACCESS_READ);
        cv::Mat des2mat = features2.descriptors.getMat(cv::ACCESS_READ);

        for (int w = 0; w < des1mat.cols; w++) {
            for (int h = 0; h < des1mat.rows; h++) {
                int index = h * features1.descriptors.cols + w;
                des1[index] = des1mat.at<float>(h, w);
            }
        }

        for (int w = 0; w < des2mat.cols; w++) {
            for (int h = 0; h < des2mat.rows; h++) {
                int index = h * features2.descriptors.cols + w;
                des2[index] = des2mat.at<float>(h, w);
            }
        }

        // 
        // CUDA Provider initialization
        // 
        // to interaction with the ORT runtime, enabling the execution of ONNX models.
        const auto& api = Ort::GetApi();

        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        api.CreateCUDAProviderOptions(&cuda_options);
        std::vector<const char*> keys{"device_id"};
        std::vector<const char*> values{"0"};

        api.UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size());

        Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "BBLightGlue");

        Ort::SessionOptions sessionOptions;

        api.SessionOptionsAppendExecutionProvider_CUDA_V2(sessionOptions, cuda_options);

        // 
        // Load the BBLightGlue network
        // 

        static Ort::Session session(env, this->m_modelPath.c_str(), sessionOptions);

        Ort::MemoryInfo memoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
        Ort::Allocator cuda_allocator(session, memoryInfo);

        // 
        // input binding
        // 
        const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
        const char* output_names[] = {"matches0", "matches1", "mscores0", "mscores1"};

        vector<int64_t> kp1Shape{1, (int64_t)features1.keypoints.size(), 2};
        vector<int64_t> kp2Shape{1, (int64_t)features2.keypoints.size(), 2};

        vector<int64_t> des1Shape{1, (int64_t)features1.keypoints.size(), features1.descriptors.cols};
        vector<int64_t> des2Shape{1, (int64_t)features2.keypoints.size(), features2.descriptors.cols};

        Ort::IoBinding io_binding(session);

        // 
        // Source: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#performance-tuning
        // 
        
        auto input_data_kp1 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(kp1.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));
        auto input_data_kp2 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(kp2.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));

        auto input_data_des1 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(des1.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));
        auto input_data_des2 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(des2.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));

        cudaMemcpy(input_data_kp1.get(), kp1.data(), sizeof(float) * kp1.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(input_data_kp2.get(), kp2.data(), sizeof(float) * kp2.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(input_data_des1.get(), des1.data(), sizeof(float) * des1.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(input_data_des2.get(), des2.data(), sizeof(float) * des2.size(), cudaMemcpyHostToDevice);

        // Create an OrtValue tensor backed by data on CUDA memory
        // reinterpret_cast<float*>: a type-casting operation used to interpret the raw pointer as a pointer to float. CreateTensor function expects a float* pointer.
        Ort::Value bound_x_kp1 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_kp1.get()), kp1.size(), kp1Shape.data(), kp1Shape.size());
        Ort::Value bound_x_kp2 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_kp2.get()), kp2.size(), kp2Shape.data(), kp2Shape.size());
        Ort::Value bound_x_des1 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_des1.get()), des1.size(), des1Shape.data(), des1Shape.size());
        Ort::Value bound_x_des2 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_des2.get()), des2.size(), des2Shape.data(), des2Shape.size());

        io_binding.BindInput("kpts0", bound_x_kp1);
        io_binding.BindInput("kpts1", bound_x_kp2);
        io_binding.BindInput("desc0", bound_x_des1);
        io_binding.BindInput("desc1", bound_x_des2);

        // 
        // output binding
        // 

        Ort::MemoryInfo output_mem_info{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};

        for (size_t i = 0; i < sizeof(output_names) / sizeof(output_names[0]); ++i) {

            io_binding.BindOutput(output_names[i], output_mem_info);
        }

        // Run the model (executing the graph)

        TIC
        session.Run(Ort::RunOptions(), io_binding);
        TOC

        vector<Ort::Value> outputs = io_binding.GetOutputValues();

        // Allocate host memory for the output tensors
        std::vector<int64_t> match1shape  = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> match2shape  = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> mscoreshape1 = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> mscoreshape2 = outputs[3].GetTensorTypeAndShapeInfo().GetShape();

        int match1counts = match1shape[1];
        int match2counts = match2shape[1];
        int mscore1count = mscoreshape1[1];
        int mscore2count = mscoreshape2[1];

        // int64_t* match1_host = new int64_t[match1shape[0] * match1shape[1]];
        // int64_t* match2_host = new int64_t[match2shape[0] * match2shape[1]];
        // float* mscore1_host = new float[mscoreshape1[0] * mscoreshape1[1]];
        // float* mscore2_host = new float[mscoreshape2[0] * mscoreshape2[1]];

        // Use std::vector instead of dynamic arrays
        std::vector<int64_t> match1_host(match1shape[0] * match1shape[1]);
        std::vector<int64_t> match2_host(match2shape[0] * match2shape[1]);
        std::vector<float> mscore1_host(mscoreshape1[0] * mscoreshape1[1]);
        std::vector<float> mscore2_host(mscoreshape2[0] * mscoreshape2[1]);

        // Copy data from GPU to CPU
        int64_t* match1 = (int64_t*)outputs[0].GetTensorMutableData<void>();
        int64_t* match2 = (int64_t*)outputs[1].GetTensorMutableData<void>();
        float* mscore1 = (float*)outputs[2].GetTensorMutableData<void>();
        float* mscore2 = (float*)outputs[3].GetTensorMutableData<void>();

        cudaMemcpy(match1_host.data(), match1, sizeof(int64_t) * match1shape[0] * match1shape[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(match2_host.data(), match2, sizeof(int64_t) * match2shape[0] * match2shape[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(mscore1_host.data(), mscore1, sizeof(float) * mscoreshape1[0] * mscoreshape1[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(mscore2_host.data(), mscore2, sizeof(float) * mscoreshape2[0] * mscoreshape2[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(match1_host, match1, sizeof(int64_t) * match1shape[0] * match1shape[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(match2_host, match2, sizeof(int64_t) * match2shape[0] * match2shape[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(mscore1_host, mscore1, sizeof(float) * mscoreshape1[0] * mscoreshape1[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(mscore2_host, mscore2, sizeof(float) * mscoreshape2[0] * mscoreshape2[1], cudaMemcpyDeviceToHost);

        // matches_info.match1 = match1_host;
        // matches_info.match2 = match2_host;
        // matches_info.mscore1 = mscore1_host;
        // matches_info.mscore2 = mscore2_host;

        matches_info.match1 = match1_host.data();
        matches_info.match2 = match2_host.data();
        matches_info.mscore1 = mscore1_host.data();
        matches_info.mscore2 = mscore2_host.data();

        matches_info.src_img_idx = features1.img_idx;
	    matches_info.dst_img_idx = features2.img_idx;

        matches_info.match1shape = match1shape;
        matches_info.match2shape = match2shape;

        matches_info.match1counts = match1counts;
        matches_info.match2counts = match2counts;

        matches_info.mscoreshape1 = mscoreshape1;
        matches_info.mscoreshape2 = mscoreshape2;

        matches_info.mscore1count = mscore1count;
        matches_info.mscore2count = mscore2count;

        //
        // !!!
        // !!! end of Awfulness !!!
        // !!!
        // 

        // to store unique pairs of matched keypoints. This set will be used to ensure that duplicates are not included in the final matches.
        std::set<std::pair<int, int> > matches;
        /**
         * For each match, it checks if 
         *      the match index is valid, 
         *      the score is above a threshold, 
         *      and if it has a reciprocal match. 
         * If these conditions are met, a cv::DMatch object mt is created, and it's added to matches_info.matches. 
         * The pair of keypoints is added to the matches set.
        */

        for (int i = 0; i < matches_info.match1counts; i++) {

            if (matches_info.match1[i] > -1 && matches_info.mscore1[i] > this->m_matchThresh && matches_info.match2[matches_info.match1[i]] == i) {
                cv::DMatch mt;
                mt.queryIdx = i;
                mt.trainIdx = matches_info.match1[i];
                matches_info.matches.push_back(mt);
                matches.insert(std::make_pair(mt.queryIdx, mt.trainIdx));

                // Banafshe enters ...
                // associates the matched MapPoint (pMP) with the feature in the current frame
                MapPoint* pMP = vpMapPointsKF[mt.queryIdx];
                if(pMP && !pMP->isBad()) {
                    vpMapPointMatches[mt.trainIdx] = pMP;
                }
            }
        }

        for (int i = 0; i < matches_info.match2counts; i++) {

            if (matches_info.match2[i] > -1 && matches_info.mscore2[i] > this->m_matchThresh && matches_info.match1[matches_info.match2[i]] == i) {

                cv::DMatch mt;
                mt.queryIdx = matches_info.match2[i];
                mt.trainIdx = i;

                if (matches.find(std::make_pair(mt.queryIdx, mt.trainIdx)) == matches.end()) {

                    matches_info.matches.push_back(mt);

                    // Banafshe enters ...
                    MapPoint* pMP = vpMapPointsKF[mt.queryIdx];
                    if(pMP && !pMP->isBad()) {
                        vpMapPointMatches[mt.trainIdx] = pMP;
                    }
                }
            }
        }

        return matches_info.matches.size();

    } // match method


    // 
    // 
    // 
    // !!! ACHTUNG ACHTUNG !!!
    // 
    // This method should merge with the method above 
    // 
    // !!! ACHTUNG ACHTUNG !!!
    // 
    // 
        // int BBLightGlue::match(Frame* pKF, Frame &F, vector<MapPoint*> &vpMapPointMatches) {
        int BBLightGlue::match(Frame lastFrame, Frame &F) {

        /**
         * 
         * Step 1. 
         * Initialization
         * 
        */
        // retrieves a vector of MapPoint pointers from the keyframe
        const vector<MapPoint*> vpMapPointsKF = lastFrame.mvpMapPoints;

        // will store matched MapPoints for the current frame
        // vpMapPointMatches = vector<MapPoint*>(F.N, static_cast<MapPoint*>(NULL));

        // keyframe
        SELMSLAM::ImageFeatures features1;

        long unsigned int kfId = lastFrame.mnId;
        cv::Mat kfDescriptors = lastFrame.mDescriptors;
        vector<cv::KeyPoint> kfKeypoints = lastFrame.mvKeys;

        features1.img_idx = kfId;
        features1.img_size = cv::Size(640, 480); // @todo Di Okt. 31 07:45 read from settings
        features1.keypoints = kfKeypoints;
        features1.descriptors = kfDescriptors.getUMat(cv::ACCESS_FAST);

        // current frame
        SELMSLAM::ImageFeatures features2;

        long unsigned int fId = F.mnId;
        cv::Mat fDescriptors = F.mDescriptors;
        vector<cv::KeyPoint> fKeypoints = F.mvKeys;

        features2.img_idx = fId;
        features2.img_size = cv::Size(640, 480);
        features2.keypoints = fKeypoints;
        features2.descriptors = fDescriptors.getUMat(cv::ACCESS_FAST);

        SELMSLAM::MatchesInfo matches_info;

        // 
        // !!!
        // !!! Awful coding. improve it !!!
        // !!!
        // 
        vector<float> kp1;
        vector<float> kp2;

        kp1.resize(features1.keypoints.size() * 2);
        kp2.resize(features2.keypoints.size() * 2);

        float f1wid = features1.img_size.width / 2.0f;
        float f1hei = features1.img_size.height / 2.0f;

        for (int i = 0; i < features1.keypoints.size(); i++) {
            kp1[2 * i] = (features1.keypoints[i].pt.x - f1wid) / f1wid;
            kp1[2 * i + 1] = (features1.keypoints[i].pt.y - f1hei) / f1hei;
        }

        float f2wid = features2.img_size.width / 2.0f;
        float f2hei = features2.img_size.height / 2.0f;

        for (int i = 0; i < features2.keypoints.size(); i++) {
            kp2[2 * i] = (features2.keypoints[i].pt.x - f2wid) / f2wid;
            kp2[2 * i + 1] = (features2.keypoints[i].pt.y - f2hei) / f2hei;
        }

        vector<float> des1;
        vector<float> des2;

        des1.resize(features1.keypoints.size() * 256);
        des2.resize(features2.keypoints.size() * 256);

        cv::Mat des1mat = features1.descriptors.getMat(cv::ACCESS_READ);
        cv::Mat des2mat = features2.descriptors.getMat(cv::ACCESS_READ);

        for (int w = 0; w < des1mat.cols; w++) {
            for (int h = 0; h < des1mat.rows; h++) {
                int index = h * features1.descriptors.cols + w;
                des1[index] = des1mat.at<float>(h, w);
            }
        }

        for (int w = 0; w < des2mat.cols; w++) {
            for (int h = 0; h < des2mat.rows; h++) {
                int index = h * features2.descriptors.cols + w;
                des2[index] = des2mat.at<float>(h, w);
            }
        }

        // 
        // CUDA Provider initialization
        // 
        // to interaction with the ORT runtime, enabling the execution of ONNX models.
        const auto& api = Ort::GetApi();

        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        api.CreateCUDAProviderOptions(&cuda_options);
        std::vector<const char*> keys{"device_id"};
        std::vector<const char*> values{"0"};

        api.UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size());

        Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "BBLightGlue");

        Ort::SessionOptions sessionOptions;

        api.SessionOptionsAppendExecutionProvider_CUDA_V2(sessionOptions, cuda_options);

        // 
        // Load the BBLightGlue network
        // 

        static Ort::Session session(env, this->m_modelPath.c_str(), sessionOptions);

        Ort::MemoryInfo memoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
        Ort::Allocator cuda_allocator(session, memoryInfo);

        // 
        // input binding
        // 
        const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
        const char* output_names[] = {"matches0", "matches1", "mscores0", "mscores1"};

        vector<int64_t> kp1Shape{1, (int64_t)features1.keypoints.size(), 2};
        vector<int64_t> kp2Shape{1, (int64_t)features2.keypoints.size(), 2};

        vector<int64_t> des1Shape{1, (int64_t)features1.keypoints.size(), features1.descriptors.cols};
        vector<int64_t> des2Shape{1, (int64_t)features2.keypoints.size(), features2.descriptors.cols};

        Ort::IoBinding io_binding(session);

        // 
        // Source: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#performance-tuning
        // 
        
        auto input_data_kp1 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(kp1.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));
        auto input_data_kp2 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(kp2.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));

        auto input_data_des1 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(des1.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));
        auto input_data_des2 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(des2.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));

        cudaMemcpy(input_data_kp1.get(), kp1.data(), sizeof(float) * kp1.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(input_data_kp2.get(), kp2.data(), sizeof(float) * kp2.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(input_data_des1.get(), des1.data(), sizeof(float) * des1.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(input_data_des2.get(), des2.data(), sizeof(float) * des2.size(), cudaMemcpyHostToDevice);

        // Create an OrtValue tensor backed by data on CUDA memory
        // reinterpret_cast<float*>: a type-casting operation used to interpret the raw pointer as a pointer to float. CreateTensor function expects a float* pointer.
        Ort::Value bound_x_kp1 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_kp1.get()), kp1.size(), kp1Shape.data(), kp1Shape.size());
        Ort::Value bound_x_kp2 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_kp2.get()), kp2.size(), kp2Shape.data(), kp2Shape.size());
        Ort::Value bound_x_des1 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_des1.get()), des1.size(), des1Shape.data(), des1Shape.size());
        Ort::Value bound_x_des2 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_des2.get()), des2.size(), des2Shape.data(), des2Shape.size());

        io_binding.BindInput("kpts0", bound_x_kp1);
        io_binding.BindInput("kpts1", bound_x_kp2);
        io_binding.BindInput("desc0", bound_x_des1);
        io_binding.BindInput("desc1", bound_x_des2);

        // 
        // output binding
        // 

        Ort::MemoryInfo output_mem_info{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};

        for (size_t i = 0; i < sizeof(output_names) / sizeof(output_names[0]); ++i) {

            io_binding.BindOutput(output_names[i], output_mem_info);
        }

        // Run the model (executing the graph)

        TIC
        session.Run(Ort::RunOptions(), io_binding);
        TOC

        vector<Ort::Value> outputs = io_binding.GetOutputValues();

        // Allocate host memory for the output tensors
        std::vector<int64_t> match1shape  = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> match2shape  = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> mscoreshape1 = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> mscoreshape2 = outputs[3].GetTensorTypeAndShapeInfo().GetShape();

        int match1counts = match1shape[1];
        int match2counts = match2shape[1];
        int mscore1count = mscoreshape1[1];
        int mscore2count = mscoreshape2[1];

        // int64_t* match1_host = new int64_t[match1shape[0] * match1shape[1]];
        // int64_t* match2_host = new int64_t[match2shape[0] * match2shape[1]];
        // float* mscore1_host = new float[mscoreshape1[0] * mscoreshape1[1]];
        // float* mscore2_host = new float[mscoreshape2[0] * mscoreshape2[1]];

        // Use std::vector instead of dynamic arrays
        std::vector<int64_t> match1_host(match1shape[0] * match1shape[1]);
        std::vector<int64_t> match2_host(match2shape[0] * match2shape[1]);
        std::vector<float> mscore1_host(mscoreshape1[0] * mscoreshape1[1]);
        std::vector<float> mscore2_host(mscoreshape2[0] * mscoreshape2[1]);

        // Copy data from GPU to CPU
        int64_t* match1 = (int64_t*)outputs[0].GetTensorMutableData<void>();
        int64_t* match2 = (int64_t*)outputs[1].GetTensorMutableData<void>();
        float* mscore1 = (float*)outputs[2].GetTensorMutableData<void>();
        float* mscore2 = (float*)outputs[3].GetTensorMutableData<void>();

        cudaMemcpy(match1_host.data(), match1, sizeof(int64_t) * match1shape[0] * match1shape[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(match2_host.data(), match2, sizeof(int64_t) * match2shape[0] * match2shape[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(mscore1_host.data(), mscore1, sizeof(float) * mscoreshape1[0] * mscoreshape1[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(mscore2_host.data(), mscore2, sizeof(float) * mscoreshape2[0] * mscoreshape2[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(match1_host, match1, sizeof(int64_t) * match1shape[0] * match1shape[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(match2_host, match2, sizeof(int64_t) * match2shape[0] * match2shape[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(mscore1_host, mscore1, sizeof(float) * mscoreshape1[0] * mscoreshape1[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(mscore2_host, mscore2, sizeof(float) * mscoreshape2[0] * mscoreshape2[1], cudaMemcpyDeviceToHost);

        // matches_info.match1 = match1_host;
        // matches_info.match2 = match2_host;
        // matches_info.mscore1 = mscore1_host;
        // matches_info.mscore2 = mscore2_host;

        matches_info.match1 = match1_host.data();
        matches_info.match2 = match2_host.data();
        matches_info.mscore1 = mscore1_host.data();
        matches_info.mscore2 = mscore2_host.data();

        matches_info.src_img_idx = features1.img_idx;
	    matches_info.dst_img_idx = features2.img_idx;

        matches_info.match1shape = match1shape;
        matches_info.match2shape = match2shape;

        matches_info.match1counts = match1counts;
        matches_info.match2counts = match2counts;

        matches_info.mscoreshape1 = mscoreshape1;
        matches_info.mscoreshape2 = mscoreshape2;

        matches_info.mscore1count = mscore1count;
        matches_info.mscore2count = mscore2count;

        //
        // !!!
        // !!! end of Awfulness !!!
        // !!!
        // 

        // to store unique pairs of matched keypoints. This set will be used to ensure that duplicates are not included in the final matches.
        std::set<std::pair<int, int> > matches;
        /**
         * For each match, it checks if 
         *      the match index is valid, 
         *      the score is above a threshold, 
         *      and if it has a reciprocal match. 
         * If these conditions are met, a cv::DMatch object mt is created, and it's added to matches_info.matches. 
         * The pair of keypoints is added to the matches set.
        */

        for (int i = 0; i < matches_info.match1counts; i++) {

            if (matches_info.match1[i] > -1 && matches_info.mscore1[i] > this->m_matchThresh && matches_info.match2[matches_info.match1[i]] == i) {
                cv::DMatch mt;
                mt.queryIdx = i;
                mt.trainIdx = matches_info.match1[i];
                matches_info.matches.push_back(mt);
                matches.insert(std::make_pair(mt.queryIdx, mt.trainIdx));

                // Banafshe enters ...
                // associates the matched MapPoint (pMP) with the feature in the current frame
                MapPoint* pMP = vpMapPointsKF[mt.queryIdx];
                if(pMP && !pMP->isBad()) {
                    // vpMapPointMatches[mt.trainIdx] = pMP;
                    F.mvpMapPoints[mt.trainIdx] = pMP;
                }
            }
        }

        for (int i = 0; i < matches_info.match2counts; i++) {

            if (matches_info.match2[i] > -1 && matches_info.mscore2[i] > this->m_matchThresh && matches_info.match1[matches_info.match2[i]] == i) {

                cv::DMatch mt;
                mt.queryIdx = matches_info.match2[i];
                mt.trainIdx = i;

                if (matches.find(std::make_pair(mt.queryIdx, mt.trainIdx)) == matches.end()) {

                    matches_info.matches.push_back(mt);

                    // Banafshe enters ...
                    MapPoint* pMP = vpMapPointsKF[mt.queryIdx];
                    if(pMP && !pMP->isBad()) {
                        // vpMapPointMatches[mt.trainIdx] = pMP;
                        F.mvpMapPoints[mt.trainIdx] = pMP;
                    }
                }
            }
        }

        return matches_info.matches.size();

    } // match method

    // 
    // !!! END OF ACHTUNG BLOCK !!!
    // 
    // 
    // 

    int BBLightGlue::match(const vector<MapPoint*> &vpMapPoints, cv::Mat MPDescriptors, std::vector<cv::KeyPoint> vMPKeypoint, Frame &F, std::function<void(MapPoint*, Frame&, const int i)> callback) {

        // MapPoint/Last frame
        SELMSLAM::ImageFeatures features1;
        cv::Mat mpDescriptors = MPDescriptors;
        vector<cv::KeyPoint> mpKeypoints = vMPKeypoint;
        features1.img_size = cv::Size(640, 480); // @todo Di Okt. 31 07:45 read from settings
        features1.keypoints = mpKeypoints;
        features1.descriptors = mpDescriptors.getUMat(cv::ACCESS_FAST);

        // current frame
        SELMSLAM::ImageFeatures features2;
        long unsigned int fId = F.mnId;
        cv::Mat fDescriptors = F.mDescriptors;
        vector<cv::KeyPoint> fKeypoints = F.mvKeys;
        features2.img_idx = fId;
        features2.img_size = cv::Size(640, 480);
        features2.keypoints = fKeypoints;
        features2.descriptors = fDescriptors.getUMat(cv::ACCESS_FAST);

        SELMSLAM::MatchesInfo matches_info;

        // TIC
        // BBLightGlue::perform_match(features1, features2, matches_info);
        // TOC

        // 
        // !!!
        // !!! Awful coding. improve it !!!
        // !!!
        // 
        vector<float> kp1;
        vector<float> kp2;

        kp1.resize(features1.keypoints.size() * 2);
        kp2.resize(features2.keypoints.size() * 2);

        float f1wid = features1.img_size.width / 2.0f;
        float f1hei = features1.img_size.height / 2.0f;

        for (int i = 0; i < features1.keypoints.size(); i++) {
            kp1[2 * i] = (features1.keypoints[i].pt.x - f1wid) / f1wid;
            kp1[2 * i + 1] = (features1.keypoints[i].pt.y - f1hei) / f1hei;
        }

        float f2wid = features2.img_size.width / 2.0f;
        float f2hei = features2.img_size.height / 2.0f;

        for (int i = 0; i < features2.keypoints.size(); i++) {
            kp2[2 * i] = (features2.keypoints[i].pt.x - f2wid) / f2wid;
            kp2[2 * i + 1] = (features2.keypoints[i].pt.y - f2hei) / f2hei;
        }

        vector<float> des1;
        vector<float> des2;

        des1.resize(features1.keypoints.size() * 256);
        des2.resize(features2.keypoints.size() * 256);

        cv::Mat des1mat = features1.descriptors.getMat(cv::ACCESS_READ);
        cv::Mat des2mat = features2.descriptors.getMat(cv::ACCESS_READ);

        for (int w = 0; w < des1mat.cols; w++) {
            for (int h = 0; h < des1mat.rows; h++) {
                int index = h * features1.descriptors.cols + w;
                des1[index] = des1mat.at<float>(h, w);
            }
        }

        for (int w = 0; w < des2mat.cols; w++) {
            for (int h = 0; h < des2mat.rows; h++) {
                int index = h * features2.descriptors.cols + w;
                des2[index] = des2mat.at<float>(h, w);
            }
        }

        // 
        // CUDA Provider initialization
        // 
        // to interaction with the ORT runtime, enabling the execution of ONNX models.
        const auto& api = Ort::GetApi();

        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        api.CreateCUDAProviderOptions(&cuda_options);
        std::vector<const char*> keys{"device_id"};
        std::vector<const char*> values{"0"};

        api.UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size());

        Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "BBLightGlue");

        Ort::SessionOptions sessionOptions;

        api.SessionOptionsAppendExecutionProvider_CUDA_V2(sessionOptions, cuda_options);

        // 
        // Load the BBLightGlue network
        // 

        static Ort::Session session(env, this->m_modelPath.c_str(), sessionOptions);

        Ort::MemoryInfo memoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
        Ort::Allocator cuda_allocator(session, memoryInfo);

        // 
        // input binding
        // 
        const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
        const char* output_names[] = {"matches0", "matches1", "mscores0", "mscores1"};

        vector<int64_t> kp1Shape{1, (int64_t)features1.keypoints.size(), 2};
        vector<int64_t> kp2Shape{1, (int64_t)features2.keypoints.size(), 2};

        vector<int64_t> des1Shape{1, (int64_t)features1.keypoints.size(), features1.descriptors.cols};
        vector<int64_t> des2Shape{1, (int64_t)features2.keypoints.size(), features2.descriptors.cols};

        Ort::IoBinding io_binding(session);

        // 
        // Source: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#performance-tuning
        // 
        
        auto input_data_kp1 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(kp1.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));
        auto input_data_kp2 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(kp2.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));

        auto input_data_des1 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(des1.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));
        auto input_data_des2 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(des2.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));

        cudaMemcpy(input_data_kp1.get(), kp1.data(), sizeof(float) * kp1.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(input_data_kp2.get(), kp2.data(), sizeof(float) * kp2.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(input_data_des1.get(), des1.data(), sizeof(float) * des1.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(input_data_des2.get(), des2.data(), sizeof(float) * des2.size(), cudaMemcpyHostToDevice);

        // Create an OrtValue tensor backed by data on CUDA memory
        // reinterpret_cast<float*>: a type-casting operation used to interpret the raw pointer as a pointer to float. CreateTensor function expects a float* pointer.
        Ort::Value bound_x_kp1 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_kp1.get()), kp1.size(), kp1Shape.data(), kp1Shape.size());
        Ort::Value bound_x_kp2 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_kp2.get()), kp2.size(), kp2Shape.data(), kp2Shape.size());
        Ort::Value bound_x_des1 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_des1.get()), des1.size(), des1Shape.data(), des1Shape.size());
        Ort::Value bound_x_des2 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_des2.get()), des2.size(), des2Shape.data(), des2Shape.size());

        io_binding.BindInput("kpts0", bound_x_kp1);
        io_binding.BindInput("kpts1", bound_x_kp2);
        io_binding.BindInput("desc0", bound_x_des1);
        io_binding.BindInput("desc1", bound_x_des2);

        // 
        // output binding
        // 

        Ort::MemoryInfo output_mem_info{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};

        for (size_t i = 0; i < sizeof(output_names) / sizeof(output_names[0]); ++i) {

            io_binding.BindOutput(output_names[i], output_mem_info);
        }

        // Run the model (executing the graph)

        TIC
        session.Run(Ort::RunOptions(), io_binding);
        TOC

        vector<Ort::Value> outputs = io_binding.GetOutputValues();

        // Allocate host memory for the output tensors
        std::vector<int64_t> match1shape  = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> match2shape  = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> mscoreshape1 = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> mscoreshape2 = outputs[3].GetTensorTypeAndShapeInfo().GetShape();

        int match1counts = match1shape[1];
        int match2counts = match2shape[1];
        int mscore1count = mscoreshape1[1];
        int mscore2count = mscoreshape2[1];

        // int64_t* match1_host = new int64_t[match1shape[0] * match1shape[1]];
        // int64_t* match2_host = new int64_t[match2shape[0] * match2shape[1]];
        // float* mscore1_host = new float[mscoreshape1[0] * mscoreshape1[1]];
        // float* mscore2_host = new float[mscoreshape2[0] * mscoreshape2[1]];

        // Use std::vector instead of dynamic arrays
        std::vector<int64_t> match1_host(match1shape[0] * match1shape[1]);
        std::vector<int64_t> match2_host(match2shape[0] * match2shape[1]);
        std::vector<float> mscore1_host(mscoreshape1[0] * mscoreshape1[1]);
        std::vector<float> mscore2_host(mscoreshape2[0] * mscoreshape2[1]);

        // Copy data from GPU to CPU
        int64_t* match1 = (int64_t*)outputs[0].GetTensorMutableData<void>();
        int64_t* match2 = (int64_t*)outputs[1].GetTensorMutableData<void>();
        float* mscore1 = (float*)outputs[2].GetTensorMutableData<void>();
        float* mscore2 = (float*)outputs[3].GetTensorMutableData<void>();

        cudaMemcpy(match1_host.data(), match1, sizeof(int64_t) * match1shape[0] * match1shape[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(match2_host.data(), match2, sizeof(int64_t) * match2shape[0] * match2shape[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(mscore1_host.data(), mscore1, sizeof(float) * mscoreshape1[0] * mscoreshape1[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(mscore2_host.data(), mscore2, sizeof(float) * mscoreshape2[0] * mscoreshape2[1], cudaMemcpyDeviceToHost);

        // cudaMemcpy(match1_host, match1, sizeof(int64_t) * match1shape[0] * match1shape[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(match2_host, match2, sizeof(int64_t) * match2shape[0] * match2shape[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(mscore1_host, mscore1, sizeof(float) * mscoreshape1[0] * mscoreshape1[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(mscore2_host, mscore2, sizeof(float) * mscoreshape2[0] * mscoreshape2[1], cudaMemcpyDeviceToHost);

        // matches_info.match1 = match1_host;
        // matches_info.match2 = match2_host;
        // matches_info.mscore1 = mscore1_host;
        // matches_info.mscore2 = mscore2_host;

        matches_info.match1 = match1_host.data();
        matches_info.match2 = match2_host.data();
        matches_info.mscore1 = mscore1_host.data();
        matches_info.mscore2 = mscore2_host.data();

        matches_info.src_img_idx = features1.img_idx;
	    matches_info.dst_img_idx = features2.img_idx;

        matches_info.match1shape = match1shape;
        matches_info.match2shape = match2shape;

        matches_info.match1counts = match1counts;
        matches_info.match2counts = match2counts;

        matches_info.mscoreshape1 = mscoreshape1;
        matches_info.mscoreshape2 = mscoreshape2;

        matches_info.mscore1count = mscore1count;
        matches_info.mscore2count = mscore2count;

        //
        // !!!
        // !!! end of Awfulness !!!
        // !!!
        // 
        std::set<std::pair<int, int> > matches;

        for (int i = 0; i < matches_info.match1counts; i++) {

            if (matches_info.match1[i] > -1 && matches_info.mscore1[i] > this->m_matchThresh && matches_info.match2[matches_info.match1[i]] == i) {

                cv::DMatch mt;
                mt.queryIdx = i;
                mt.trainIdx = matches_info.match1[i];

                matches_info.matches.push_back(mt);

                matches.insert(std::make_pair(mt.queryIdx, mt.trainIdx)); // is used in the next loop

                // MapPoint* pMP = vpMapPoints[mt.queryIdx];
                MapPoint* pMP = vpMapPoints[i];

                // if(pMP && !pMP->isBad()) {
                //     F.mvpMapPoints[mt.trainIdx] = pMP;
                // }
                // callback(pMP, F, i);
                
                callback(pMP, F, mt.trainIdx);
                // if(F.mvpMapPoints[mt.trainIdx]->Observations() <= 0) {
                    // F.mvpMapPoints[mt.trainIdx] = pMP;
                // }

            }
        }

        for (int i = 0; i < matches_info.match2counts; i++) {
            if (matches_info.match2[i] > -1 && matches_info.mscore2[i] > this->m_matchThresh && matches_info.match1[matches_info.match2[i]] == i) {
                cv::DMatch mt;
                mt.queryIdx = matches_info.match2[i];
                mt.trainIdx = i;

                if (matches.find(std::make_pair(mt.queryIdx, mt.trainIdx)) == matches.end()){
                    matches_info.matches.push_back(mt);

                    MapPoint* pMP = vpMapPoints[mt.queryIdx];
                    // if(pMP && !pMP->isBad()) {
                    //     F.mvpMapPoints[mt.trainIdx] = pMP;
                    // }
                    // callback(pMP, F, i);
                    callback(pMP, F, mt.trainIdx);
                }
            }
        }

        return matches_info.matches.size();

    } // match method

    void BBLightGlue::perform_match(SELMSLAM::ImageFeatures features1, SELMSLAM::ImageFeatures features2, SELMSLAM::MatchesInfo &matches_info) {

        // SELMSLAM::MatchesInfo matches_info;

        vector<float> kp1;
        vector<float> kp2;

        kp1.resize(features1.keypoints.size() * 2);
        kp2.resize(features2.keypoints.size() * 2);

        float f1wid = features1.img_size.width / 2.0f;
        float f1hei = features1.img_size.height / 2.0f;

        for (int i = 0; i < features1.keypoints.size(); i++) {
            kp1[2 * i] = (features1.keypoints[i].pt.x - f1wid) / f1wid;
            kp1[2 * i + 1] = (features1.keypoints[i].pt.y - f1hei) / f1hei;
        }

        float f2wid = features2.img_size.width / 2.0f;
        float f2hei = features2.img_size.height / 2.0f;

        for (int i = 0; i < features2.keypoints.size(); i++) {
            kp2[2 * i] = (features2.keypoints[i].pt.x - f2wid) / f2wid;
            kp2[2 * i + 1] = (features2.keypoints[i].pt.y - f2hei) / f2hei;
        }

        vector<float> des1;
        vector<float> des2;

        des1.resize(features1.keypoints.size() * 256);
        des2.resize(features2.keypoints.size() * 256);

        cv::Mat des1mat = features1.descriptors.getMat(cv::ACCESS_READ);
        cv::Mat des2mat = features2.descriptors.getMat(cv::ACCESS_READ);

        for (int w = 0; w < des1mat.cols; w++) {
            for (int h = 0; h < des1mat.rows; h++) {
                int index = h * features1.descriptors.cols + w;
                des1[index] = des1mat.at<float>(h, w);
            }
        }

        for (int w = 0; w < des2mat.cols; w++) {
            for (int h = 0; h < des2mat.rows; h++) {
                int index = h * features2.descriptors.cols + w;
                des2[index] = des2mat.at<float>(h, w);
            }
        }

        // 
        // CUDA Provider initialization
        // 
        // to interaction with the ORT runtime, enabling the execution of ONNX models.
        const auto& api = Ort::GetApi();

        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        api.CreateCUDAProviderOptions(&cuda_options);
        std::vector<const char*> keys{"device_id"};
        std::vector<const char*> values{"0"};

        api.UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size());

        Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "BBLightGlue");

        Ort::SessionOptions sessionOptions;

        api.SessionOptionsAppendExecutionProvider_CUDA_V2(sessionOptions, cuda_options);

        // 
        // Load the BBLightGlue network
        // 

        static Ort::Session session(env, this->m_modelPath.c_str(), sessionOptions);
        Ort::MemoryInfo memoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
        Ort::Allocator cuda_allocator(session, memoryInfo);

        // 
        // input binding
        // 
        const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
        const char* output_names[] = {"matches0", "matches1", "mscores0", "mscores1"};

        vector<int64_t> kp1Shape{1, (int64_t)features1.keypoints.size(), 2};
        vector<int64_t> kp2Shape{1, (int64_t)features2.keypoints.size(), 2};

        vector<int64_t> des1Shape{1, (int64_t)features1.keypoints.size(), features1.descriptors.cols};
        vector<int64_t> des2Shape{1, (int64_t)features2.keypoints.size(), features2.descriptors.cols};

        Ort::IoBinding io_binding(session);

        // 
        // Source: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#performance-tuning
        // 
        
        auto input_data_kp1 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(kp1.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));
        auto input_data_kp2 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(kp2.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));

        auto input_data_des1 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(des1.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));
        auto input_data_des2 = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(des2.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));

        // to copy  data from the host (specified by x_values.data()) to the device (specified by input_data.get()) with a size of sizeof(float) * x_values.size() bytes.
        // cudaMemcpyHostToDevice is in cuda_runtime.h in the follwoing path
        // -DCUDA_INCLUDE_PATH="/usr/local/cuda-11.8/targets/x86_64-linux/include"
        cudaMemcpy(input_data_kp1.get(), kp1.data(), sizeof(float) * kp1.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(input_data_kp2.get(), kp2.data(), sizeof(float) * kp2.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(input_data_des1.get(), des1.data(), sizeof(float) * des1.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(input_data_des2.get(), des2.data(), sizeof(float) * des2.size(), cudaMemcpyHostToDevice);

        // Create an OrtValue tensor backed by data on CUDA memory
        // reinterpret_cast<float*>: a type-casting operation used to interpret the raw pointer as a pointer to float. CreateTensor function expects a float* pointer.
        Ort::Value bound_x_kp1 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_kp1.get()), kp1.size(), kp1Shape.data(), kp1Shape.size());
        Ort::Value bound_x_kp2 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_kp2.get()), kp2.size(), kp2Shape.data(), kp2Shape.size());
        Ort::Value bound_x_des1 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_des1.get()), des1.size(), des1Shape.data(), des1Shape.size());
        Ort::Value bound_x_des2 = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data_des2.get()), des2.size(), des2Shape.data(), des2Shape.size());

        io_binding.BindInput("kpts0", bound_x_kp1);
        io_binding.BindInput("kpts1", bound_x_kp2);
        io_binding.BindInput("desc0", bound_x_des1);
        io_binding.BindInput("desc1", bound_x_des2);

        // 
        // output binding
        // 

        Ort::MemoryInfo output_mem_info{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};

        for (size_t i = 0; i < sizeof(output_names) / sizeof(output_names[0]); ++i) {

            io_binding.BindOutput(output_names[i], output_mem_info);
        }

        // Run the model (executing the graph)

        TIC
        session.Run(Ort::RunOptions(), io_binding);
        TOC

        vector<Ort::Value> outputs = io_binding.GetOutputValues();

        cout << endl << "B.B in Perform matcher method. the size of outputs is: " << outputs.size() << endl;

        // Allocate host memory for the output tensors
        std::vector<int64_t> match1shape  = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> match2shape  = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> mscoreshape1 = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> mscoreshape2 = outputs[3].GetTensorTypeAndShapeInfo().GetShape();

        cout << endl << "B.B match1shape size: " << match1shape << ", match2shape size: " << match2shape << ", mscoreshape1 size: " << mscoreshape1 << ", mscoreshape2 size: " << mscoreshape2;

        int match1counts = match1shape[1];
        int match2counts = match2shape[1];
        int mscore1count = mscoreshape1[1];
        int mscore2count = mscoreshape2[1];

        cout << endl << "B.B match1counts: " << match1counts << ", match2counts: " << match2counts << ", mscore1count: " << mscore1count << ", mscore2count: " << mscore2count << endl;

        // int64_t* match1_host = new int64_t[match1shape[0] * match1shape[1]];
        // int64_t* match2_host = new int64_t[match2shape[0] * match2shape[1]];
        // float* mscore1_host = new float[mscoreshape1[0] * mscoreshape1[1]];
        // float* mscore2_host = new float[mscoreshape2[0] * mscoreshape2[1]];

        // Use std::vector instead of dynamic arrays
        std::vector<int64_t> match1_host(match1shape[0] * match1shape[1]);
        std::vector<int64_t> match2_host(match2shape[0] * match2shape[1]);
        std::vector<float> mscore1_host(mscoreshape1[0] * mscoreshape1[1]);
        std::vector<float> mscore2_host(mscoreshape2[0] * mscoreshape2[1]);

        cout << endl << "B.B BLightGlue::perform_match. host memory is initialized ..." << endl;

        // Copy data from GPU to CPU
        int64_t* match1 = (int64_t*)outputs[0].GetTensorMutableData<void>();
        int64_t* match2 = (int64_t*)outputs[1].GetTensorMutableData<void>();
        float* mscore1 = (float*)outputs[2].GetTensorMutableData<void>();
        float* mscore2 = (float*)outputs[3].GetTensorMutableData<void>();

        cudaMemcpy(match1_host.data(), match1, sizeof(int64_t) * match1shape[0] * match1shape[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(match2_host.data(), match2, sizeof(int64_t) * match2shape[0] * match2shape[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(mscore1_host.data(), mscore1, sizeof(float) * mscoreshape1[0] * mscoreshape1[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(mscore2_host.data(), mscore2, sizeof(float) * mscoreshape2[0] * mscoreshape2[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(match1_host, match1, sizeof(int64_t) * match1shape[0] * match1shape[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(match2_host, match2, sizeof(int64_t) * match2shape[0] * match2shape[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(mscore1_host, mscore1, sizeof(float) * mscoreshape1[0] * mscoreshape1[1], cudaMemcpyDeviceToHost);
        // cudaMemcpy(mscore2_host, mscore2, sizeof(float) * mscoreshape2[0] * mscoreshape2[1], cudaMemcpyDeviceToHost);

        cout << endl << "B.B BLightGlue::perform_match. data from GPU is copied to CPU memory ..." << endl;

        // matches_info.match1 = match1_host;
        // matches_info.match2 = match2_host;
        // matches_info.mscore1 = mscore1_host;
        // matches_info.mscore2 = mscore2_host;

        matches_info.match1 = match1_host.data();
        matches_info.match2 = match2_host.data();
        matches_info.mscore1 = mscore1_host.data();
        matches_info.mscore2 = mscore2_host.data();

        // for(size_t i = 0; i < match1_host.size(); ++i) {
        //     std::cout << "m1: " << matches_info.match1[i] << " - " << match1_host[i] << endl;
        // }

        /**
         * Since I encountered issues accessing match1, match2, mscore1, and mscore2 values outside of this class, 
         * I staticlly store their values in these four data structure
        */

       
        matches_info.vmatch1.resize(match1_host.size());
        for(size_t i = 0; i < match1_host.size(); ++i) {
            // matches_info.vmatch1.push_back(static_cast<int>(match1_host[i])); WRONG WRONG WRONG
            matches_info.vmatch1[i] = static_cast<int>(match1_host[i]);
        }
        matches_info.vmatch2.resize(match2_host.size());
        for(size_t i = 0; i < match2_host.size(); ++i) {
            matches_info.vmatch2[i]  = static_cast<int>(match2_host[i]);
        }
        matches_info.vmscore1.resize(mscore1_host.size());
        for(size_t i = 0; i < mscore1_host.size(); ++i) {
            matches_info.vmscore1[i] = mscore1_host[i];
        }
        matches_info.vmscore2.resize(mscore2_host.size());
        for(size_t i = 0; i < mscore2_host.size(); ++i) {
            matches_info.vmscore2[i] = mscore2_host[i];
        }

        cout << endl << "B.B BLightGlue::perform_match. data is copied to matches_info object ..." << endl;

        matches_info.src_img_idx = features1.img_idx;
	    matches_info.dst_img_idx = features2.img_idx;

        matches_info.match1shape = match1shape;
        matches_info.match2shape = match2shape;

        matches_info.match1counts = match1counts;
        matches_info.match2counts = match2counts;

        matches_info.mscoreshape1 = mscoreshape1;
        matches_info.mscoreshape2 = mscoreshape2;

        matches_info.mscore1count = mscore1count;
        matches_info.mscore2count = mscore2count;

        // cout << endl << "B.B BLightGlue::perform_match. matches_info initialization is finished. matches_info.matches.size(): " << matches_info.matches.size();
        // cin.get();
        

    }

    // void BBLightGlue::perform_match (SELMSLAM::ImageFeatures features1, SELMSLAM::ImageFeatures features2, SELMSLAM::MatchesInfo &matches_info) {

    //     // creat an instance of the Ort::Env class with the name env used to create an execution environment for running models and managing resources related to inference tasks.
    //     // ORT_LOGGING_LEVEL_FATAL: only the most severe error messages should be logged.
    //     // BBLightGlue: identifier for the created environmnet
    //     Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "BBLightGlue");

    //     // configures session options for the Ort session, 
	//     // Including setting the number of threads for intra-operation parallelism and enabling graph optimization.
    //     Ort::SessionOptions sessionOptions;

    //     sessionOptions.SetIntraOpNumThreads(1); 
    //     sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    //     // Load the BBLightGlue network
    //     static Ort::Session lightGlueSession(env, this->m_modelPath.c_str(), sessionOptions);

    //     /**
    //      * 
    //      * Step 2.
    //      * Preprocessing
    //      * 
    //     */

    //     // processes keypoints and descriptors from features1 and features2 images,
    //     // converting them into appropriate data formats that can be used as inputs for the model. This includes resizing and reshaping the data.
    //     // The values in these two vectors are in the [0, 1] range (calculated in two following for statements). 
    //     vector<float> kp1;
    //     vector<float> kp2;

    //     // The size of these vectors is set to twice the number of keypoints in features1 and features2 because each keypoint has two components (x, y).
    //     kp1.resize(features1.keypoints.size() * 2);
    //     kp2.resize(features2.keypoints.size() * 2);

    //     /**
    //      * Centering the Coordinates: By dividing by half of the image's width and height, the code effectively centers the coordinates of each keypoint around the origin (0,0). 
    //      * This centering makes the coordinates relative to the center of the image, 
    //      * with the center being (0,0). This is a common normalization step in computer vision tasks.
    //      * 
    //      * Scaling the Coordinates: The division scales the coordinates such that they are in the range of -1 to 1. 
    //      * This scaling ensures that the coordinates are within a reasonable and consistent range for the neural network. 
    //      * Normalizing the data to a fixed range can help improve the training and inference process, especially when dealing with deep learning models.
    //     */
    //     float f1wid = features1.img_size.width / 2.0f;
    //     float f1hei = features1.img_size.height / 2.0f;

    //     for (int i = 0; i < features1.keypoints.size(); i++) {
    //         // calculates normalized (scaled) x and y coordinates for each keypoint.
    //         kp1[2 * i] = (features1.keypoints[i].pt.x - f1wid) / f1wid;
    //         kp1[2 * i + 1] = (features1.keypoints[i].pt.y - f1hei) / f1hei;
    //     }

    //     float f2wid = features2.img_size.width / 2.0f;
    //     float f2hei = features2.img_size.height / 2.0f;

    //     for (int i = 0; i < features2.keypoints.size(); i++) {
    //         kp2[2 * i] = (features2.keypoints[i].pt.x - f2wid) / f2wid;
    //         kp2[2 * i + 1] = (features2.keypoints[i].pt.y - f2hei) / f2hei;
    //     }

    //     // used to convert descriptors from features1 and features2 into a format suitable for input to the neural network model.
    //     vector<float> des1;
    //     vector<float> des2;

    //     des1.resize(features1.keypoints.size() * 256);
    //     des2.resize(features2.keypoints.size() * 256);

    //     // requesting read-only access to the data in features1.descriptors when creating the Mat object des1mat.
    //     cv::Mat des1mat = features1.descriptors.getMat(cv::ACCESS_READ);
    //     cv::Mat des2mat = features2.descriptors.getMat(cv::ACCESS_READ);

    //     // calculates an index in a 1D array that corresponds to a 2D matrix
    //     for (int w = 0; w < des1mat.cols; w++) {
    //         for (int h = 0; h < des1mat.rows; h++) {
    //             // By multiplying h with this value, the code moves down to the correct row.
    //             // Adding w to the result of the previous multiplication, it navigates to the correct column within that row.
    //             int index = h * features1.descriptors.cols + w;
    //             des1[index] = des1mat.at<float>(h, w);
    //         }
    //     }

    //     for (int w = 0; w < des2mat.cols; w++) {
    //         for (int h = 0; h < des2mat.rows; h++) {
    //             int index = h * features2.descriptors.cols + w;
    //             des2[index] = des2mat.at<float>(h, w);
    //         }
    //     }

    //     // creates input tensors for the Ort session using the processed data.
    //     const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};

    //     // !!! ACHTUNG ACHTUNG !!! GPU???
    //     // to define and manage memory information, including device type and memory type.
    //     // it's creating a memory info object for CPU memory.
    //     // OrtDeviceAllocator: device type indicates that CPU memory is used.
    //     // OrtMemTypeCPU: memory type = CPU memory
    //     Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    //     // This vector will store the input tensors for the ONNX session.
    //     vector<Ort::Value> inputTensor;

    //     // the shapes of the input tensors for keypoints 
    //     // defines the shape of the tensor to be created, specifies a 3D tensor with a shape of (1, number_of_keypoints, 2)
	//     // the tensor has one batch, and each element in the batch has two dimensions for each of the keypoints.
    //     vector<int64_t> kp1Shape{1, (int64_t)features1.keypoints.size(), 2};
    //     vector<int64_t> kp2Shape{1, (int64_t)features2.keypoints.size(), 2};

    //     // creates an Ort::Value object representing a tensor of type float.
    //     // memoryInfo: The memory information used for allocating memory for the tensor, 
    //     // kp1.data(): A pointer to the data for the tensor.
    //     // kp1.size(): The total number of elements in the data
    //     // kp1Shape.data(): A pointer to the shape information for the tensor.
    //     // kp1Shape.size(): The number of dimensions in the shape
    //     // emplace_back: The newly created tensor is then added to the inputTensor vector
    //     inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, kp1.data(), kp1.size(), kp1Shape.data(), kp1Shape.size()));
    //     inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, kp2.data(), kp2.size(), kp2Shape.data(), kp2Shape.size()));

    //     vector<int64_t> des1Shape{1, (int64_t)features1.keypoints.size(), features1.descriptors.cols};
    //     vector<int64_t> des2Shape{1, (int64_t)features2.keypoints.size(), features2.descriptors.cols};

    //     inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, des1.data(), des1.size(), des1Shape.data(), des1Shape.size()));
    //     inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, des2.data(), des2.size(), des2Shape.data(), des2Shape.size()));

    //     // runs the Ort session with the input tensors and retrieves output tensors for matches and scores.
    //     const char* output_names[] = {"matches0", "matches1", "mscores0", "mscores1"};

    //     // options and settings for the inference run. It can include parameters such as timeout settings, execution providers
    //     Ort::RunOptions run_options;

    //     // outputs: data generated by the model during inference.
    //     // 4: The number of input names (input node count).
    //     // output_names: the names of the model's output nodes. these names identify where to retrieve the output data.
    //     // 4: The number of output names (output node count)
    //     // Run: takes the input data, executes the model, and stores the results in the outputs vector.
    //     vector<Ort::Value> outputs = lightGlueSession.Run(run_options, input_names, inputTensor.data(), 4, output_names, 4);

    //     // Process output tensors to obtain information about matches and their scores.
    //     std::vector<int64_t> match1shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    //     int64_t* match1 = (int64_t*)outputs[0].GetTensorMutableData<void>();
    //     int match1counts = match1shape[1]; // The number of matches

    //     std::vector<int64_t> mscoreshape1 = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	//     float* mscore1 = (float*)outputs[2].GetTensorMutableData<void>();
    //     int mscore1count = mscoreshape1[1];

    //     std::vector<int64_t> match2shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    //     int64_t* match2 = (int64_t*)outputs[1].GetTensorMutableData<void>();
    //     int match2counts = match2shape[1];

    //     std::vector<int64_t> mscoreshape2 = outputs[3].GetTensorTypeAndShapeInfo().GetShape();
    //     float* mscore2 = (float*)outputs[3].GetTensorMutableData<void>();
    //     int mscore2count = mscoreshape2[1];

    //     // Populate matches_info structure with the computed matches, source and destination image indices, and other information.
    //     matches_info.src_img_idx = features1.img_idx;
	//     matches_info.dst_img_idx = features2.img_idx;

    //     matches_info.match1 = match1;
    //     matches_info.match2 = match2;

    //     matches_info.match1shape = match1shape;
    //     matches_info.match2shape = match2shape;

    //     matches_info.match1counts = match1counts;
    //     matches_info.match2counts = match2counts;

    //     matches_info.mscoreshape1 = mscoreshape1;
    //     matches_info.mscoreshape2 = mscoreshape2;

    //     matches_info.mscore1count = mscore1count;
    //     matches_info.mscore2count = mscore2count;

    //     matches_info.mscore1 = mscore1;
    //     matches_info.mscore2 = mscore2;

    // }
} // namespace