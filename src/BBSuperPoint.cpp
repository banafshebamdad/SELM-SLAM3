/*
 * Author: Banafshe Bamdad
 * Created on Mon Oct 16 2023 11:25:45 CET
 *
 */

#include "BBLightGlue.hpp"
#include "BBSuperPoint.hpp"

#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <string>

#define WITH_TICTOC
    #include <tictoc.hpp>
#define WITH_TICTOC

namespace SELMSLAM {

    using namespace std;

    struct ScoreIndex {
        float value;
        int index;
    };

    BBSuperPoint::BBSuperPoint(std::string modelPath) {
        this->m_modelPath = modelPath;
    }

    /**
     * takes an OpenCV Mat object image as input and converts it to a grayscale float image.
     * The pixel values are then normalized to the range [0, 1] and stored in a vector<float> named imgData.
    */
    vector<float> BBSuperPoint::preprocessImage(const cv::Mat& image, float& mean, float& std) {

        cv::Mat resozed, floatImage;

        // target data type being 32-bit floating-point, single channel (grayscale). (if input pxs are 8-bit unsigned integers (CV_8U))
        image.convertTo(floatImage, CV_32FC1);

        vector<float> imgData;
        for (int h = 0; h < image.rows; h++) {
            for (int w = 0; w < image.cols; w++) {
                imgData.push_back(floatImage.at<float>(h, w) / 255.0f);
            }
        }
        return imgData;
    }
    /**
     * @todo Mo Okt. 16, 2023 13:28
     * I assume the input image is in grayscale. Convert it first to grayscale if it's not.
    */

/*************************************************/
    // Run on CPU
    void BBSuperPoint::featureExtractor(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray& descriptors) {

        // @todo Fr. Dez. 8, 2023 read from setting file
        int nFeatures = 1200; // 200 500, 750, 900, 1000, 1200

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "BBSuperPoint");

        Ort::SessionOptions sessionOptions;

        sessionOptions.SetIntraOpNumThreads(1); // !!! ACHTUNG !!!  check whether is required for GPU execusion
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        static Ort::Session extractorSession(env, this->m_modelPath.c_str(), sessionOptions);

        cv::Mat img = image.getMat();

        cv::Mat grayImg = img; // @todo

        float mean, std;

        vector<float> imgData = preprocessImage(grayImg, mean, std);

        // single data point (sample), single channel (depth)
        vector<int64_t> inputShape{ 1, 1, grayImg.rows, grayImg.cols };

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // The preprocessed image data is loaded into an ONNX tensor.
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

        const char* input_names[] = { "image" };
        const char* output_names[] = { "keypoints", "scores", "descriptors" };
        Ort::RunOptions run_options;

        // The SuperPoint model is run using the ONNX Runtime, and keypoints and descriptors are extracted from the model's output tensors.

        cout << endl << "B.B In BBSuperPoint::featureExtractor. Before running SuperPoint model." << endl;
        TIC
        std::vector<Ort::Value> outputs = extractorSession.Run(run_options, input_names, &inputTensor, 1, output_names, 3);
        TOC

        cout << endl << "B.B In BBSuperPoint::featureExtractor. After running SuperPoint model." << endl;

        std::vector<int64_t> kpshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int64* kp = (int64*)outputs[0].GetTensorMutableData<void>();
        int keypntcounts = kpshape[1];

        // To extract only nFeature from the frame, uncomment the following commented code and comment the following line.
        nFeatures = keypntcounts;
        // if (keypntcounts < nFeatures) {
        //     nFeatures = keypntcounts;
        // }

        // keypoints.resize(keypntcounts);
        keypoints.resize(nFeatures);

        float* scores = (float*)outputs[1].GetTensorMutableData<void>();
        std::vector<ScoreIndex> scoreIndices;

        cout << "Scores: " << endl << "--------";
        for (int i = 0; i < keypntcounts; i++) {
            ScoreIndex si;
            si.value = scores[i];
            si.index = i;
            scoreIndices.push_back(si);
            // cout << endl << i << ". score: " <<  scores[i];
        }

        // lambda function
        std::sort(scoreIndices.begin(), scoreIndices.end(), [](const ScoreIndex& a, const ScoreIndex& b) {
            return a.value > b.value;
        });

        // cout << endl << "Sorted Scores: " << endl << "--------" << endl;
        // for (const ScoreIndex& si : scoreIndices) {
        //     std::cout << "Index: " << si.index << ", Sorted Value: " << si.value << std::endl;
        // }

        // keypoints

        // Retrieve nFeatures features with higher scores
        for (int j = 0; j < nFeatures; j++) {
            
            ScoreIndex si = scoreIndices[j];
            
            cv::KeyPoint p;
            int index = si.index * 2;
            p.pt.x = (float)kp[index];
            p.pt.y = (float)kp[index + 1];
            keypoints[j] = p;
        }

        // for (int i = 0; i < keypntcounts; i++) {
        // for (int i = 0; i < nFeatures; i++) {
        //     cv::KeyPoint p;
        //     int index = i * 2;
        //     p.pt.x = (float)kp[index];
        //     p.pt.y = (float)kp[index + 1];
        //     keypoints[i] = p;
        // }
        
        std::vector<int64_t> desshape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	    float* des = (float*)outputs[2].GetTensorMutableData<void>();

        // B.B descriptors
        cv::Mat desmat = descriptors.getMat();
        if (desshape[1] < nFeatures) {
            nFeatures = desshape[1];
        }
        desmat.create(cv::Size(desshape[2], nFeatures), CV_32FC1);
        for (int h = 0; h < nFeatures; h++) {
            for (int w = 0; w < desshape[2]; w++) {
                ScoreIndex si = scoreIndices[h];
                int index = desshape[2] * si.index + w;
                desmat.at<float>(h, w) = des[index];
            }
        }
        // for (int h = 0; h < nFeatures; h++) {
        //     for (int w = 0; w < desshape[2]; w++) {
        //         int index = h * desshape[2] + w;
        //         desmat.at<float>(h, w) = des[index];
        //     }
        // }


        // cv::Mat desmat = descriptors.getMat();
        // desmat.create(cv::Size(desshape[2], desshape[1]), CV_32FC1);
        // for (int h = 0; h < desshape[1]; h++) {
        //     for (int w = 0; w < desshape[2]; w++) {
        //         int index = h * desshape[2] + w;
        //         desmat.at<float>(h, w) = des[index];
        //     }
        // }
        desmat.copyTo(descriptors);
    }
/*****************************/
    // Run on GPU
/***************************************************
    void BBSuperPoint::featureExtractor(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray& descriptors) {


        cv::Mat img = image.getMat();

        cv::Mat grayImg = img; // @todo

        float mean, std;

        vector<float> imgData = preprocessImage(grayImg, mean, std);

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

        // @todo Fr. Dez. 8, 2023 read from setting file
        int nFeatures = 1200; // 200 500, 750, 900, 1000, 1200

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "BBSuperPoint");

        Ort::SessionOptions sessionOptions;
        api.SessionOptionsAppendExecutionProvider_CUDA_V2(sessionOptions, cuda_options);

        // 
        // Load the BBSuperPoint network
        // 

        static Ort::Session session(env, this->m_modelPath.c_str(), sessionOptions);
        Ort::MemoryInfo memoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
        Ort::Allocator cuda_allocator(session, memoryInfo);

        const char* input_names[] = { "image" };
        const char* output_names[] = { "keypoints", "scores", "descriptors" };

        // single data point (sample), single channel (depth)
        vector<int64_t> inputShape{ 1, 1, grayImg.rows, grayImg.cols };

        Ort::IoBinding io_binding(session);

        auto input_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(imgData.size() * sizeof(float)), CudaMemoryDeleter(&cuda_allocator));
        cudaMemcpy(input_data.get(), imgData.data(), sizeof(float) * imgData.size(), cudaMemcpyHostToDevice);

        // Create an OrtValue tensor backed by data on CUDA memory
        // reinterpret_cast<float*>: a type-casting operation used to interpret the raw pointer as a pointer to float. CreateTensor function expects a float* pointer.
        Ort::Value bound_x = Ort::Value::CreateTensor(memoryInfo, reinterpret_cast<float*>(input_data.get()), imgData.size(), inputShape.data(), inputShape.size());
        io_binding.BindInput("image", bound_x);

        // 
        // output binding
        // 

        Ort::MemoryInfo output_mem_info{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};


        io_binding.BindOutput(output_names[0], output_mem_info);
        io_binding.BindOutput(output_names[1], output_mem_info);
        io_binding.BindOutput(output_names[2], output_mem_info);

        // Run the model (executing the graph)

        TIC
        session.Run(Ort::RunOptions(), io_binding);
        TOC

        vector<Ort::Value> outputs = io_binding.GetOutputValues();

        // Allocate host memory for the output tensors
        std::vector<int64_t> kpshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> scoresshape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> desshape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();

        int keypntcounts = kpshape[1];

        // To extract only nFeature from the frame, uncomment the following commented code and comment the following line.
        nFeatures = keypntcounts;
        // if (keypntcounts < nFeatures) {
        //     nFeatures = keypntcounts;
        // }

        // keypoints.resize(keypntcounts);
        keypoints.resize(nFeatures);

        // Use std::vector instead of dynamic arrays
        std::vector<int64_t> kp_host(kpshape[0] * kpshape[1]);
        std::vector<float> scores_host(scoresshape[0] * scoresshape[1]);
        std::vector<float> des_host(desshape[0] * desshape[1]);

        // Copy data from GPU to CPU
        int64_t* kp = (int64_t*)outputs[0].GetTensorMutableData<void>();
        float* scores = (float*)outputs[1].GetTensorMutableData<void>();
        float* des = (float*)outputs[2].GetTensorMutableData<void>();

        cudaMemcpy(kp_host.data(), kp, sizeof(int64_t) * kpshape[0] * kpshape[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(scores_host.data(), scores, sizeof(float) * scoresshape[0] * scoresshape[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(des_host.data(), des, sizeof(float) * desshape[0] * desshape[1], cudaMemcpyDeviceToHost);

        std::vector<ScoreIndex> scoreIndices;

        cout << "Scores: " << endl << "--------";
        for (int i = 0; i < keypntcounts; i++) {
            ScoreIndex si;
            // si.value = scores[i];
            si.value = scores_host.data()[i];
            si.index = i;
            scoreIndices.push_back(si);
            // cout << endl << i << ". score: " <<  scores[i];
        }

        // lambda function
        std::sort(scoreIndices.begin(), scoreIndices.end(), [](const ScoreIndex& a, const ScoreIndex& b) {
            return a.value > b.value;
        });

        // cout << endl << "Sorted Scores: " << endl << "--------" << endl;
        // for (const ScoreIndex& si : scoreIndices) {
        //     std::cout << "Index: " << si.index << ", Sorted Value: " << si.value << std::endl;
        // }

        // keypoints

        // Retrieve nFeatures features with higher scores
        for (int j = 0; j < nFeatures; j++) {
            
            ScoreIndex si = scoreIndices[j];
            
            cv::KeyPoint p;
            int index = si.index * 2;
            // p.pt.x = (float)kp[index];
            // p.pt.y = (float)kp[index + 1];
            p.pt.x = (float)kp_host.data()[index];
            p.pt.y = (float)kp_host.data()[index + 1];
            keypoints[j] = p;
        }

        // for (int i = 0; i < keypntcounts; i++) {
        // for (int i = 0; i < nFeatures; i++) {
        //     cv::KeyPoint p;
        //     int index = i * 2;
        //     p.pt.x = (float)kp[index];
        //     p.pt.y = (float)kp[index + 1];
        //     keypoints[i] = p;
        // }
        
        // std::vector<int64_t> desshape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	    // float* des = (float*)outputs[2].GetTensorMutableData<void>();

        // B.B descriptors
        cv::Mat desmat = descriptors.getMat();
        if (desshape[1] < nFeatures) {
            nFeatures = desshape[1];
        }
        desmat.create(cv::Size(desshape[2], nFeatures), CV_32FC1);
        for (int h = 0; h < nFeatures; h++) {
            for (int w = 0; w < desshape[2]; w++) {
                ScoreIndex si = scoreIndices[h];
                int index = desshape[2] * si.index + w;
                // desmat.at<float>(h, w) = des[index];
                desmat.at<float>(h, w) = des_host.data()[index];
            }
        }
        // for (int h = 0; h < nFeatures; h++) {
        //     for (int w = 0; w < desshape[2]; w++) {
        //         int index = h * desshape[2] + w;
        //         desmat.at<float>(h, w) = des[index];
        //     }
        // }


        // cv::Mat desmat = descriptors.getMat();
        // desmat.create(cv::Size(desshape[2], desshape[1]), CV_32FC1);
        // for (int h = 0; h < desshape[1]; h++) {
        //     for (int w = 0; w < desshape[2]; w++) {
        //         int index = h * desshape[2] + w;
        //         desmat.at<float>(h, w) = des[index];
        //     }
        // }
        desmat.copyTo(descriptors);
    }

    *************************************/
} // namespace
