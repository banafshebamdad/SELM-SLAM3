# SUPERSLAM3

SUPERSLAM3 is a visual odometry pipeline that combines the Superpoint-based front end with the ORB-SLAM3 pose estimation backend. In the implemented pipeline, traditional ORB and BRIEF feature detection and description methods have been replaced with the Superpoint pipeline.

<img src="https://github.com/isarlab-department-engineering/SUPERSLAM3/assets/12141143/ffc7a731-ff20-4681-9eb7-fc004fdca6ec" alt="SUPERSLAM3 pipeline" width="900px"/>

In the SUPERSLAM3 pipeline, input images are converted to grayscale and fed into the Superpoint detector pipeline (A). The Superpoint encoder-decoder pipeline consists of a learned encoder, utilizing several convolutional layers, and two non-learned decoders for joint feature and descriptor extraction. The detected features are then processed by the ORB-SLAM3 backend, which comprises three primary components operating in parallel threads: the Tracking, Local Mapping, and Loop & Map Merging threads (B). The backend extracts keyframes, initializes and updates the map, and performs both local and global motion and pose estimation within the Local Mapping Thread and Loop & Map Merging thread. If a loop closure is detected, the pose estimation is further refined.

This repository was forked from [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3). The pre-trained model of SuperPoint come from the official [MagicLeap 
repository](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).

<!-- The [Changelog](https://https://github.com/isarlab-department-engineering/SUPERSLAM3/edit/main/README.md) describes the features of each version. -->
<!--***6, September 2023 update***</br> 
- REPOSITORY UNDER CONSTRUCTION: We are in the process of uploading and building the GitHub repository.  </br> -->

***7, September 2023 update***</br> 
- SUPERSLAM3 v1.0 is now publicly available!  </br>
- REPOSITORY UNDER CONSTRUCTION: We are in the process of uploading and building the GitHub repository.
- We are currently testing the project on Ubuntu 20.04 and 22.02 with upgraded CUDA, CuDNN, and libtorch libraries. </br>

## Related Publications:
[ORB-SLAM3] Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel and Juan D. Tardós, **ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM**, *IEEE Transactions on Robotics 37(6):1874-1890, Dec. 2021*. **[PDF](https://arxiv.org/abs/2007.11898)**.

[Superpoint] DeTone, Daniel, Tomasz Malisiewicz, and Andrew Rabinovich. **Superpoint: Self-supervised interest point detection and description.** Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2018, **[PDF](https://arxiv.org/abs/1712.07629v4)**.

[DBoW2 Place Recognition] Dorian Gálvez-López and Juan D. Tardós. **Bags of Binary Words for Fast Place Recognition in Image Sequences**. *IEEE Transactions on Robotics,* vol. 28, no. 5, pp. 1188-1197, 2012, **[PDF](http://doriangalvez.com/php/dl.php?dlp=GalvezTRO12.pdf)**.

[SuperPoint-SLAM] Deng, Chengqi, et al. **Comparative study of deep learning based features in SLAM.** 2019 4th Asia-Pacific Conference on Intelligent Robot Systems (ACIRS). IEEE, 2019, **[PDF](https://ieeexplore.ieee.org/abstract/document/8935995?casa_token=sEqTDfOsz-4AAAAA:WWFr0OAw4lb8yZfJs1EVUcIoOjEMFru0Re2iKnkDRoDTXnhflEYxkMt63iXHb-TwVrZ1D3zB)**.

### Citing:
**If you use SUPERSLAM3 in an academic work, please cite:**
```
  @article{mollica2023integrating,
    title={Integrating Sparse Learning-Based Feature Detectors into Simultaneous Localization and Mapping—A Benchmark Study},
    author={Mollica, Giuseppe and Legittimo, Marco and Dionigi, Alberto and Costante, Gabriele and Valigi, Paolo},
    journal={Sensors},
    volume={23},
    number={4},
    pages={2286},
    year={2023},
    publisher={MDPI}
  }
```
# 1. Prerequisites
We have tested the libraries and executables on **Ubuntu 18.04**.

## C++11 or C++0x Compiler
ORBSLAM3 uses the new thread and chrono functionalities of C++11.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at least 3.0. Tested with OpenCV 3.4.11**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0. Tested with Eigen3 3.4.0**.

## DBoW3, DBoW2, Pangolin and g2o (Included in Thirdparty folder)
We use a BOW vocabulary based on the [BOW3](https://github.com/rmsalinas/DBow3) library to perform place recognition, and [g2o](https://github.com/RainerKuemmerle/g2o) library is used to perform non-linear optimizations. All these libraries are included in the *Thirdparty* folder.

### Download Vocabulary
The DBOW3 vocabulary can be downloaded from [google drive](https://drive.google.com/file/d/1p1QEXTDYsbpid5ELp3IApQ8PGgm_vguC/view?usp=sharing). Place the uncompressed vocabulary file into the `Vocabulary` directory within the SUPERSLAM3 project. 
For more informations please refer to [this repo](https://github.com/KinglittleQ/SuperPoint_SLAM/tree/master).

## Nvidia-driver & Cuda Toolkit 10.2 with cuDNN 7.6.5
Please, follow these [instructions](https://developer.nvidia.com/cuda-10.2-download-archive) for the installation of the Cuda Toolkit 10.2.

If not installed during the Cuda Toolkit installation process, please install the nvidia driver 440:
``` shell
sudo apt-get install nvidia-driver-440
```

Export Cuda paths 
``` shell
echo 'export PATH=/usr/local/cuda-10.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig  
```

Verify the Nvidia driver availability:
``` shell
nvidia-smi
```

Download [CuDNN 7.6.5](https://developer.download.nvidia.com/compute/redist/cudnn/v7.6.5/cudnn-10.2-linux-x64-v7.6.5.32.tgz) from the official NVidia page, and install the headers and libraries in the local CUDA installation folder:

``` shell
sudo cp -P <PATH_TO_CUDNN_FOLDER>/cuda/include/cudnn.h <PATH_TO_CUDA10.1_FOLDER>/include/
sudo cp -P <PATH_TO_CUDNN_FOLDER>/cuda/lib64/libcudnn* <PATH_TO_CUDA10.1_FOLDER>/lib64/
sudo chmod a+r <PATH_TO_CUDA10.2_FOLDER>/lib64/libcudnn*	
```
<!-- sudo cp -P <PATH_TO_CUDNN_FOLDER>/cuda/include/cudnn_version.h <PATH_TO_CUDA10.1_FOLDER>/include/ -->

The CUDA installation can be verified by running:
``` shell
nvcc -V
```

## LibTorch 1.6.0 version (with GPU | Cuda Toolkit 10.2, cuDNN 7.6.5)
If only CPU can be used, install cpu-version LibTorch. Some code change about tensor device should be required.

```shell
wget -O LibTorch.zip https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip
sudo unzip LibTorch.zip -d /usr/local
```

# 2. Building SUPERSLAM3 library and examples
Clone the repository:
```shell
git clone --recursive https://github.com/isarlab-department-engineering/SUPERSLAM3
```

Use the provided script `build.sh` to build the *Thirdparty* libraries and *SUPERSLAM3* project. Please make sure you have **installed all required dependencies** (see section 1). 

Open the build.sh file and modify the weights file path according to the absolute path of the weights file in your PC (<PATH_TO_SUPERSLAM3_FOLDER>/Weights/superpoint.pt)

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DSUPERPOINT_WEIGHTS_PATH="<PATH_TO_SUPERSLAM3_FOLDER>/Weights/superpoint.pt"
```

Build the project:

```shell
cd <PATH_TO_SUPERSLAM3_FOLDER>
chmod +x build.sh
./build.sh
```

# 3. Monocular Examples
## EUROC Dataset
To test SUPERSLAM3 with the EUROC dataset:

1) Download the MH01 sequence (ASL Dataset format) from [this link](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets#downloads).

2) Unzip the downloaded sequence and execute the following command. Change PATH_TO_MH01_SEQUENCE_FOLDER to the uncompressed dataset folder.

```shell
cd <PATH_TO_SUPERSLAM3_FOLDER>
./Examples/Monocular/mono_euroc ./Vocabulary/superpoint_voc.yml ./Examples/Monocular/EuRoC.yaml <PATH_TO_MH01_SEQUENCE_FOLDER> ./Examples/Monocular/EuRoC_TimeStamps/MH01.txt
```

