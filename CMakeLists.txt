
cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM3)

### Start 1: SuperSLAM config

# Get cuDNN version
message("1. ::: Banafshe is asking for the path to cudnn :::")
message("${CUDNN_INCLUDE_PATH}")
message("--- 1. B.B ---")

if(EXISTS ${CUDNN_INCLUDE_PATH}/cudnn_version.h)
        file(READ ${CUDNN_INCLUDE_PATH}/cudnn_version.h CUDNN_HEADER_CONTENTS)
else()
        file(READ ${CUDNN_INCLUDE_PATH}/cudnn.h CUDNN_HEADER_CONTENTS)
endif()

### LIBTORCH
#set(Torch_DIR "/home/ubuntu/pytorch_master/pytorch/torch/share/cmake/Torch")  # libtorch

# B.B modifies the CMAKE_PREFIX_PATH variable by appending the path "/usr/local/libtorch/" to its current value.
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/libtorch/")


#find_package(Torch REQUIRED QUIET)
find_package(Torch REQUIRED)

message("2. ::: Banafshe is asking for the TORCH_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1 :::")
# This flag is often automatically set by CMake when you use find_package(Torch REQUIRED)
message("${TORCH_CXX_FLAGS}")
message("--- 2. B.B ---")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# include_directories(${TORCH_INCLUDE_DIRS})
###

### End 1: SuperSLAM config


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

message("3. ::: Banafshe is asking for the PROJECT_SOURCE_DIR=/home/banafshe/SUPERSLAM3 :::")
message("${PROJECT_SOURCE_DIR}")
message("--- 3. B.B ---")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.2.0)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.2.0 not found.") # B.B > 4.2.0 ???
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
#find_package(realsense2)


# Added by B.B
# find_package(Python 3.8 EXACT REQUIRED)
## find_package(PythonLibs REQUIRED)
## include_directories(${PYTHON_INCLUDE_DIRS})

# Added by Banafshe
message("::: Banafshe is asking for the path to ONNXRineTime :::")
message("${ONNX_RUNTIME_PATH}")
message("--- B.B ---")
#find_package(onnxruntime REQUIRED)
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNX_RUNTIME_PATH}/include")

message("4. ::: Banafshe is asking for nclude directories. :::")
message("\tEIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}")
message("\tPangolin_INCLUDE_DIRS=${Pangolin_INCLUDE_DIRS}")
message("\tTORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")
message("--- 4. B.B ---")

# B.B to specify directories where the CMake build system should look for header files when compiling C++ code.
# ${ONNXRUNTIME_INCLUDE_DIRS}, ${CUDA_INCLUDE_PATH} added by Banafshe Bamdad
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${PROJECT_SOURCE_DIR}/Thirdparty/DUtils
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${TORCH_INCLUDE_DIRS}
${ONNXRUNTIME_INCLUDE_DIRS}
${CUDA_INCLUDE_PATH}
#${TORCH_INCLUDES_2}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h

include/Defs.h

src/SPDetector.cpp
src/SuperPoint.cpp
include/SPDetector.hpp
include/SuperPoint.hpp

src/BBLightGlue.cpp
include/BBLightGlue.hpp

src/BBSuperPoint.cpp
include/BBSuperPoint.hpp

src/BBSPExtractor.cpp
include/BBSPExtractor.hpp

src/BBLGMatcher.cpp
include/BBLGMatcher.hpp

src/BBLogger.cpp
include/BBLogger.hpp
)

add_subdirectory(Thirdparty/g2o)

if(DEFINED SUPERPOINT_WEIGHTS_PATH)
    message("Macro SUPERPOINT_WEIGHTS_PATH is defined with value: ${SUPERPOINT_WEIGHTS_PATH}")
    target_compile_definitions(${PROJECT_NAME} PRIVATE WEIGHTS_PATH="${SUPERPOINT_WEIGHTS_PATH}")
endif()

# Banafshe bamdad
if(DEFINED BBSUPERPOINT_WEIGHT_PATH)
    message("Macro BBSUPERPOINT_WEIGHT_PATH is defined with value: ${BBSUPERPOINT_WEIGHT_PATH}")
   #  to make a value from CMake's configuration available to C++ source code as a preprocessor macro
    target_compile_definitions(${PROJECT_NAME} PRIVATE BBSUPERPOINT_WEIGHT_PATH="${BBSUPERPOINT_WEIGHT_PATH}")
endif()

# Banafshe bamdad
if(DEFINED BBLIGHTGLUE_WEIGHT_PATH)
    message("Macro BBLIGHTGLUE_WEIGHT_PATH is defined with value: ${BBLIGHTGLUE_WEIGHT_PATH}")
   #  to make a value from CMake's configuration available to C++ source code as a preprocessor macro
    target_compile_definitions(${PROJECT_NAME} PRIVATE BBLIGHTGLUE_WEIGHT_PATH="${BBLIGHTGLUE_WEIGHT_PATH}")
endif()

# Banafshe bamdad
if(DEFINED BBLOGFILE_PATH)
    message("Macro BBLOGFILE_PATH is defined with value: ${BBLOGFILE_PATH}")
   #  to make a value from CMake's configuration available to C++ source code as a preprocessor macro
    target_compile_definitions(${PROJECT_NAME} PRIVATE BBLOGFILE_PATH="${BBLOGFILE_PATH}")
endif()

# B.B added ${PYTHON_LIBRARIES}
# B.B added ${PROJECT_SOURCE_DIR}/Thirdparty/onnxruntime-linux-x64-gpu-1.16.1/libonnxruntime.so
target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/DBow3/build/src/libDBoW3.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/Thirdparty/onnxruntime-linux-x64-gpu-1.16.1/lib/libonnxruntime.so
-lboost_serialization
-lcrypto
${TORCH_LIBRARIES}
)

# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
    ${realsense_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME}
    ${realsense2_LIBRARY}
    )
endif()


# Build examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            Examples/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()


# RGB-D inertial examples
# Commented by B.B
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D-Inertial)

# if(realsense2_FOUND)
#     add_executable(rgbd_inertial_realsense_D435i
#             Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
#     target_link_libraries(rgbd_inertial_realsense_D435i ${PROJECT_NAME})
# endif()

#Stereo examples
# Commented by B.B
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

# add_executable(stereo_kitti
#         Examples/Stereo/stereo_kitti.cc)
# target_link_libraries(stereo_kitti ${PROJECT_NAME})

# add_executable(stereo_euroc
#         Examples/Stereo/stereo_euroc.cc)
# target_link_libraries(stereo_euroc ${PROJECT_NAME})

# add_executable(stereo_tum_vi
#         Examples/Stereo/stereo_tum_vi.cc)
# target_link_libraries(stereo_tum_vi ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(stereo_realsense_t265
#             Examples/Stereo/stereo_realsense_t265.cc)
#     target_link_libraries(stereo_realsense_t265 ${PROJECT_NAME})

#     add_executable(stereo_realsense_D435i
#             Examples/Stereo/stereo_realsense_D435i.cc)
#     target_link_libraries(stereo_realsense_D435i ${PROJECT_NAME})
# endif()

# Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

# Added by Banafshe Bamdad
add_executable(mono_selm Examples/Monocular/mono_selm.cc)
target_link_libraries(mono_selm ${PROJECT_NAME}) 

add_executable(mono_Incrowd-VI Examples/Monocular/mono_Incrowd-VI.cc)
target_link_libraries(mono_Incrowd-VI ${PROJECT_NAME})

# Commented by B.B
# add_executable(mono_tum
#         Examples/Monocular/mono_tum.cc)
# target_link_libraries(mono_tum ${PROJECT_NAME})

# add_executable(mono_kitti
#         Examples/Monocular/mono_kitti.cc)
# target_link_libraries(mono_kitti ${PROJECT_NAME})

# add_executable(mono_euroc
#         Examples/Monocular/mono_euroc.cc)
# target_link_libraries(mono_euroc ${PROJECT_NAME})

# add_executable(mono_tum_vi
#         Examples/Monocular/mono_tum_vi.cc)
# target_link_libraries(mono_tum_vi ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(mono_realsense_t265
#             Examples/Monocular/mono_realsense_t265.cc)
#     target_link_libraries(mono_realsense_t265 ${PROJECT_NAME})

#     add_executable(mono_realsense_D435i
#             Examples/Monocular/mono_realsense_D435i.cc)
#     target_link_libraries(mono_realsense_D435i ${PROJECT_NAME})
# endif()

#Monocular inertial examples
# Commented by B.B
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

# add_executable(mono_inertial_euroc
#         Examples/Monocular-Inertial/mono_inertial_euroc.cc)
# target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

# add_executable(mono_inertial_tum_vi
#         Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
# target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(mono_inertial_realsense_t265
#             Examples/Monocular-Inertial/mono_inertial_realsense_t265.cc)
#     target_link_libraries(mono_inertial_realsense_t265 ${PROJECT_NAME})

#     add_executable(mono_inertial_realsense_D435i
#             Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
#     target_link_libraries(mono_inertial_realsense_D435i ${PROJECT_NAME})
# endif()

#Stereo Inertial examples
# Commented by B.B
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)

# add_executable(stereo_inertial_euroc
#         Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
# target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})

# add_executable(stereo_inertial_tum_vi
#         Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
# target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(stereo_inertial_realsense_t265
#             Examples/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
#     target_link_libraries(stereo_inertial_realsense_t265 ${PROJECT_NAME})

#     add_executable(stereo_inertial_realsense_D435i
#             Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
#     target_link_libraries(stereo_inertial_realsense_D435i ${PROJECT_NAME})
# endif()

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Calibration)
# if(realsense2_FOUND)
#     add_executable(recorder_realsense_D435i
#             Examples/Calibration/recorder_realsense_D435i.cc)
#     target_link_libraries(recorder_realsense_D435i ${PROJECT_NAME})

#     add_executable(recorder_realsense_T265
#             Examples/Calibration/recorder_realsense_T265.cc)
#     target_link_libraries(recorder_realsense_T265 ${PROJECT_NAME})
# endif()

#Old examples

# RGB-D examples
# Commented by B.B
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D)

# add_executable(rgbd_tum_old
#         Examples_old/RGB-D/rgbd_tum.cc)
# target_link_libraries(rgbd_tum_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(rgbd_realsense_D435i_old
#             Examples_old/RGB-D/rgbd_realsense_D435i.cc)
#     target_link_libraries(rgbd_realsense_D435i_old ${PROJECT_NAME})
# endif()


# RGB-D inertial examples
# Commented by B.B
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D-Inertial)

# if(realsense2_FOUND)
#     add_executable(rgbd_inertial_realsense_D435i_old
#             Examples_old/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
#     target_link_libraries(rgbd_inertial_realsense_D435i_old ${PROJECT_NAME})
# endif()

#Stereo examples
# Commented by B.B
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo)

# add_executable(stereo_kitti_old
#         Examples_old/Stereo/stereo_kitti.cc)
# target_link_libraries(stereo_kitti_old ${PROJECT_NAME})

# add_executable(stereo_euroc_old
#         Examples_old/Stereo/stereo_euroc.cc)
# target_link_libraries(stereo_euroc_old ${PROJECT_NAME})

# add_executable(stereo_tum_vi_old
#         Examples_old/Stereo/stereo_tum_vi.cc)
# target_link_libraries(stereo_tum_vi_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(stereo_realsense_t265_old
#             Examples_old/Stereo/stereo_realsense_t265.cc)
#     target_link_libraries(stereo_realsense_t265_old ${PROJECT_NAME})

#     add_executable(stereo_realsense_D435i_old
#             Examples_old/Stereo/stereo_realsense_D435i.cc)
#     target_link_libraries(stereo_realsense_D435i_old ${PROJECT_NAME})
# endif()

#Monocular examples
# Commented by B.B
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular)

# add_executable(mono_tum_old
#         Examples_old/Monocular/mono_tum.cc)
# target_link_libraries(mono_tum_old ${PROJECT_NAME})

# add_executable(mono_kitti_old
#         Examples_old/Monocular/mono_kitti.cc)
# target_link_libraries(mono_kitti_old ${PROJECT_NAME})

# add_executable(mono_euroc_old
#         Examples_old/Monocular/mono_euroc.cc)
# target_link_libraries(mono_euroc_old ${PROJECT_NAME})

# add_executable(mono_tum_vi_old
#         Examples_old/Monocular/mono_tum_vi.cc)
# target_link_libraries(mono_tum_vi_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(mono_realsense_t265_old
#             Examples_old/Monocular/mono_realsense_t265.cc)
#     target_link_libraries(mono_realsense_t265_old ${PROJECT_NAME})

#     add_executable(mono_realsense_D435i_old
#             Examples_old/Monocular/mono_realsense_D435i.cc)
#     target_link_libraries(mono_realsense_D435i_old ${PROJECT_NAME})
# endif()

#Monocular inertial examples
# Commented by B.B
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular-Inertial)

# add_executable(mono_inertial_euroc_old
#         Examples_old/Monocular-Inertial/mono_inertial_euroc.cc)
# target_link_libraries(mono_inertial_euroc_old ${PROJECT_NAME})

# add_executable(mono_inertial_tum_vi_old
#         Examples_old/Monocular-Inertial/mono_inertial_tum_vi.cc)
# target_link_libraries(mono_inertial_tum_vi_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(mono_inertial_realsense_t265_old
#             Examples_old/Monocular-Inertial/mono_inertial_realsense_t265.cc)
#     target_link_libraries(mono_inertial_realsense_t265_old ${PROJECT_NAME})

#     add_executable(mono_inertial_realsense_D435i_old
#             Examples_old/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
#     target_link_libraries(mono_inertial_realsense_D435i_old ${PROJECT_NAME})
# endif()

#Stereo Inertial examples
# Commented by B.B
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo-Inertial)

# add_executable(stereo_inertial_euroc_old
#         Examples_old/Stereo-Inertial/stereo_inertial_euroc.cc)
# target_link_libraries(stereo_inertial_euroc_old ${PROJECT_NAME})

# add_executable(stereo_inertial_tum_vi_old
#         Examples_old/Stereo-Inertial/stereo_inertial_tum_vi.cc)
# target_link_libraries(stereo_inertial_tum_vi_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(stereo_inertial_realsense_t265_old
#             Examples_old/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
#     target_link_libraries(stereo_inertial_realsense_t265_old ${PROJECT_NAME})

#     add_executable(stereo_inertial_realsense_D435i_old
#             Examples_old/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
#     target_link_libraries(stereo_inertial_realsense_D435i_old ${PROJECT_NAME})
# endif()

# Added by Banafshe Bamdad
#if(Python_FOUND)
#    message("::: Halli hallo halluschen. Ich habe Python gefunden. :::")
#endif()
