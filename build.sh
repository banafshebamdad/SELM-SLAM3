SELMSLAM3_HOME=$PWD
THIRDPARTH_PATH=$SELMSLAM3_HOME/Thirdparty
BBLOGFILE_PATH=$SELMSLAM3_HOME/my_logs/

#read -p "To configure and build Thirdparty/DBow3, press Enter ..."
#cd $THIRDPARTH_PATH/DBow3
#rm -rf build
#mkdir build
#cd build
#cmake .. -DCMAKE_BUILD_TYPE=Release
#make -j20

cd $SELMSLAM3_HOME
read -p "To Configure and build ORB_SLAM3, please press Enter ..."
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSUPERPOINT_WEIGHTS_PATH="$SELMSLAM3_HOME/Weights/superpoint.pt" -DCUDA_INCLUDE_PATH="/usr/local/cuda-11.8/targets/x86_64-linux/include" -DCUDNN_INCLUDE_PATH="/usr/include" -DBBSUPERPOINT_WEIGHT_PATH="$SELMSLAM3_HOME/Weights/BBPretrained_Models/superpoint.onnx" -DBBLIGHTGLUE_WEIGHT_PATH="$SELMSLAM3_HOME/Weights/BBPretrained_Models/superpoint_lightglue.onnx" -DONNX_RUNTIME_PATH="$THIRDPARTH_PATH/onnxruntime-linux-x64-gpu-1.16.1" -DBBLOGFILE_PATH="$BBLOGFILE_PATH"
make -j20
