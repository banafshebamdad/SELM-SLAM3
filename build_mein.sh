SUPERSLAM3_HOME=$PWD

read -p "To configure and build Thirdparty/DBow3, press Enter ..."
cd $SUPERSLAM3_HOME/Thirdparty/DBow3
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j20

read -p "To configure and build Thirdparty/DBoW2, press Enter ..."
cd $SUPERSLAM3_HOME/Thirdparty/DBoW2
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j20

read -p "To configure and build Thirdparty/g2o, press Enter ..."
cd $SUPERSLAM3_HOME/Thirdparty/g2o
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j20

read -p "to configure and build Thirdparty/Sophus, press Enter ..."
cd $SUPERSLAM3_HOME/Thirdparty/Sophus
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j20


read -p "To uncompress vocabulary, press Enter ..."
cd $SUPERSLAM3_HOME/Vocabulary
tar -xf ORBvoc.txt.tar.gz

read -p "To configure and build SuperSLAM3, press Enter ..."
cd $SUPERSLAM3_HOME
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSUPERPOINT_WEIGHTS_PATH="$SUPERSLAM3_HOME/Weights/superpoint.pt" -DCUDNN_INCLUDE_PATH="/usr/include"
make -j20
