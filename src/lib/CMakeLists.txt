SET(CMAKE_CXX_STANDARD 17)

# Core library
add_library(${PROJECT_NAME}
  camera_models/CameraModel.cpp
  camera_models/DoubleSphere.cpp
  camera_models/KannalaBrandt.cpp
  camera_models/Radtan.cpp
  camera_models/Cataditropic.cpp
  measurement_extractors/CameraExtractor.cpp
  measurement_extractors/LidarExtractor.cpp
  measurement_extractors/CylinderCameraExtractor.cpp
  measurement_extractors/CylinderLidarExtractor.cpp
  measurement_extractors/DiamondCameraExtractor.cpp
  measurement_extractors/DiamondLidarExtractor.cpp
  measurement_extractors/DiamondCornersLidarExtractor.cpp
  measurement_extractors/IsolateTargetPoints.cpp
  optimization/Optimizer.cpp
  optimization/CeresOptimizer.cpp
  TfTree.cpp
  Utils.cpp
  Gflags.cpp
  JsonTools.cpp
  Visualizer.cpp
  CropBox.cpp
  ViconCalibrator.cpp
  CalibrationVerification.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ../../include
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ceres
  nlohmann_json::nlohmann_json
  ${rosbag_LIBRARIES}
  ${cv_bridge_LIBRARIES}
)
