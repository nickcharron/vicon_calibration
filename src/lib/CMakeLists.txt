SET(CMAKE_CXX_STANDARD 17)

# Core library
add_library(${PROJECT_NAME}
  ViconCalibrator.cpp
  measurement_extractors/CameraExtractor.cpp
  measurement_extractors/LidarExtractor.cpp
  measurement_extractors/CylinderCameraExtractor.cpp
  measurement_extractors/CylinderLidarExtractor.cpp
  measurement_extractors/DiamondCameraExtractor.cpp
  measurement_extractors/DiamondLidarExtractor.cpp
  measurement_extractors/IsolateTargetPoints.cpp
  gtsam/Graph.cpp
  TfTree.cpp
  utils.cpp
  JsonTools.cpp
  CalibrationVerification.cpp
)
target_compile_options(${PROJECT_NAME}
  PUBLIC
    -march=native -msse4.2 -mfpmath=sse
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ../../include
)
target_link_libraries(${PROJECT_NAME}
  catkin::catkin
  OpenCV::OpenCV
  PCL::PCL
  beam::utils
  beam::calibration
  beam::filtering
  gtsam
  nlohmann_json::nlohmann_json
  ${rosbag_LIBRARIES}
  ${cv_bridge_LIBRARIES}
)
