SET(CMAKE_CXX_STANDARD 17)

# Core library
add_library(${PROJECT_NAME}
  ViconCalibrator.cpp
  CameraExtractor.cpp
  CylinderCameraExtractor.cpp
  LidarExtractor.cpp
  CylinderLidarExtractor.cpp
  TfTree.cpp
  #CameraFactor.cpp
  #LidarFactor.cpp
  GTSAMGraph.cpp
  utils.cpp
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
