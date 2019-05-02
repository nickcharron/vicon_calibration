# Core library
add_library(${PROJECT_NAME}
  CamCylExtractor.cpp
  LidarCylExtractor.cpp
  GTSAMGraph.cpp

)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ../../include
)
target_link_libraries(${PROJECT_NAME}
  catkin::catkin
  OpenCV::OpenCV
  ${rosbag_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  PCL::PCL
  beam::utils
  beam::calibration
  nlohmann_json::nlohmann_json
)
