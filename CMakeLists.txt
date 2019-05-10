cmake_minimum_required(VERSION 3.14)
project(vicon_calibration)

SET(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  tf
)

find_package(rosbag REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(beam REQUIRED utils calibration)
find_package(nlohmann_json 3.2.0 REQUIRED)

# Add the core library
add_subdirectory(src/lib)
add_subdirectory(tests)

# Add main executable
add_executable(${PROJECT_NAME}_main
	src/ViconCalibrator.cpp
)
target_link_libraries(${PROJECT_NAME}_main
	${PROJECT_NAME}
	beam::utils
	beam::calibration
)
