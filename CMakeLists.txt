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

find_package(gflags REQUIRED)
find_package(rosbag REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(beam REQUIRED utils calibration filtering)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(Catch2 REQUIRED)
find_package(GTSAM REQUIRED)
#find_package(Ceres REQUIRED)

################## Add the core library ##################
add_subdirectory(src/lib)
add_subdirectory(tests)

# Add main executables
add_executable(${PROJECT_NAME}_main
	src/main.cpp
)

target_link_libraries(${PROJECT_NAME}_main
  ${PROJECT_NAME}
	beam::utils
	beam::calibration
  beam::filtering
  gflags
)

################## tools ##################
add_executable(${PROJECT_NAME}_generate_checkerboard_keypoints
  src/tools/generate_checkerboard_keypoints.cpp
)

################## tests ##################

### TEST1 ###
add_executable(${PROJECT_NAME}_tests_cylinder_lidar_extractor
  tests/cylinder_lidar_extractor_test.cpp
)

target_include_directories(${PROJECT_NAME}_tests_cylinder_lidar_extractor
  PUBLIC
    include
)

target_link_libraries(${PROJECT_NAME}_tests_cylinder_lidar_extractor
  ${PROJECT_NAME}
  Catch2::Catch2
  beam::utils
  beam::filtering
)

### TEST2 ###
add_executable(${PROJECT_NAME}_tests_cylinder_camera_extractor
  tests/cylinder_camera_extractor_test.cpp
)

target_include_directories(${PROJECT_NAME}_tests_cylinder_camera_extractor
  PUBLIC
    include
)

target_link_libraries(${PROJECT_NAME}_tests_cylinder_camera_extractor
  ${PROJECT_NAME}
  Catch2::Catch2
)

### TEST3 ###
add_executable(${PROJECT_NAME}_tests_gtsam_factors
  tests/gtsam_factors_test.cpp
)

target_include_directories(${PROJECT_NAME}_tests_gtsam_factors
  PUBLIC
    include
)

target_link_libraries(${PROJECT_NAME}_tests_gtsam_factors
  ${PROJECT_NAME}
  Catch2::Catch2
  beam::calibration
)

### TEST4 ###
add_executable(${PROJECT_NAME}_tests_diamond_camera_extractor
  tests/diamond_camera_extractor_test.cpp
)

target_include_directories(${PROJECT_NAME}_tests_diamond_camera_extractor
  PUBLIC
    include
)

target_link_libraries(${PROJECT_NAME}_tests_diamond_camera_extractor
  ${PROJECT_NAME}
  Catch2::Catch2
  beam::calibration
)

### TEST5 ###
add_executable(${PROJECT_NAME}_tests_diamond_lidar_extractor
  tests/diamond_lidar_extractor_test.cpp
)

target_include_directories(${PROJECT_NAME}_tests_diamond_lidar_extractor
  PUBLIC
    include
)

target_link_libraries(${PROJECT_NAME}_tests_diamond_lidar_extractor
  ${PROJECT_NAME}
  Catch2::Catch2
  beam::utils
  beam::filtering
)

#############################
