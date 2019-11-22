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
find_package(beam REQUIRED utils calibration filtering)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(Catch2 REQUIRED)
find_package(GTSAM REQUIRED)

################## tools ##################
add_executable(${PROJECT_NAME}_generate_checkerboard_keypoints
  src/tools/generate_checkerboard_keypoints.cpp
)

################## tests ##################
add_executable(${PROJECT_NAME}_lidar_cylinder_extractor_tests
  tests/lidar_cyl_extract_test.cpp
)

target_include_directories(${PROJECT_NAME}_lidar_cylinder_extractor_tests
  PUBLIC
    include
)

target_link_libraries(${PROJECT_NAME}_lidar_cylinder_extractor_tests
  ${PROJECT_NAME}
  Catch2::Catch2
  beam::utils
  beam::filtering
)

add_executable(${PROJECT_NAME}_camera_cylinder_extractor_tests
  tests/cam_cyl_extract_tests.cpp
)

target_include_directories(${PROJECT_NAME}_camera_cylinder_extractor_tests
  PUBLIC
    include
)

target_link_libraries(${PROJECT_NAME}_camera_cylinder_extractor_tests
  ${PROJECT_NAME}
  Catch2::Catch2
)

# Add the core library
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
)

add_executable(${PROJECT_NAME}_main_ig
	src/main_ig.cpp
)

target_link_libraries(${PROJECT_NAME}_main_ig
	${PROJECT_NAME}
	beam::utils
	beam::calibration
  beam::filtering
)

add_executable(${PROJECT_NAME}_main_roben
	src/main_roben.cpp
)

target_link_libraries(${PROJECT_NAME}_main_roben
	${PROJECT_NAME}
	beam::utils
	beam::calibration
  beam::filtering
)
