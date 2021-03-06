cmake_minimum_required(VERSION 3.14)
project(vicon_calibration)

SET(CMAKE_CXX_STANDARD 17)

find_package(gflags REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(beam REQUIRED utils calibration filtering)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(Catch2 REQUIRED)
find_package(Ceres REQUIRED)

set(catkin_deps roscpp rosbag geometry_msgs nav_msgs sensor_msgs std_msgs tf)
find_package(catkin REQUIRED ${catkin_deps})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


################## Add the core library ##################
add_subdirectory(src/lib)
add_subdirectory(tests)

# Add main executables
add_executable(${PROJECT_NAME}_main
	src/main.cpp
)

target_link_libraries(${PROJECT_NAME}_main
  ${PROJECT_NAME}
  gflags
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
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
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Catch2::Catch2
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
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  Catch2::Catch2
)

### TEST3 ###
add_executable(${PROJECT_NAME}_tests_diamond_camera_extractor
  tests/diamond_camera_extractor_test.cpp
)

target_include_directories(${PROJECT_NAME}_tests_diamond_camera_extractor
  PUBLIC
    include
)

target_link_libraries(${PROJECT_NAME}_tests_diamond_camera_extractor
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  Catch2::Catch2
)

### TEST4 ###
add_executable(${PROJECT_NAME}_tests_diamond_lidar_extractor
  tests/diamond_lidar_extractor_test.cpp
)

target_include_directories(${PROJECT_NAME}_tests_diamond_lidar_extractor
  PUBLIC
    include
)

target_link_libraries(${PROJECT_NAME}_tests_diamond_lidar_extractor
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Catch2::Catch2
)

### TEST5 ###
add_executable(${PROJECT_NAME}_tests_ceres_camera_tests
  tests/ceres_camera_tests.cpp
)

target_include_directories(${PROJECT_NAME}_tests_ceres_camera_tests
  PUBLIC
    include
)

target_link_libraries(${PROJECT_NAME}_tests_ceres_camera_tests
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ceres
  Catch2::Catch2
)

### TEST6 ###
add_executable(${PROJECT_NAME}_tests_ceres_lidar_tests
  tests/ceres_lidar_tests.cpp
)

target_include_directories(${PROJECT_NAME}_tests_ceres_lidar_tests
  PUBLIC
    include
)

target_link_libraries(${PROJECT_NAME}_tests_ceres_lidar_tests
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ceres
  Catch2::Catch2
)

### TEST8 ###
add_executable(${PROJECT_NAME}_tests_optimizer_tests
  tests/optimizer_tests.cpp
)

target_include_directories(${PROJECT_NAME}_tests_optimizer_tests
  PUBLIC
    include
)

target_link_libraries(${PROJECT_NAME}_tests_optimizer_tests
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ceres
  Catch2::Catch2
)
#############################
