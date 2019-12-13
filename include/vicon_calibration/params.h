#pragma once

#include <Eigen/Geometry>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/time.h>
#include <string>

namespace vicon_calibration {

/**
 * @brief Enum class for different types of sensors
 */
enum class SensorType { CAMERA = 0, LIDAR };

struct LidarParams {
  std::string topic;
  std::string frame;
  double time_steps;
};

struct CameraParams {
  std::string topic;
  std::string frame;
  std::string intrinsics;
  bool images_distorted;
  double time_steps;
};

struct TargetParams {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::string frame_id;
  std::string extractor_type;
  std::string target_config_path;
  Eigen::VectorXd crop_scan;
  Eigen::VectorXd crop_image;
  pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> keypoints_lidar;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> keypoints_camera;
  TargetParams(){
    crop_scan = Eigen::VectorXd(3);
    crop_image = Eigen::VectorXd(2);
  }
};

struct LidarMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints;
  Eigen::MatrixXd T_VICONBASE_TARGET;
  int lidar_id;
  int target_id;
  std::string lidar_frame;
  std::string target_frame;
  LidarMeasurement(){
    T_VICONBASE_TARGET = Eigen::MatrixXd(4,4);
  }
};

struct CameraMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  pcl::PointCloud<pcl::PointXY>::Ptr keypoints;
  Eigen::MatrixXd T_VICONBASE_TARGET;
  int camera_id;
  int target_id;
  std::string camera_frame;
  std::string target_frame;
  CameraMeasurement(){
    T_VICONBASE_TARGET = Eigen::MatrixXd(4,4);
  }
};

struct LoopClosureMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  pcl::PointCloud<pcl::PointXY>::Ptr keypoints_camera;
  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_lidar;
  Eigen::MatrixXd T_VICONBASE_TARGET;
  int camera_id;
  int lidar_id;
  int target_id;
  std::string camera_frame;
  std::string lidar_frame;
  std::string target_frame;
  LoopClosureMeasurement(){
    T_VICONBASE_TARGET = Eigen::MatrixXd(4,4);
  }
};

struct Correspondence {
  int target_point_index;
  int measured_point_index;
  int measurement_index;
};

struct CalibrationResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::MatrixXd transform;
  SensorType type;
  int sensor_id;
  std::string to_frame;   // this is usually the vicon baselink on the robot
  std::string from_frame; // this is the sensor frame
  CalibrationResult(){
    transform = Eigen::MatrixXd(4,4);
  }
};

} // end namespace vicon_calibration
