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
  std::string frame_id;
  std::string extractor_type;
  std::string target_config_path;
  Eigen::Vector3d crop_scan;
  Eigen::Vector2d crop_image;
  pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud;
  std::vector<Eigen::Vector3d> keypoints_lidar;
  std::vector<Eigen::Vector3d> keypoints_camera;
};

struct LidarMeasurement {
  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints;
  Eigen::Matrix4d T_VICONBASE_TARGET;
  int lidar_id;
  int target_id;
  std::string lidar_frame;
  std::string target_frame;
};

struct CameraMeasurement {
  pcl::PointCloud<pcl::PointXY>::Ptr keypoints;
  Eigen::Matrix4d T_VICONBASE_TARGET;
  int camera_id;
  int target_id;
  std::string camera_frame;
  std::string target_frame;
};

struct LoopClosureMeasurement {
  pcl::PointCloud<pcl::PointXY>::Ptr keypoints_camera;
  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_lidar;
  Eigen::Matrix4d T_VICONBASE_TARGET;
  int camera_id;
  int lidar_id;
  int target_id;
  std::string camera_frame;
  std::string lidar_frame;
  std::string target_frame;
};

struct Correspondence {
  int target_point_index;
  int measured_point_index;
  int measurement_index;
};

struct CalibrationResult {
  Eigen::Matrix4d transform;
  SensorType type;
  int sensor_id;
  std::string to_frame;   // this is the sensor frame
  std::string from_frame; // this is usually the vicon baselink on the robot
};

} // end namespace vicon_calibration
