#pragma once

#include <string>
#include <Eigen/Geometry>
#include <ros/time.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>

namespace vicon_calibration {

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

struct TargetParams{
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

struct CameraCorresspondance {
  gtsam::Point2 pixel;
  gtsam::Point3 point;
  int camera_id;
  int target_id;
};

struct CalibrationResult {
  Eigen::Matrix4d transform;
  std::string type; // either LIDAR or CAMERA (TODO: make this enum??)
  std::string to_frame; // this is the sensor frame
  std::string from_frame; // this is usually the vicon baselink on the robot
};

} // end namespace vicon_calibration
