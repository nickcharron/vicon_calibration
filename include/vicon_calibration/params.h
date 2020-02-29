#pragma once

#include <Eigen/Geometry>
#include <beam_calibration/CameraModel.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/time.h>
#include <string>

namespace vicon_calibration {

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudColor;
typedef Eigen::aligned_allocator<Eigen::Vector3d> AlignVec3d;
typedef Eigen::aligned_allocator<Eigen::Vector2d> AlignVec2d;
typedef Eigen::aligned_allocator<Eigen::Affine3d> AlignAff3d;

/**
 * @brief Enum class for different types of sensors
 */
enum class SensorType { CAMERA = 0, LIDAR };

struct LidarParams {
  std::string topic;
  std::string frame;
};

struct CameraParams {
  std::string topic;
  std::string frame;
  std::string intrinsics;
  std::shared_ptr<beam_calibration::CameraModel> camera_model;
  bool images_distorted;
};

struct TargetParams {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::string frame_id;
  std::string extractor_type;
  std::string target_config_path;
  Eigen::VectorXd crop_scan;
  Eigen::VectorXd crop_image;
  PointCloud::Ptr template_cloud;
  std::vector<Eigen::Vector3d, AlignVec3d> keypoints_lidar;
  std::vector<Eigen::Vector3d, AlignVec3d> keypoints_camera;
  TargetParams() {
    crop_scan = Eigen::VectorXd(3);
    crop_image = Eigen::VectorXd(2);
  }
};

struct LidarMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PointCloud::Ptr keypoints;
  Eigen::MatrixXd T_VICONBASE_TARGET;
  int lidar_id;
  int target_id;
  std::string lidar_frame;
  std::string target_frame;
  ros::Time time_stamp;
  LidarMeasurement() { T_VICONBASE_TARGET = Eigen::MatrixXd(4, 4); }
};

struct CameraMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  pcl::PointCloud<pcl::PointXY>::Ptr keypoints;
  Eigen::MatrixXd T_VICONBASE_TARGET;
  int camera_id;
  int target_id;
  std::string camera_frame;
  std::string target_frame;
  ros::Time time_stamp;
  CameraMeasurement() { T_VICONBASE_TARGET = Eigen::MatrixXd(4, 4); }
};

struct LoopClosureMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  pcl::PointCloud<pcl::PointXY>::Ptr keypoints_camera;
  PointCloud::Ptr keypoints_lidar;
  Eigen::MatrixXd T_VICONBASE_TARGET;
  int camera_id;
  int lidar_id;
  int target_id;
  std::string camera_frame;
  std::string lidar_frame;
  std::string target_frame;
  LoopClosureMeasurement() { T_VICONBASE_TARGET = Eigen::MatrixXd(4, 4); }
};

struct Correspondence {
  int target_point_index;
  int measured_point_index;
  int measurement_index;
  int sensor_index;
};

struct LoopCorrespondence {
  int camera_target_point_index;
  int camera_measurement_point_index;
  int lidar_target_point_index;
  int lidar_measurement_point_index;
  int camera_id;
  int lidar_id;
  int target_id;
  int measurement_index;
};

struct CalibrationResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::MatrixXd transform;
  SensorType type;
  int sensor_id;
  std::string to_frame;   // this is usually the vicon baselink on the robot
  std::string from_frame; // this is the sensor frame
  CalibrationResult() { transform = Eigen::MatrixXd(4, 4); }
};

struct CalibratorConfig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::string bag_file;
  std::string initial_calibration_file;
  bool lookup_tf_calibrations{false};
  bool using_simulation{false};
  std::string vicon_baselink_frame;
  double time_steps;
  bool show_camera_measurements{false};
  bool show_lidar_measurements{false};
  bool run_verification{true};
  bool use_loop_closure_measurements{true};
  Eigen::VectorXd initial_guess_perturbation; // for testing sim
  double min_target_motion{0.05};
  double max_target_velocity{0.7};
  std::vector<std::shared_ptr<vicon_calibration::TargetParams>> target_params;
  std::vector<std::shared_ptr<vicon_calibration::CameraParams>> camera_params;
  std::vector<std::shared_ptr<vicon_calibration::LidarParams>> lidar_params;
};

} // end namespace vicon_calibration
