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
#include <nlohmann/json.hpp>

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
  double max_angular_resolution_deg;
};

struct CameraParams {
  CameraParams(const std::string &intrinsics_path,
               const std::string &input_topic = "",
               const std::string &input_frame = "") {
    intrinsics = intrinsics_path;
    topic = input_topic;
    frame = input_frame;
    try {
      camera_model = beam_calibration::CameraModel::Create(intrinsics);
    } catch (nlohmann::detail::parse_error &ex) {
      LOG_ERROR("Unable to load json config file: %s", intrinsics.c_str());
      LOG_ERROR("%s", ex.what());
    }
  };
  std::string topic;
  std::string frame;
  std::string intrinsics;
  std::shared_ptr<beam_calibration::CameraModel> camera_model;
};

struct TargetParams {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::string frame_id;
  std::string extractor_type;
  std::string target_config_path;
  Eigen::VectorXd crop_scan;
  Eigen::VectorXd crop_image;
  std::vector<Eigen::Vector3d, AlignVec3d> keypoints_lidar;
  std::vector<Eigen::Vector3d, AlignVec3d> keypoints_camera;
  PointCloud::Ptr template_cloud;
  bool is_target_2d;
  // These are calculated automatically and only used in IsolateTargetPoints
  Eigen::VectorXd template_centroid;
  double template_size;
  TargetParams() {
    crop_scan = Eigen::VectorXd(3);
    crop_image = Eigen::VectorXd(2);
    template_centroid = Eigen::VectorXd(4);
    template_centroid.setZero();
    template_size = 0;
    is_target_2d = false;
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
  double min_target_rotation{5};
  double max_target_velocity{0.7};
  double start_delay{0};
  std::vector<std::shared_ptr<vicon_calibration::TargetParams>> target_params;
  std::vector<std::shared_ptr<vicon_calibration::CameraParams>> camera_params;
  std::vector<std::shared_ptr<vicon_calibration::LidarParams>> lidar_params;
};

struct Counters {
  int total_camera{0};
  int camera_accepted{0};
  int camera_rejected_fast{0};
  int camera_rejected_still{0};
  int camera_rejected_invalid{0};
  int total_lidar{0};
  int lidar_accepted{0};
  int lidar_rejected_fast{0};
  int lidar_rejected_still{0};
  int lidar_rejected_invalid{0};

  void reset() {
    total_camera = 0;
    camera_accepted = 0;
    total_lidar = 0;
    camera_rejected_fast = 0;
    camera_rejected_still = 0;
    camera_rejected_invalid = 0;
    lidar_accepted = 0;
    lidar_rejected_fast = 0;
    lidar_rejected_still = 0;
    lidar_rejected_invalid = 0;
  }
};

} // end namespace vicon_calibration
