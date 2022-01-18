#pragma once

#include <Eigen/Geometry>
#include <boost/make_shared.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/time.h>

#include <vicon_calibration/Aliases.h>
#include <vicon_calibration/Utils.h>
#include <vicon_calibration/camera_models/CameraModel.h>

namespace vicon_calibration {

inline std::string
    TransformMatrixToQuaternionAndTranslationStr(const Eigen::Matrix4d& T) {
  Eigen::Matrix3d R = T.block(0, 0, 3, 3);
  Eigen::Quaternion<double> q = Eigen::Quaternion<double>(R);
  std::vector<double> pose{q.w(),   q.x(),   q.y(),  q.z(),
                           T(0, 3), T(1, 3), T(2, 3)};

  std::string output{"["};
  for (double element : pose) {
    output += std::to_string(element);
    output += ", ";
  }
  output.erase(output.end() - 2, output.end());
  output += "]";
  return output;
}

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
  CameraParams(const std::string& intrinsics_path,
               const std::string& input_topic = "",
               const std::string& input_frame = "") {
    intrinsics = intrinsics_path;
    topic = input_topic;
    frame = input_frame;
    camera_model = CameraModel::Create(intrinsics);
  };
  std::string topic;
  std::string frame;
  std::string intrinsics;
  std::shared_ptr<CameraModel> camera_model;
  void Print() {
    std::cout << "----------------------------------\n"
              << "Printing camera params for frame: " << frame << "\n"
              << "Topic: " << topic << "\n"
              << "Intrinsics path: " << intrinsics << "\n"
              << "Image dims: [" << camera_model->GetHeight() << ", "
              << camera_model->GetWidth() << "]\n"
              << "Intrinsics: [";
    for (int i = 0; i < camera_model->GetIntrinsics().size(); i++) {
      std::cout << camera_model->GetIntrinsics()[i] - 1 << ", ";
    }
    std::cout << camera_model
                     ->GetIntrinsics()[camera_model->GetIntrinsics().size() - 1]
              << "]\n";
  }
};

struct TargetParams {
  std::string frame_id;
  std::string extractor_type;
  std::string target_config_path;
  Eigen::Matrix<float, 6, 1>
      crop_scan;              // [min x, max x, min y, max y, min z, max z]
  Eigen::Vector2d crop_image; // [ %u, %v]
  Eigen::Matrix3Xd keypoints_lidar;
  Eigen::Matrix3Xd keypoints_camera;
  PointCloud::Ptr template_cloud;
  bool is_target_2d;
  // These are calculated automatically and only used in IsolateTargetPoints
  Eigen::Vector4d template_centroid;
  Eigen::Vector3d template_dimensions;
  TargetParams() {
    template_cloud = std::make_shared<PointCloud>();
    ;
    is_target_2d = false;
  }
  void Print() {
    std::cout << "----------------------------------\n"
              << "Printing target params for frame: " << frame_id << "\n"
              << "Extractor type: " << extractor_type << "\n"
              << "Config path: " << target_config_path << "\n"
              << "Scan Crop: [" << crop_scan[0] << ", " << crop_scan[1] << ", "
              << crop_scan[2] << "]\n"
              << "Image Crop: [" << crop_image[0] << ", " << crop_image[1]
              << "]\n"
              << "Lidar keypoints size: " << keypoints_lidar.cols() << "\n"
              << "Camera keypoints size: " << keypoints_camera.cols() << "\n"
              << "Is target 2D: " << is_target_2d << "\n";
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct LidarMeasurement {
  PointCloud::Ptr keypoints;
  Eigen::Matrix4d T_VICONBASE_TARGET;
  int lidar_id;
  int target_id;
  std::string lidar_frame;
  std::string target_frame;
  ros::Time time_stamp;
  LidarMeasurement() { keypoints = std::make_shared<PointCloud>(); }
  void Print() {
    std::cout << "----------------------------------------------\n"
              << "Printing measurement for Lidar " << lidar_id << " , Target "
              << target_id << "\n"
              << "Keypoints size: " << keypoints->size() << "\n"
              << "T_VICONBASE_TARGET [qw qx qy qz tx ty tx]: "
              << TransformMatrixToQuaternionAndTranslationStr(
                     T_VICONBASE_TARGET)
              << "\n"
              << "Lidar Frame: " << lidar_frame << "\n"
              << "Target Frame: " << target_frame << "\n";
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct CameraMeasurement {
  pcl::PointCloud<pcl::PointXY>::Ptr keypoints;
  Eigen::Matrix4d T_VICONBASE_TARGET;
  int camera_id;
  int target_id;
  std::string camera_frame;
  std::string target_frame;
  ros::Time time_stamp;
  CameraMeasurement() {
    keypoints = std::make_shared<pcl::PointCloud<pcl::PointXY>>();
  }
  void Print() {
    std::cout << "----------------------------------------------\n"
              << "Printing measurement for Camera " << camera_id << " , Target "
              << target_id << "\n"
              << "Keypoints size: " << keypoints->size() << "\n"
              << "T_VICONBASE_TARGET [qw qx qy qz tx ty tx]: "
              << TransformMatrixToQuaternionAndTranslationStr(
                     T_VICONBASE_TARGET)
              << "\n"
              << "Camera Frame: " << camera_frame << "\n"
              << "Target Frame: " << target_frame << "\n";
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct LoopClosureMeasurement {
  pcl::PointCloud<pcl::PointXY>::Ptr keypoints_camera;
  PointCloud::Ptr keypoints_lidar;
  Eigen::Matrix4d T_VICONBASE_TARGET;
  int camera_id;
  int lidar_id;
  int target_id;
  std::string camera_frame;
  std::string lidar_frame;
  std::string target_frame;
  LoopClosureMeasurement() {
    keypoints_camera = std::make_shared<pcl::PointCloud<pcl::PointXY>>();
  }
  void Print() {
    std::cout << "----------------------------------------------\n"
              << "Printing measurement for Lidar " << lidar_id << " , Camera "
              << camera_id << " , Target " << target_id << "\n"
              << "Lidar Keypoints size: " << keypoints_lidar->size() << "\n"
              << "Camera Keypoints size: " << keypoints_camera->size() << "\n"
              << "T_VICONBASE_TARGET [qw qx qy qz tx ty tx]: "
              << TransformMatrixToQuaternionAndTranslationStr(
                     T_VICONBASE_TARGET)
              << "\n"
              << "Lidar Frame: " << lidar_frame << "\n"
              << "Camera Frame: " << camera_frame << "\n"
              << "Target Frame: " << target_frame << "\n";
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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
  Eigen::Matrix4d transform;
  SensorType type;
  int sensor_id;
  std::string to_frame;   // this is usually the vicon baselink on the robot
  std::string from_frame; // this is the sensor frame
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct CalibratorInputs {
  std::string bag;
  std::string initial_calibration;
  std::string calibration_config;
  std::string optimizer_config;
  std::string ceres_config;
  std::string target_config_path;
  std::string target_data_path;
  std::string camera_intrinsics_path;
  std::string verification_config;
  std::string output_directory;
  bool show_camera_measurements;
  bool show_lidar_measurements;
};

struct CalibratorConfig {
  std::string bag_file;
  std::string initial_calibration_file;
  bool show_camera_measurements{false};
  bool show_lidar_measurements{false};
  std::string vicon_baselink_frame;
  double time_steps;
  int max_measurements{0};
  double min_target_motion{0.05};
  double min_target_rotation{5};
  double max_target_velocity{0.7};
  std::vector<double> crop_time{0, 0};
  std::string optimizer_type{"CERES"}; // Options: CERES
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

typedef std::vector<std::shared_ptr<CameraParams>> CameraParamsVector;
typedef std::vector<std::shared_ptr<TargetParams>> TargetParamsVector;
typedef std::vector<std::vector<std::shared_ptr<CameraMeasurement>>>
    CameraMeasurements;
typedef std::vector<std::vector<std::shared_ptr<LidarMeasurement>>>
    LidarMeasurements;
typedef std::vector<std::shared_ptr<LoopClosureMeasurement>>
    LoopClosureMeasurements;
typedef std::vector<CalibrationResult> CalibrationResults;

} // end namespace vicon_calibration
