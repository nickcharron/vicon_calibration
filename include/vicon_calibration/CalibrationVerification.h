#pragma once

#include "vicon_calibration/TfTree.h"
#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"
#include <Eigen/Geometry>
#include <ros/time.h>
#include <rosbag/bag.h>

namespace vicon_calibration {

class CalibrationVerification {

public:
  void LoadJSON(const std::string &file_name = "CalibrationVerification.json");

  void SetInitialCalib(
      const std::vector<vicon_calibration::CalibrationResult> &calib);

  void SetPeturbedCalib(
      const std::vector<vicon_calibration::CalibrationResult> &calib);

  void SetOptimizedCalib(
      const std::vector<vicon_calibration::CalibrationResult> &calib);

  void SetConfig(const std::string &calib_config);

  void SetParams(std::shared_ptr<CalibratorConfig> &params);

  void ProcessResults();

private:
  void CreateDirectories();

  void SaveLidarResults();

  std::vector<Eigen::Vector3d, AlignVec3d> GetLidarErrors(
      const std::vector<Eigen::Affine3d, AlignAff3d> &T_VICONBASE_TGTS,
      const PointCloud::Ptr &scan_viconbase, uint8_t &lidar_id);

  std::vector<Eigen::Vector2d, AlignVec2d> GetCameraErrors(
      const std::vector<Eigen::Affine3d, AlignAff3d> &T_VICONBASE_TGTS,
      const Eigen::Matrix4d &T_VICONBASE_CAMERA,
      const std::shared_ptr<cv::Mat> &img, uint8_t &camera_id);

  void SaveCameraResults();

  void PrintConfig();

  void
  PrintCalibrations(std::vector<vicon_calibration::CalibrationResult> &calib,
                    const std::string &file_name);

  void PrintCalibrationErrors();

  std::shared_ptr<cv::Mat> ProjectTargetToImage(
      const std::shared_ptr<cv::Mat> &img_in,
      const std::vector<Eigen::Affine3d, AlignAff3d> &T_VICONBASE_TGTS,
      const Eigen::Matrix4d &T_VICONBASE_SENSOR, const int &cam_iter,
      cv::Scalar colour);

  void LoadLookupTree();

  void PrintErrors();

  // params:
  std::vector<double> template_downsample_size_{0.005, 0.005, 0.005};
  double concave_hull_alpha_{5};

  // member variables:
  int num_tgts_in_img_;
  std::shared_ptr<CalibratorConfig> params_;
  std::string output_directory_;
  std::string calibration_config_;
  std::string date_and_time_;
  std::string results_directory_;
  rosbag::Bag bag_;
  ros::Duration time_increment_ = ros::Duration(10);
  int max_image_results_{20};
  int max_lidar_results_{3};
  double max_pixel_cor_dist_{500}; // in pixels
  double max_point_cor_dist_{0.3}; // in m
  ros::Time lookup_time_;
  std::shared_ptr<vicon_calibration::TfTree> lookup_tree_ =
      std::make_shared<vicon_calibration::TfTree>();
  std::vector<vicon_calibration::CalibrationResult> calibrations_result_,
      calibrations_initial_, calibrations_perturbed_;
  std::vector<Eigen::Vector3d, AlignVec3d> lidar_errors_opt_,
      lidar_errors_init_;
  std::vector<Eigen::Vector2d, AlignVec2d> camera_errors_opt_,
      camera_errors_init_;
};

} // end namespace vicon_calibration
