#pragma once

#include "vicon_calibration/TfTree.h"
#include "vicon_calibration/params.h"
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

  void SaveCameraResults();

  void PrintConfig();

  void
  PrintCalibrations(std::vector<vicon_calibration::CalibrationResult> &calib,
                    const std::string &file_name);

  std::shared_ptr<cv::Mat> ProjectTargetToImage(
      const std::shared_ptr<cv::Mat> &img_in,
      const std::vector<Eigen::Affine3d,
                        Eigen::aligned_allocator<Eigen::Affine3d>>
          &T_viconbase_tgts,
      const Eigen::Matrix4d &T_VICONBASE_SENSOR, const int &cam_iter,
      cv::Scalar colour);

  void LoadLookupTree();

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
  ros::Time lookup_time_;
  std::shared_ptr<vicon_calibration::TfTree> lookup_tree_ =
      std::make_shared<vicon_calibration::TfTree>();
  std::vector<vicon_calibration::CalibrationResult> calibrations_result_,
      calibrations_initial_, calibrations_perturbed_;
};

} // end namespace vicon_calibration
