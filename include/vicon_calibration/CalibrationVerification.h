#pragma once

#include "vicon_calibration/TfTree.h"
#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"
#include <Eigen/Geometry>
#include <pcl/registration/correspondence_estimation.h>
#include <ros/time.h>
#include <rosbag/bag.h>

namespace vicon_calibration {

class CalibrationVerification {
public:
  void LoadJSON(const std::string& file_name = "CalibrationVerification.json");

  void CheckInputs();

  void SetInitialCalib(
      const std::vector<vicon_calibration::CalibrationResult>& calib);

  void SetGroundTruthCalib(
      const std::vector<vicon_calibration::CalibrationResult>& calib);

  void SetOptimizedCalib(
      const std::vector<vicon_calibration::CalibrationResult>& calib);

  void SetConfig(const std::string& calib_config);

  void SetParams(std::shared_ptr<CalibratorConfig>& params);

  void SetLidarMeasurements(
      const std::vector<std::vector<std::shared_ptr<LidarMeasurement>>>&
          lidar_measurements);

  void SetCameraMeasurements(
      const std::vector<std::vector<std::shared_ptr<CameraMeasurement>>>&
          camera_measurements);

  void ProcessResults();

private:
  void CreateDirectories();

  void SaveLidarVisuals();

  PointCloud::Ptr GetLidarScanFromBag(const std::string& topic);

  void SaveScans(const PointCloud::Ptr& scan_est,
                 const PointCloud::Ptr& scan_opt,
                 const PointCloud::Ptr& targets, const std::string& save_path,
                 const int& scan_count);

  void GetLidarErrors();

  std::vector<Eigen::Vector3d, AlignVec3d>
      CalculateLidarErrors(const PointCloud::Ptr& measured_keypoints,
                           const PointCloud::Ptr& estimated_keypoints);

  void SaveCameraVisuals();

  std::shared_ptr<cv::Mat> GetImageFromBag(const std::string& topic);

  void GetCameraErrors();

  std::vector<Eigen::Vector2d, AlignVec2d>
      CalculateCameraErrors(const PointCloud::Ptr& measured_keypoints,
                            const Eigen::Matrix4d& T_SENSOR_TARGET,
                            const int& target_id, const int& camera_id);

  void PrintConfig();

  void PrintCalibrations(
      std::vector<vicon_calibration::CalibrationResult>& calib,
      const std::string& file_name);

  std::string CalibrationErrorsToString(
      const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2,
      const std::string& from_frame, const std::string& to_frame);

  void PrintCalibrationErrors();

  std::shared_ptr<cv::Mat> ProjectTargetToImage(
      const std::shared_ptr<cv::Mat>& img_in,
      const std::vector<Eigen::Affine3d, AlignAff3d>& T_VICONBASE_TGTS,
      const Eigen::Matrix4d& T_VICONBASE_SENSOR, const int& cam_iter,
      cv::Scalar colour);

  void LoadLookupTree();

  void PrintErrorsSummary();

  // params:
  std::vector<double> template_downsample_size_{0.001, 0.001, 0.001};

  // member variables:
  bool initial_calib_set_{false}, optimized_calib_set_{false},
      ground_truth_calib_set_{false}, params_set_{false},
      config_path_set_{false}, lidar_measurements_set_{false},
      camera_measurements_set_{false}, show_target_outline_on_image_{true};
  int num_tgts_in_img_;
  std::shared_ptr<CalibratorConfig> params_;
  std::string output_directory_;
  std::string calibration_config_;
  std::string date_and_time_;
  std::string results_directory_;
  rosbag::Bag bag_;
  int max_image_results_{20};
  int max_lidar_results_{3};
  int keypoint_circle_diameter_{5};
  int outline_circle_diameter_{2};
  double max_pixel_cor_dist_{500}; // in pixels
  double max_point_cor_dist_{0.3}; // in m
  double concave_hull_alpha_multiplier_{1.5};
  ros::Time lookup_time_;
  std::shared_ptr<vicon_calibration::TfTree> lookup_tree_ =
      std::make_shared<vicon_calibration::TfTree>();
  std::vector<vicon_calibration::CalibrationResult> calibrations_result_,
      calibrations_initial_, calibrations_ground_truth_;
  std::vector<Eigen::Vector3d, AlignVec3d> lidar_errors_opt_,
      lidar_errors_init_, lidar_errors_true_;
  std::vector<Eigen::Vector2d, AlignVec2d> camera_errors_opt_,
      camera_errors_init_, camera_errors_true_;
  std::vector<std::vector<std::shared_ptr<LidarMeasurement>>>
      lidar_measurements_;
  std::vector<std::vector<std::shared_ptr<CameraMeasurement>>>
      camera_measurements_;
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
      corr_est_;
  boost::shared_ptr<pcl::Correspondences> correspondences_ =
      boost::make_shared<pcl::Correspondences>();
};

} // end namespace vicon_calibration
