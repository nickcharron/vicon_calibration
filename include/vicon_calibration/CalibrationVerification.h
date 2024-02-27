#pragma once

#include <Eigen/Geometry>
#include <pcl/registration/correspondence_estimation.h>
#include <ros/time.h>
#include <rosbag/bag.h>

#include <vicon_calibration/Params.h>
#include <vicon_calibration/TfTree.h>
#include <vicon_calibration/Utils.h>

namespace vicon_calibration {

class CalibrationVerification {
public:
  struct Results {
    int num_lidars;
    int num_cameras;
    int num_lidar_measurements;
    int num_camera_measurements;
    double lidar_average_point_errors_mm;
    double camera_average_reprojection_errors_pixels;
    std::vector<double> calibration_translation_errors_mm;
    std::vector<double> calibration_rotation_errors_deg;
    std::vector<std::string> calibration_frames;
    bool ground_truth_set{false};
  };

  CalibrationVerification(const std::string& config_file_name,
                          const std::string& output_directory,
                          const std::string& calibration_config);

  void LoadJSON();

  void CheckInputs();

  void SetInitialCalib(
      const std::vector<vicon_calibration::CalibrationResult>& calib);

  void SetGroundTruthCalib(
      const std::vector<vicon_calibration::CalibrationResult>& calib);

  void SetOptimizedCalib(
      const std::vector<vicon_calibration::CalibrationResult>& calib);

  void SetTargetCorrections(const std::vector<Eigen::Matrix4d>& corrections);

  void SetConfig(const std::string& calib_config);

  void SetParams(std::shared_ptr<CalibratorConfig>& params);

  void SetLidarMeasurements(
      const std::vector<std::vector<LidarMeasurementPtr>>& lidar_measurements);

  void SetCameraMeasurements(
      const std::vector<std::vector<CameraMeasurementPtr>>&
          camera_measurements);

  void ProcessResults(bool save_measurements = true);

  Results GetSummary();

private:
  void CreateDirectories();

  void SaveLidarVisuals();

  PointCloud::Ptr GetLidarScanFromBag(const std::string& topic);

  void SaveScans(const PointCloud::Ptr& scan_est,
                 const PointCloud::Ptr& scan_opt,
                 const PointCloud::Ptr& targets, const std::string& save_path,
                 int scan_count);

  void GetLidarErrors();

  std::vector<Eigen::Vector3d>
      CalculateLidarErrors(const PointCloud::Ptr& measured_keypoints,
                           const PointCloud::Ptr& estimated_keypoints);

  void SaveCameraVisuals();

  void GetCameraErrors();

  std::vector<Eigen::Vector2d>
      CalculateCameraErrors(const PointCloud::Ptr& measured_keypoints,
                            const Eigen::Matrix4d& T_Sensor_Target,
                            int target_id, int camera_id);

  void PrintConfig();

  void PrintCalibrations(
      std::vector<vicon_calibration::CalibrationResult>& calib,
      const std::string& file_name);

  void PrintTargetCorrections(const std::string& file_name);

  std::string CalibrationErrorsToString(const Eigen::Matrix4d& T1,
                                        const Eigen::Matrix4d& T2,
                                        const std::string& from_frame,
                                        const std::string& to_frame);

  void PrintCalibrationErrors();

  std::shared_ptr<cv::Mat>
      ProjectTargetToImage(const std::shared_ptr<cv::Mat>& img_in,
                           const std::vector<Eigen::Affine3d>& T_Robot_Targets,
                           const Eigen::Matrix4d& T_Robot_Sensor, int cam_iter,
                           cv::Scalar colour);

  void LoadLookupTree();

  void PrintErrorsSummary();

  // params:
  std::vector<double> template_downsample_size_{0.001, 0.001, 0.001};

  // member variables:
  bool initial_calib_set_{false};
  bool params_set_{false};
  bool optimized_calib_set_{false};
  bool ground_truth_calib_set_{false};
  bool lidar_measurements_set_{false};
  bool camera_measurements_set_{false};
  bool show_target_outline_on_image_{true};
  int num_tgts_in_img_;
  std::shared_ptr<CalibratorConfig> params_;
  std::string output_directory_;
  std::string config_file_name_;
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
  Results results_;

  ros::Time lookup_time_;
  std::shared_ptr<vicon_calibration::TfTree> lookup_tree_ =
      std::make_shared<vicon_calibration::TfTree>();
  std::vector<vicon_calibration::CalibrationResult> calibrations_result_;
  std::vector<vicon_calibration::CalibrationResult> calibrations_initial_;
  std::vector<vicon_calibration::CalibrationResult> calibrations_ground_truth_;
  std::vector<Eigen::Matrix4d> target_corrections_;
  std::vector<Eigen::Vector3d> lidar_errors_opt_;
  std::vector<Eigen::Vector3d> lidar_errors_init_;
  std::vector<Eigen::Vector3d> lidar_errors_true_;
  std::vector<Eigen::Vector2d> camera_errors_opt_;
  std::vector<Eigen::Vector2d> camera_errors_init_;
  std::vector<Eigen::Vector2d> camera_errors_true_;
  std::vector<std::vector<LidarMeasurementPtr>> lidar_measurements_;
  std::vector<std::vector<CameraMeasurementPtr>> camera_measurements_;
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>
      corr_est_;
  std::shared_ptr<pcl::Correspondences> correspondences_ =
      std::make_shared<pcl::Correspondences>();
	std::vector<float> camera_angular_errors_;
};

} // end namespace vicon_calibration
