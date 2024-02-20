#pragma once

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <ros/time.h>
#include <rosbag/bag.h>

#include <vicon_calibration/CropBox.h>
#include <vicon_calibration/Params.h>
#include <vicon_calibration/TfTree.h>
#include <vicon_calibration/Utils.h>
#include <vicon_calibration/measurement_extractors/CameraExtractor.h>
#include <vicon_calibration/measurement_extractors/LidarExtractor.h>
#include <vicon_calibration/optimization/Optimizer.h>

namespace vicon_calibration {

/**
 * @brief class for running the vicon calibration
 */
class ViconCalibrator {
public:
  /**
   * @brief default constructor
   */
  ViconCalibrator(const CalibratorInputs& inputs);

  /**
   * @brief default deconstructor
   */
  ~ViconCalibrator() = default;

  /**
   * @brief run calibration
   */
  void RunCalibration();

private:
  /**
   * @brief loads extrinsic estimates and saves as tf tree object. These come
   * from either a JSON or from the tf messages in the bag
   */
  void LoadEstimatedExtrinsics();

  void GetTimeWindow();

  void Setup();

  void GetMeasurements();

  void LoadLookupTree(const ros::Time& lookup_time);

  void GetInitialCalibrations();

  void RunVerification();

  /**
   * @brief get initial guess of where the targets are located at the current
   * time point
   * @param sensor_frame
   * @return T_sensor_tgts_estimated estimated transform from targets to sensor
   * frame
   */
  std::vector<Eigen::Affine3d, AlignAff3d>
      GetInitialGuess(const ros::Time& lookup_time,
                      const std::string& sensor_frame, SensorType type,
                      int sensor_id);

  /**
   * @brief Compute the measurements from one lidar
   * @param lidar_iter
   */
  void GetLidarMeasurements(uint8_t& lidar_iter);

  /**
   * @brief Compute the measurements from the cameras
   * @param cam_iter
   */
  void GetCameraMeasurements(uint8_t& cam_iter);

  void SetCalibrationInitials();

  bool PassedMinMotion(const Eigen::Affine3d& TA_S_T_prev,
                       const Eigen::Affine3d& TA_S_T_curr);

  bool PassedMaxVelocity(const Eigen::Affine3d& TA_S_T_before,
                         const Eigen::Affine3d& TA_S_T_after);

  void OutputMeasurementStats();

  void Solve();

  const CalibratorInputs inputs_;
  std::shared_ptr<CalibratorConfig> params_;
  std::vector<vicon_calibration::Counters> lidar_counters_;
  std::vector<vicon_calibration::Counters> camera_counters_;
  std::string results_directory_;
  std::string config_file_path_;
  std::shared_ptr<LidarExtractor> lidar_extractor_;
  std::shared_ptr<CameraExtractor> camera_extractor_;
  ros::Time time_start_;
  ros::Time time_end_;
  std::shared_ptr<TfTree> estimate_extrinsics_ = std::make_shared<TfTree>();
  std::shared_ptr<TfTree> lookup_tree_ = std::make_shared<TfTree>();
  LidarMeasurements lidar_measurements_;
  CameraMeasurements camera_measurements_;
  CalibrationResults calibrations_initial_;
  CalibrationResults calibrations_final_;
  std::vector<Eigen::Matrix4d> target_corrections_;

  rosbag::Bag bag_;
  std::shared_ptr<Visualizer> pcl_viewer_;

  CropBox input_cropbox_;
  float input_cropbox_max_{5};
};

} // end namespace vicon_calibration
