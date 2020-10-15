#pragma once

#include "vicon_calibration/TfTree.h"
#include "vicon_calibration/optimization/Optimizer.h"
#include "vicon_calibration/measurement_extractors/CylinderCameraExtractor.h"
#include "vicon_calibration/measurement_extractors/CylinderLidarExtractor.h"
#include "vicon_calibration/measurement_extractors/DiamondCameraExtractor.h"
#include "vicon_calibration/measurement_extractors/DiamondLidarExtractor.h"
#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <ros/time.h>
#include <rosbag/bag.h>

namespace vicon_calibration {

/**
 * @brief class for running the vicon calibration
 */
class ViconCalibrator {
public:
  /**
   * @brief default constructor
   */
  ViconCalibrator() = default;

  /**
   * @brief default deconstructor
   */
  ~ViconCalibrator() = default;

  /**
   * @brief run calibration
   */
  void RunCalibration(const std::string& config_file,
                      bool show_lidar_measurements,
                      bool show_camera_measurements);

private:
  /**
   * @brief loads extrinsic estimates and saves as tf tree object. These come
   * from either a JSON or from the tf messages in the bag
   */
  void LoadEstimatedExtrinsics();

  void LoadLookupTree();

  void GetInitialCalibration(std::string& sensor_frame, SensorType type,
                             uint8_t& sensor_id);

  void GetInitialCalibrationPerturbed(std::string& sensor_frame,
                                      SensorType type, uint8_t& sensor_id);

  /**
   * @brief get initial guess of where the targets are located at the current
   * time point
   * @param sensor_frame
   * @return T_sensor_tgts_estimated estimated transform from targets to sensor
   * frame
   */
  std::vector<Eigen::Affine3d, AlignAff3d>
      GetInitialGuess(std::string& sensor_frame);

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

  void GetLoopClosureMeasurements();

  void SetCalibrationInitials();

  bool PassedMinTranslation(const Eigen::Affine3d& TA_S_T_prev,
                            const Eigen::Affine3d& TA_S_T_curr);

  bool PassedMaxVelocity(const Eigen::Affine3d& TA_S_T_before,
                         const Eigen::Affine3d& TA_S_T_after);

  void OutputMeasurementStats();

  std::shared_ptr<CalibratorConfig> params_;
  vicon_calibration::Counters counters_;
  std::string results_directory_;
  std::string config_file_path_;
  std::shared_ptr<LidarExtractor> lidar_extractor_;
  std::shared_ptr<CameraExtractor> camera_extractor_;
  ros::Time lookup_time_, time_start_, time_end_;
  std::shared_ptr<TfTree> estimate_extrinsics_ = std::make_shared<TfTree>();
  std::shared_ptr<TfTree> lookup_tree_ = std::make_shared<TfTree>();
  LidarMeasurements lidar_measurements_;
  CameraMeasurements camera_measurements_;
  LoopClosureMeasurements loop_closure_measurements_;
  CalibrationResults calibrations_result_;
  CalibrationResults calibrations_initial_;
  CalibrationResults calibrations_perturbed_; // for testing with sim ONLY
  rosbag::Bag bag_;
  std::shared_ptr<Optimizer> optimizer_;
  Eigen::MatrixXd T_VICONBASE_SENSOR_ = Eigen::MatrixXd(4, 4);
  Eigen::MatrixXd T_VICONBASE_SENSOR_pert_ = Eigen::MatrixXd(4, 4);
  // Pert for testing simulation ONLY
};

} // end namespace vicon_calibration
