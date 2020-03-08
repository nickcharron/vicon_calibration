#pragma once

#include "vicon_calibration/TfTree.h"
#include "vicon_calibration/gtsam/Graph.h"
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
  void RunCalibration(std::string config_file = "ViconCalibrationConfig.json");

private:
  /**
   * @brief loads extrinsic estimates and saves as tf tree object. These come
   * from either a JSON or from the tf messages in the bag
   */
  void LoadEstimatedExtrinsics();

  void LoadLookupTree();

  void GetInitialCalibration(std::string &sensor_frame, SensorType type,
                             uint8_t &sensor_id);

  void GetInitialCalibrationPerturbed(std::string &sensor_frame,
                                      SensorType type, uint8_t &sensor_id);

  /**
   * @brief get initial guess of where the targets are located at the current
   * time point
   * @param sensor_frame
   * @return T_sensor_tgts_estimated estimated transform from targets to sensor
   * frame
   */
  std::vector<Eigen::Affine3d, AlignAff3d>
  GetInitialGuess(std::string &sensor_frame);

  /**
   * @brief Compute the measurements from one lidar
   * @param lidar_iter
   */
  void GetLidarMeasurements(uint8_t &lidar_iter);

  /**
   * @brief Compute the measurements from the cameras
   * @param cam_iter
   */
  void GetCameraMeasurements(uint8_t &cam_iter);

  void GetLoopClosureMeasurements();

  void SetCalibrationInitials();

  bool PassedMinTranslation(const Eigen::Affine3d &TA_S_T_prev,
                            const Eigen::Affine3d &TA_S_T_curr);

  bool PassedMaxVelocity(const Eigen::Affine3d &TA_S_T_before,
                         const Eigen::Affine3d &TA_S_T_after);

  std::shared_ptr<CalibratorConfig> params_;
  std::string results_directory_;
  std::string config_file_path_;
  std::shared_ptr<LidarExtractor> lidar_extractor_;
  std::shared_ptr<CameraExtractor> camera_extractor_;
  ros::Time lookup_time_, time_start_, time_end_;
  std::shared_ptr<vicon_calibration::TfTree> estimate_extrinsics_ =
      std::make_shared<vicon_calibration::TfTree>();
  std::shared_ptr<vicon_calibration::TfTree> lookup_tree_ =
      std::make_shared<vicon_calibration::TfTree>();
  std::vector<std::vector<std::shared_ptr<LidarMeasurement>>>
      lidar_measurements_;
  std::vector<std::vector<std::shared_ptr<CameraMeasurement>>>
      camera_measurements_;
  std::vector<std::shared_ptr<LoopClosureMeasurement>>
      loop_closure_measurements_;
  std::vector<vicon_calibration::CalibrationResult> calibrations_result_,
      calibrations_initial_,
      calibrations_perturbed_; // pertubed use for testing with simulation ONLY
  rosbag::Bag bag_;
  vicon_calibration::Graph graph_;
  Eigen::MatrixXd T_VICONBASE_SENSOR_ = Eigen::MatrixXd(4, 4);
  Eigen::MatrixXd T_VICONBASE_SENSOR_pert_ = Eigen::MatrixXd(4, 4);
  // Pert for testing simulation ONLY
};

} // end namespace vicon_calibration
