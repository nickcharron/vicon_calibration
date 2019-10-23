#pragma once

#include "vicon_calibration/params.h"
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

namespace vicon_calibration {

/**
 * @brief class for building and storing the GTSAM graph used in
 * vicon_calibration
 */
class GTSAMGraph {
public:
  GTSAMGraph() = default;
  ~GTSAMGraph() = default;

  void LoadTargetPoints(std::string &template_cloud_path);

  void SetLidarMeasurements(
      std::vector<vicon_calibration::LidarMeasurement> &lidar_measurements);

  void SetCameraMeasurements(
      std::vector<vicon_calibration::CameraMeasurement> &camera_measurements);

  void SetInitialGuess(
      std::vector<vicon_calibration::CalibrationResult> &initial_guess);

  void SetCameraParams(
      std::vector<vicon_calibration::CameraParams> &camera_params);

  void SolveGraph();

  std::vector<vicon_calibration::CalibrationResult> GetResults();

  void Print(std::string &file_name, bool print_to_terminal);

private:
  void AddInitials();

  void AddLidarMeasurements();

  void Clear();

  void Optimize();

  std::vector<vicon_calibration::LidarMeasurement> lidar_measurements_;
  std::vector<vicon_calibration::CameraMeasurement> camera_measurements_;
  std::vector<vicon_calibration::CalibrationResult> calibration_results_,
      calibration_initials_;
  std::vector<vicon_calibration::CameraParams> camera_params_;
  std::vector<Eigen::Vector4d> target_points_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initials_, initials_updated_, results_;
  std::vector<std::shared_ptr<beam_calibration::CameraModel>> camera_models_;
  std::vector<vicon_calibration::CameraCorresspondance> camera_corresspondances_;
};

} // end namespace vicon_calibration
