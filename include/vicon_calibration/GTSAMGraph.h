#pragma once

#include "vicon_calibration/params.h"
#include <beam_calibration/CameraModel.h>
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

  void SetTargetParams(std::vector<std::shared_ptr<vicon_calibration::TargetParams>> &target_params);

  void SetLidarMeasurements(
      std::vector<vicon_calibration::LidarMeasurement> &lidar_measurements);

  void SetCameraMeasurements(
      std::vector<vicon_calibration::CameraMeasurement> &camera_measurements);

  void SetInitialGuess(
      std::vector<vicon_calibration::CalibrationResult> &initial_guess);

  void SetCameraParams(
      std::vector<std::shared_ptr<vicon_calibration::CameraParams>> &camera_params);

  void SolveGraph();

  std::vector<vicon_calibration::CalibrationResult> GetResults();

  void Print(std::string &file_name, bool print_to_terminal);

private:
  bool CheckConvergence();

  void CheckInputs();

  void Clear();

  void AddInitials();

  void SetImageCorrespondences();

  void SetLidarCorrespondences();

  void SetImageFactors();

  void SetLidarFactors();

  void SetLidarCameraFactors();

  void Optimize();

  std::vector<vicon_calibration::LidarMeasurement> lidar_measurements_;
  std::vector<vicon_calibration::CameraMeasurement> camera_measurements_;
  std::vector<vicon_calibration::LoopClosureMeasurement> loop_closure_measurements_;
  std::vector<vicon_calibration::CalibrationResult> calibration_results_;
  std::vector<vicon_calibration::CalibrationResult> calibration_initials_;
  std::vector<std::shared_ptr<vicon_calibration::CameraParams>> camera_params_;
  std::vector<std::shared_ptr<vicon_calibration::TargetParams>> target_params_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initials_, initials_updated_, results_;
  std::vector<std::shared_ptr<beam_calibration::CameraModel>> camera_models_;
  std::vector<vicon_calibration::Correspondence> camera_correspondences_;
  std::vector<vicon_calibration::Correspondence> lidar_correspondences_;

  //convergence params:
  bool output_errors_{true};
  double relative_error_tol_{1e-5};
  double absolute_error_tol_{1e-5};
  double error_tol_{1e-9};
  uint16_t max_iterations_{50};
  double current_error_{0}, new_error_{1};

};

} // end namespace vicon_calibration
