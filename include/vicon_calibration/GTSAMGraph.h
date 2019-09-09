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

  void SetLidarMeasurements(
      std::vector<vicon_calibration::LidarMeasurement> &lidar_measurements);

  void SetCameraMeasurements(
      std::vector<vicon_calibration::CameraMeasurement> &camera_measurements);

  void SetInitialGuess(
      std::vector<vicon_calibration::CalibrationResult> &initial_guess);

  void SolveGraph();

  std::vector<vicon_calibration::CalibrationResult> GetResults();

  void Print(std::string &file_name, bool print_to_terminal);

private:
  void AddInitials();

  void AddLidarMeasurements();

  void AddImageMeasurements();

  void Clear();

  void Optimize();

  std::vector<vicon_calibration::LidarMeasurement> lidar_measurements_;
  std::vector<vicon_calibration::CameraMeasurement> camera_measurements_;
  std::vector<vicon_calibration::CalibrationResult> calibration_results_,
      calibration_initials_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initials_, results_;
};

} // end namespace vicon_calibration
