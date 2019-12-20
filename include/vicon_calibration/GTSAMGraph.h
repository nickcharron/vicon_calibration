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
  void ViewClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr c1, pcl::PointCloud<pcl::PointXYZ>::Ptr c2);

  bool HasConverged(uint16_t iteration);

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

  // params
  uint16_t max_iterations_{40};
  double max_pixel_cor_dist_{500}; // in pixels
  double max_point_cor_dist_{0.3}; // in m
  std::vector<double> error_tol_{0.0001, 0.0001, 0.0001, 0.0005, 0.0005, 0.0005};
  std::vector<double> image_noise_{5, 5};
  std::vector<double> lidar_noise_{0.02, 0.02, 0.02};
};

} // end namespace vicon_calibration
