#pragma once

#include "vicon_calibration/params.h"
#include <beam_calibration/CameraModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <pcl/registration/correspondence_estimation.h>

namespace vicon_calibration {

/**
 * @brief class for building and storing the GTSAM graph used in
 * vicon_calibration
 */
class Graph {
public:
  Graph() = default;
  ~Graph() = default;

  void
  SetTargetParams(std::vector<std::shared_ptr<vicon_calibration::TargetParams>>
                      &target_params);

  void SetLidarMeasurements(
      std::vector<vicon_calibration::LidarMeasurement> &lidar_measurements);

  void SetCameraMeasurements(
      std::vector<vicon_calibration::CameraMeasurement> &camera_measurements);

  void SetInitialGuess(
      std::vector<vicon_calibration::CalibrationResult> &initial_guess);

  void
  SetCameraParams(std::vector<std::shared_ptr<vicon_calibration::CameraParams>>
                      &camera_params);

  void SolveGraph();

  std::vector<vicon_calibration::CalibrationResult> GetResults();

  void Print(std::string &file_name, bool print_to_terminal);

private:
  void ViewClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr c1,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr c2);

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

  void ViewCameraMeasurements(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &c1,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &c2,
      const boost::shared_ptr<pcl::Correspondences> &correspondences);

  std::vector<vicon_calibration::LidarMeasurement> lidar_measurements_;
  std::vector<vicon_calibration::CameraMeasurement> camera_measurements_;
  std::vector<vicon_calibration::LoopClosureMeasurement>
      loop_closure_measurements_;
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
  bool show_camera_measurements_{true};
  bool extract_image_target_perimeter_{true};
  double concave_hull_alpha_{10};
  double max_pixel_cor_dist_{500}; // in pixels
  double max_point_cor_dist_{0.3}; // in m
  double abs_error_tol_{1e-9};
  double rel_error_tol_{1e-9};
  double lambda_upper_bound_{1e8}; // the maximum lambda to try before assuming
                                   // the optimization has failed (default: 1e5)

  std::vector<double> error_tol_{0.0001, 0.0001, 0.0001,
                                 0.0002, 0.0002, 0.0002};
  std::vector<double> image_noise_{20, 20};
  std::vector<double> lidar_noise_{0.02, 0.02, 0.02};
  std::vector<double> template_downsample_size_{0.003, 0.003, 0.003};
  // double concave_hull_alpha_{1};
};

} // end namespace vicon_calibration
