#pragma once

#include "vicon_calibration/params.h"
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>

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
      std::vector<std::vector<std::shared_ptr<LidarMeasurement>>>
          &lidar_measurements);

  void SetCameraMeasurements(
      std::vector<std::vector<std::shared_ptr<CameraMeasurement>>>
          &camera_measurements);

  void SetLoopClosureMeasurements(
      std::vector<std::shared_ptr<LoopClosureMeasurement>>
          &loop_closure_measurements_);

  void SetInitialGuess(
      std::vector<vicon_calibration::CalibrationResult> &initial_guess);

  void
  SetCameraParams(std::vector<std::shared_ptr<vicon_calibration::CameraParams>>
                      &camera_params);

  void SolveGraph();

  std::vector<vicon_calibration::CalibrationResult> GetResults();

  void Print(std::string &file_name, bool print_to_terminal);

private:
  bool HasConverged(uint16_t iteration);

  void CheckInputs();

  void Clear();

  void AddInitials();

  void SetImageCorrespondences();

  void SetLidarCorrespondences();

  void SetLoopClosureCorrespondences();

  void SetImageFactors();

  void SetLidarFactors();

  void SetLidarCameraFactors();

  void Optimize();

  void LoadConfig();

  void ViewLidarMeasurements(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &c1,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &c2,
      const boost::shared_ptr<pcl::Correspondences> &correspondences,
      const std::string &c1_name, const std::string &c2_name);

  void ViewCameraMeasurements(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &c1,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &c2,
      const boost::shared_ptr<pcl::Correspondences> &correspondences,
      const std::string &c1_name, const std::string &c2_name);

  void ConfirmMeasurementKeyboardCallback(
      const pcl::visualization::KeyboardEvent &event, void *viewer_void);

  std::vector<std::vector<std::shared_ptr<LidarMeasurement>>>
      lidar_measurements_;
  std::vector<std::vector<std::shared_ptr<CameraMeasurement>>>
      camera_measurements_;
  std::vector<std::shared_ptr<LoopClosureMeasurement>>
      loop_closure_measurements_;
  std::vector<CalibrationResult> calibration_results_;
  std::vector<CalibrationResult> calibration_initials_;
  std::vector<std::shared_ptr<CameraParams>> camera_params_;
  std::vector<std::shared_ptr<TargetParams>> target_params_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initials_, initials_updated_, results_;
  std::vector<Correspondence> camera_correspondences_;
  std::vector<Correspondence> lidar_correspondences_;
  std::vector<LoopCorrespondence> lidar_camera_correspondences_;
  pcl::visualization::PCLVisualizer::Ptr pcl_viewer_;
  bool close_viewer_{false};
  bool skip_to_next_iteration_{false};
  bool stop_all_vis_{false};

  // params
  uint16_t max_iterations_{40};
  bool show_camera_measurements_{false};
  bool show_lidar_measurements_{false};
  bool show_loop_closure_correspondences_{false};
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
  bool print_results_to_terminal_{false};
  // double concave_hull_alpha_{1};
};

} // end namespace vicon_calibration
