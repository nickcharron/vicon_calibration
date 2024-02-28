#pragma once

#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <vicon_calibration/Params.h>
#include <vicon_calibration/Utils.h>

namespace vicon_calibration {

/**
 * @brief inputs required for any optimizer
 */
struct OptimizerInputs {
  TargetParamsVector target_params;
  CameraParamsVector camera_params;
  LidarMeasurements lidar_measurements;
  CameraMeasurements camera_measurements;
  CalibrationResults calibration_initials;
  std::string optimizer_config_path;
  std::string ceres_config_path;
};

/**
 * @brief base class for solving the optimization problem which defines the
 * interface for any optimizer of choice.
 */
class Optimizer {
public:
  Optimizer(const OptimizerInputs& inputs);

  ~Optimizer() = default;

  void Solve();

  CalibrationResults GetResults();

  virtual std::vector<Eigen::Matrix4d> GetTargetCameraCorrections() = 0;

  virtual std::vector<Eigen::Matrix4d> GetTargetLidarCorrections() = 0;

  /**
   * @brief params common to all optimizers
   */
  struct Params {
    int viz_point_size = 3;
    int viz_corr_line_width = 2;
    std::vector<double> viewer_backround_color{0, 0, 0};
    uint16_t max_correspondence_iterations{40};
    bool show_camera_measurements{false};
    bool show_lidar_measurements{false};
    bool extract_image_target_perimeter{true};
    bool output_errors{false};
    double concave_hull_alpha{10};
    double max_pixel_cor_dist{500}; // in pixels
    double max_point_cor_dist{0.3}; // in m
    bool match_centroids{true};
    bool match_centroids_on_first_iter_only{true};
    bool print_results_to_terminal{false};
    bool estimate_target_lidar_corrections{true};
    bool estimate_target_camera_corrections{true};
    std::vector<double> error_tol{0.0001, 0.0002};
    std::vector<double> template_downsample_size{0.003, 0.003, 0.003};
  };

protected:
  void LoadConfigCommon(const nlohmann::json& J);

  void ResetViewer();

  void CheckInputs();

  void GetImageCorrespondences();

  void GetLidarCorrespondences();

  PointCloud::Ptr MatchCentroids(const PointCloud::Ptr& source_cloud,
                                 const PointCloud::Ptr& target_cloud);

  void ViewLidarMeasurements(
      const PointCloud::Ptr& c1, const PointCloud::Ptr& c2,
      const std::shared_ptr<pcl::Correspondences>& correspondences,
      const std::string& c1_name, const std::string& c2_name);

  void ViewCameraMeasurements(
      const PointCloud::Ptr& c1, const PointCloud::Ptr& c2,
      const std::shared_ptr<pcl::Correspondences>& correspondences,
      const std::string& c1_name, const std::string& c2_name);

  void ConfirmMeasurementKeyboardCallback(
      const pcl::visualization::KeyboardEvent& event, void* viewer_void);

  bool HasConverged(uint16_t iteration);

  virtual void AddInitials() = 0;

  virtual void Reset() = 0;

  virtual Eigen::Matrix4d GetUpdatedInitialPose(SensorType type, int id) = 0;

  virtual Eigen::Matrix4d GetFinalPose(SensorType type, int id) = 0;

  virtual void AddImageMeasurements() = 0;

  virtual void AddLidarMeasurements() = 0;

  virtual void Optimize() = 0;

  virtual void UpdateInitials() = 0;

  OptimizerInputs inputs_;
  Params optimizer_params_;
  std::vector<CalibrationResult> calibration_results_;
  std::vector<Correspondence> camera_correspondences_;
  std::vector<Correspondence> lidar_correspondences_;
  pcl::visualization::PCLVisualizer::Ptr pcl_viewer_;
  bool close_viewer_{false};
  bool skip_to_next_iteration_{false};
  bool stop_all_vis_{false};
};

} // end namespace vicon_calibration
