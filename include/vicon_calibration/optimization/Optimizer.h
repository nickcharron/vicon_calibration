#pragma once

#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"

#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace vicon_calibration {

/**
 * @brief inputs required for any optimizer
 */
struct OptimizerInputs {
  TargetParamsVector target_params;
  CameraParamsVector camera_params;
  LidarMeasurements lidar_measurements;
  CameraMeasurements camera_measurements;
  LoopClosureMeasurements loop_closure_measurements;
  CalibrationResults calibration_initials;
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

  /**
   * @brief params common to all optimizers
   */
  struct Params {
    int viz_point_size = 3;
    int viz_corr_line_width = 2;
    uint16_t max_correspondence_iterations{40};
    bool show_camera_measurements{false};
    bool show_lidar_measurements{false};
    bool show_loop_closure_correspondences{false};
    bool extract_image_target_perimeter{true};
    bool output_errors{false};
    double concave_hull_alpha{10};
    double max_pixel_cor_dist{500}; // in pixels
    double max_point_cor_dist{0.3}; // in m
    bool match_centroids{true};
    bool match_centroids_on_first_iter_only{false};
    bool print_results_to_terminal{false};
    std::vector<double> error_tol{0.0001, 0.0001, 0.0001,
                                  0.0002, 0.0002, 0.0002};
    std::vector<double> image_noise{20, 20};
    std::vector<double> lidar_noise{0.02, 0.02, 0.02};
    std::vector<double> template_downsample_size{0.003, 0.003, 0.003};
  };

protected:
  virtual void LoadConfig() = 0;

  void LoadConfigCommon(const nlohmann::json& J);

  void ResetViewer();

  void CheckInputs();

  void GetImageCorrespondences();

  void GetLidarCorrespondences();

  void GetLoopClosureCorrespondences();

  PointCloud::Ptr MatchCentroids(const PointCloud::Ptr& source_cloud,
                                 const PointCloud::Ptr& target_cloud);

  void ViewLidarMeasurements(
      const PointCloud::Ptr& c1, const PointCloud::Ptr& c2,
      const boost::shared_ptr<pcl::Correspondences>& correspondences,
      const std::string& c1_name, const std::string& c2_name);

  void ViewCameraMeasurements(
      const PointCloud::Ptr& c1, const PointCloud::Ptr& c2,
      const boost::shared_ptr<pcl::Correspondences>& correspondences,
      const std::string& c1_name, const std::string& c2_name);

  void ConfirmMeasurementKeyboardCallback(
      const pcl::visualization::KeyboardEvent& event, void* viewer_void);

  bool HasConverged(uint16_t iteration);

  virtual void AddInitials() = 0;

  virtual void Clear() = 0;

  virtual Eigen::Matrix4d GetUpdatedInitialPose(SensorType type, int id) = 0;

  virtual Eigen::Matrix4d GetFinalPose(SensorType type, int id) = 0;

  virtual void AddImageMeasurements() = 0;

  virtual void AddLidarMeasurements() = 0;

  virtual void AddLidarCameraMeasurements() = 0;

  virtual void Optimize() = 0;

  virtual void UpdateInitials() = 0;

  OptimizerInputs inputs_;
  Params optimizer_params_;
  std::vector<CalibrationResult> calibration_results_;
  std::vector<Correspondence> camera_correspondences_;
  std::vector<Correspondence> lidar_correspondences_;
  std::vector<LoopCorrespondence> lidar_camera_correspondences_;
  pcl::visualization::PCLVisualizer::Ptr pcl_viewer_;
  bool close_viewer_{false};
  bool skip_to_next_iteration_{false};
  bool stop_all_vis_{false};
};

} // end namespace vicon_calibration
