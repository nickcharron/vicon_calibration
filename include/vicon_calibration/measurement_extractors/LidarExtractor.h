#pragma once

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/visualization/pcl_visualizer.h>

#include <vicon_calibration/Params.h>
#include <vicon_calibration/Utils.h>
#include <vicon_calibration/Visualizer.h>
#include <vicon_calibration/measurement_extractors/IsolateTargetPoints.h>

static bool show_measurements_default{false};

namespace vicon_calibration {

/**
 * @brief Class that inherits from pcl's icp with a function to access the
 * final correspondences
 */
template <typename PointSource, typename PointTarget, typename Scalar = float>
class IterativeClosestPointCustom
    : public pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar> {
public:
  pcl::CorrespondencesPtr getCorrespondencesPtr() {
    return this->correspondences_;
  }
};

/**
 * @brief Enum class for different types of camera extractors we can use
 */
enum class LidarExtractorType { CYLINDER = 0, DIAMONDCORNERS, DIAMOND };

/**
 * @brief Abstract class for LidarExtractor
 */
class LidarExtractor {
public:
  LidarExtractor(
      const std::shared_ptr<vicon_calibration::LidarParams>& lidar_params,
      const std::shared_ptr<vicon_calibration::TargetParams>& target_params,
      const bool& show_measurements,
      const std::shared_ptr<Visualizer> pcl_viewer);

  virtual ~LidarExtractor() = default;

  // alias for clarity
  using Ptr = std::shared_ptr<LidarExtractor>;

  /**
   * @brief Get the type of LidarExtractor
   * @return Returns type as one of LidarExtractor types specified in the enum
   * LidarExtractorType
   */
  virtual LidarExtractorType GetType() const = 0;

  void ProcessMeasurement(const Eigen::Matrix4d& T_LIDAR_TARGET_EST,
                          const PointCloud::Ptr& cloud_in,
                          bool& show_measurements = show_measurements_default);

  bool GetMeasurementValid();

  PointCloud::Ptr GetMeasurement();

protected:
  // This needs to be overriden in the derived class.
  // It should set member variable: keypoints_measured_
  virtual void GetKeypoints() = 0;

  // This needs to be overriden in the derived class.
  // It should set member variable measurement_valid_
  virtual void CheckMeasurementValid() = 0;

  void LoadConfig();

  void SetupVariables();

  void CheckInputs();

  void IsolatePoints();

  // if enabled, show result to user and let them accept or decline
  void GetUserInput();

  void OutputScans();

  // member variables
  IsolateTargetPoints target_isolator_;
  PointCloud::Ptr scan_in_;
  PointCloud::Ptr scan_isolated_;
  PointCloud::Ptr estimated_template_cloud_;
  PointCloud::Ptr measured_template_cloud_;
  Eigen::Matrix4d T_LIDAR_TARGET_EST_;
  Eigen::Matrix4d T_LIDAR_TARGET_OPT_;
  std::shared_ptr<Visualizer> pcl_viewer_;
  PointCloud::Ptr keypoints_measured_;
  bool measurement_valid_{true};
  bool measurement_complete_{false};
  bool target_params_set_{false};
  bool lidar_params_set_{false};

  // params:
  std::shared_ptr<vicon_calibration::LidarParams> lidar_params_;
  std::shared_ptr<vicon_calibration::TargetParams> target_params_;
  bool show_measurements_{false};
  bool icp_enable_debug_{false};
  double allowable_keypoint_error_{0.03};
  bool output_scans_{true};
  std::string output_directory_{"/home/nick/results/vicon_calibration/debug/lidar_extractor/"};

  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace vicon_calibration
