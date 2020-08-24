#pragma once


#include <optional>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"
#include "vicon_calibration/measurement_extractors/IsolateTargetPoints.h"

namespace vicon_calibration {

/**
 * @brief Enum class for different types of camera extractors we can use
 */
enum class LidarExtractorType { CYLINDER = 0, DIAMOND };

/**
 * @brief Abstract class for LidarExtractor
 */
class LidarExtractor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LidarExtractor(); // ADD INITIALIZATION OF POINTERS

  virtual ~LidarExtractor() = default;

  // alias for clarity
  using Ptr = std::shared_ptr<LidarExtractor>;

  /**
   * @brief Get the type of LidarExtractor
   * @return Returns type as one of LidarExtractor types specified in the enum
   * LidarExtractorType
   */
  virtual LidarExtractorType GetType() const = 0;

  void
  SetLidarParams(std::shared_ptr<vicon_calibration::LidarParams> &lidar_params);

  void SetTargetParams(
      std::shared_ptr<vicon_calibration::TargetParams> &target_params);

  void SetShowMeasurements(const bool &show_measurements);

  bool GetShowMeasurements();

  void ProcessMeasurement(const Eigen::Matrix4d &T_LIDAR_TARGET_EST,
                          const PointCloud::Ptr &cloud_in);

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

  void AddColouredPointCloudToViewer(const PointCloudColor::Ptr &cloud,
                                     const std::string &cloud_name,
                                     boost::optional<Eigen::MatrixXd &> T,
                                     int point_size = 1);

  void AddPointCloudToViewer(const PointCloud::Ptr &cloud,
                             const std::string &cloud_name,
                             const Eigen::Matrix4d &T,
                             int point_size = 1);

  void ConfirmMeasurementKeyboardCallback(
      const pcl::visualization::KeyboardEvent &event, void *viewer_void);

  void ShowFailedMeasurement();

  void ShowPassedMeasurement();

  void OutputScans();

  // member variables
  IsolateTargetPoints target_isolator_;
  PointCloud::Ptr scan_in_;
  PointCloud::Ptr scan_isolated_;
  PointCloud::Ptr estimated_template_cloud_;
  PointCloud::Ptr measured_template_cloud_;
  Eigen::MatrixXd T_LIDAR_TARGET_EST_ = Eigen::MatrixXd(4, 4);
  Eigen::MatrixXd T_LIDAR_TARGET_OPT_ = Eigen::MatrixXd(4, 4);
  pcl::visualization::PCLVisualizer::Ptr pcl_viewer_;
  PointCloud::Ptr keypoints_measured_;
  bool measurement_valid_{false};
  bool measurement_complete_{false};
  bool target_params_set_{false};
  bool lidar_params_set_{false};
  bool close_viewer_{false};
  bool white_cloud_on_ = true;
  bool blue_cloud_on_ = true;
  bool green_cloud_on_ = true;
  bool viewer_key_down_ = false;
  PointCloudColor::Ptr blue_cloud_, green_cloud_;
  PointCloud::Ptr white_cloud_;

  // params:
  std::shared_ptr<vicon_calibration::LidarParams> lidar_params_;
  std::shared_ptr<vicon_calibration::TargetParams> target_params_;
  bool show_measurements_{false};
  bool icp_enable_debug_{false};
  double allowable_keypoint_error_{0.03};
  bool output_scans_{true};
  std::string output_directory_{"/home/nick/tmp/"};

};

} // namespace vicon_calibration
