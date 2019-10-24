#pragma once

#include "vicon_calibration/params.h"

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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

  void SetLidarParams(LidarParams &camera_params);

  void SetTargetParams(TargetParams &target_params);

  void SetShowMeasurements(bool show_measurements);

  virtual void
  ExtractKeypoints(Eigen::Matrix4d &T_LIDAR_TARGET_EST,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in) = 0;

  bool GetMeasurementValid();

  pcl::PointCloud<pcl::PointXYZ>::Ptr GetMeasurement();

protected:
  LidarParams lidar_params_;
  TargetParams target_params_;
  bool crop_scan_{true};
  std::string template_cloud_path_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_measured_;
  bool measurement_valid_{false};
  bool measurement_complete_{false};
  bool target_params_set_{false};
  bool lidar_params_set_{false};
  bool show_measurements_{false};
};

} // namespace vicon_calibration
