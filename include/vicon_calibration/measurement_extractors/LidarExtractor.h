#pragma once

#include "vicon_calibration/params.h"

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

namespace vicon_calibration {

/**
 * @brief Enum class for different types of camera extractors we can use
 */
enum class LidarExtractorType { CYLINDER = 0, DIAMOND };

/**
 * @brief Class that inherits from pcl's icp with a function to access the
 * final correspondences
 */
template <typename PointSource, typename PointTarget, typename Scalar = float>
class IterativeClosestPointCustom
    : public pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  pcl::CorrespondencesPtr getCorrespondencesPtr() {
    return this->correspondences_;
  }
};

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

  void SetLidarParams(std::shared_ptr<vicon_calibration::LidarParams> &lidar_params);

  void SetTargetParams(std::shared_ptr<vicon_calibration::TargetParams> &target_params);

  void SetShowMeasurements(bool show_measurements);

  virtual void
  ExtractKeypoints(Eigen::Matrix4d &T_LIDAR_TARGET_EST,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in) = 0;

  bool GetMeasurementValid();

  pcl::PointCloud<pcl::PointXYZ>::Ptr GetMeasurement();

protected:
  std::shared_ptr<vicon_calibration::LidarParams> lidar_params_;
  std::shared_ptr<vicon_calibration::TargetParams> target_params_;
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
