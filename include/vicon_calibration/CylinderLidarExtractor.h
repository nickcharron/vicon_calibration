#pragma once

#include "vicon_calibration/LidarExtractor.h"

namespace vicon_calibration {
class CylinderLidarExtractor : public LidarExtractor {
public:
  // Inherit base class constructors
  using LidarExtractor::LidarExtractor;

  ~CylinderLidarExtractor() override = default;

  /**
   * @brief Get the type of LidarExtractor
   * @return Returns type as one of LidarExtractor types specified in the enum
   * LidarExtractorType
   */
  LidarExtractorType GetType() const override {
    return LidarExtractorType::CYLINDER;
  };

  void ExtractKeypoints(Eigen::Matrix4d &T_LIDAR_TARGET_EST,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in) override;
};

} // namespace vicon_calibration
