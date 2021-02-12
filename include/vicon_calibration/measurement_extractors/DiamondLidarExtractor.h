#pragma once

#include "vicon_calibration/measurement_extractors/LidarExtractor.h"

namespace vicon_calibration {

/** @brief This is a mock class with the sole purpose of accessing a protected
 * member of a class it inherits from.
 *
 * Some of the relevant documentation for correspondences:
 * http://docs.pointclouds.org/trunk/correspondence_8h_source.html#l00092
 */
class IterativeClosestPoint_Exposed
    : public pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ, float> {
public:
  pcl::CorrespondencesPtr getCorrespondencesPtr() {
    return this->correspondences_;
  }
};

class DiamondLidarExtractor : public LidarExtractor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Inherit base class constructors
  using LidarExtractor::LidarExtractor;

  /**
   * @brief Get the type of LidarExtractor
   * @return Returns type as one of LidarExtractor types specified in the enum
   * LidarExtractorType
   */
  LidarExtractorType GetType() const override {
    return LidarExtractorType::DIAMONDCORNERS;
  };

private:
  void GetKeypoints() override;

  void CheckMeasurementValid() override;

  double icp_transform_epsilon_{1e-8};
  double icp_euclidean_epsilon_{1e-2};
  int icp_max_iterations_{50};
  double icp_max_correspondence_dist_{0.1};
  bool icp_enable_debug_{false};
  int min_num_keypoints_{30};
};

} // namespace vicon_calibration
