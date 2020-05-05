#pragma once

#include "vicon_calibration/measurement_extractors/LidarExtractor.h"

namespace vicon_calibration {

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

class CylinderLidarExtractor : public LidarExtractor {
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
    return LidarExtractorType::CYLINDER;
  };

private:
  void GetKeypoints() override;

  void CheckMeasurementValid() override;

  double icp_transform_epsilon_{1e-8};
  double icp_euclidean_epsilon_{1e-2};
  int icp_max_iterations_{80};
  double icp_max_correspondence_dist_{1};
  double max_keypoint_distance_{0.04};
};

} // namespace vicon_calibration
