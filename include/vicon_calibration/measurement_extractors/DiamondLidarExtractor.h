#pragma once

#include "vicon_calibration/measurement_extractors/LidarExtractor.h"

namespace vicon_calibration {

class DiamondLidarExtractor : public LidarExtractor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Inherit base class constructors
  using LidarExtractor::LidarExtractor;

  ~DiamondLidarExtractor() override = default;

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

  // params:
  bool test_registration_{true}; // Whether to use ICP to test that the target
                                 // template can converge to the scan correctly
  double max_keypoint_distance_{0.005}; // keypoints will only be the taken when
                                        // the correspondence distance with the
                                        // est. tgt. loc. is less than this
  double dist_acceptance_criteria_{0.05}; // acceptable error between estimated
                                          // target and registered target
  double icp_transform_epsilon_{1e-8};
  double icp_euclidean_epsilon_{1e-2};
  int icp_max_iterations_{80};
  double icp_max_correspondence_dist_{1};

};

} // namespace vicon_calibration
