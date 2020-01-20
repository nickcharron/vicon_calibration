#pragma once

#include "vicon_calibration/measurement_extractors/LidarExtractor.h"

namespace vicon_calibration {

class CylinderLidarExtractor : public LidarExtractor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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

private:
  void GetKeypoints() override;

  void SaveMeasurement();

  void CalculateErrors();

  void CheckErrors();

  // member variables:
  PointCloud::Ptr scan_best_points_; // points that are corresponding after icp
  Eigen::Vector3d error_; // two angles and a distance
  Eigen::MatrixXd T_LIDAR_TARGET_OPT_ = Eigen::MatrixXd(4, 4);

  // params:
  bool test_registration_{true}; // Whether to use ICP to test that the target
                                 // template can converge to the scan correct
  double max_keypoint_distance_{0.1}; // measurement valid IFF the RELATIVE
                                      // distance between estimated and
                                      // optimized tgt is less than this
  double dist_acceptance_criteria_{0.05}; // acceptable error between estimated
                                          // target and registered target
  double rot_acceptance_criteria_{15}; // acceptable error between estimated
                                         // and optimized center axes (deg)
  double icp_transform_epsilon_{1e-8};
  double icp_euclidean_epsilon_{1e-2};
  int icp_max_iterations_{80};
  double icp_max_correspondence_dist_{1};
};

} // namespace vicon_calibration
