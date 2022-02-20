#pragma once

#include <vicon_calibration/measurement_extractors/LidarExtractor.h>

namespace vicon_calibration {

class CheckerboardLidarExtractor : public LidarExtractor {
public:
  // Inherit base class constructors
  using LidarExtractor::LidarExtractor;

  /**
   * @brief Get the type of LidarExtractor
   * @return Returns type as string
   */
  std::string GetTypeString() const override { return "CHECKERBOARD-ALL"; };

private:
  void GetKeypoints() override;

  void CheckMeasurementValid() override;

  double icp_transform_epsilon_{1e-8};
  double icp_euclidean_epsilon_{1e-2};
  int icp_max_iterations_{50};
  double icp_max_correspondence_dist_{0.2};
  bool icp_enable_debug_{false};
  int min_num_keypoints_{30};
};

} // namespace vicon_calibration