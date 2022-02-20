#pragma once

#include <vicon_calibration/measurement_extractors/LidarExtractor.h>

namespace vicon_calibration {

class CheckerboardCornersLidarExtractor : public LidarExtractor {
public:
  // Inherit base class constructors
  using LidarExtractor::LidarExtractor;

  /**
   * @brief Get the type of LidarExtractor
   * @return Returns type as string
   */
  std::string GetTypeString() const override {
    return "CHECKERBOARD-CORNERS";
  };

private:
  void GetKeypoints() override;

  void CheckMeasurementValid() override;

  double icp_transform_epsilon_{1e-8};
  double icp_euclidean_epsilon_{1e-2};
  int icp_max_iterations_{50};
  double icp_max_correspondence_dist_{1};
  bool icp_enable_debug_{false};
  double concave_hull_alpha_{0.1};
};

} // namespace vicon_calibration
