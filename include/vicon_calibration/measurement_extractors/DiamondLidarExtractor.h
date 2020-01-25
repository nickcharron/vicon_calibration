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

};

} // namespace vicon_calibration
