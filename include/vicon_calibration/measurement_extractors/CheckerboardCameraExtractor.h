#pragma once

#include <vicon_calibration/measurement_extractors/CameraExtractor.h>

namespace vicon_calibration {
class CheckerboardCameraExtractor : public CameraExtractor {
public:
  // Inherit base class constructors
  using CameraExtractor::CameraExtractor;

  ~CheckerboardCameraExtractor() override = default;

  /**
   * @brief Get the type of CameraExtractor
   * @return Returns type as one of CameraExtractor types specified in the enum
   * CameraExtractorType
   */
  CameraExtractorType GetType() const override {
    return CameraExtractorType::DIAMOND;
  };

private:
  void GetKeypoints() override;

};

} // namespace vicon_calibration
