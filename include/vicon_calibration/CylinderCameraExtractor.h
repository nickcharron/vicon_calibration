#pragma once

#include "vicon_calibration/CameraExtractor.h"

namespace vicon_calibration {
class CylinderCameraExtractor : public CameraExtractor {
public:
  // Inherit base class constructors
  using CameraExtractor::CameraExtractor;

  ~CylinderCameraExtractor() override = default;

  /**
   * @brief Get the type of CameraExtractor
   * @return Returns type as one of CameraExtractor types specified in the enum
   * CameraExtractorType
   */
  CameraExtractorType GetType() const override {
    return CameraExtractorType::CYLINDER;
  };

  void ExtractKeypoints(Eigen::Matrix4d &T_CAMERA_TARGET_EST,
                           cv::Mat &img_in) override;
};

} // namespace vicon_calibration
