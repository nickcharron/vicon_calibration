#pragma once

#include <vicon_calibration/measurement_extractors/CameraExtractor.h>

#include <libcbdetect/config.h>

namespace vicon_calibration {

/**
 * @brief Enum class for different types of corner extractors
 */
enum class CornerDetectorType { OPENCV = 0, SADDLEPOINT, MONKEYSADDLEPOINT };

class CheckerboardCameraExtractor : public CameraExtractor {
public:
  /**
   * @brief constructor that calls the base class constructor
   * @param type type of corner detector
   */
  CheckerboardCameraExtractor(
      CornerDetectorType type = CornerDetectorType::OPENCV);

  ~CheckerboardCameraExtractor() override = default;

  /**
   * @brief Get the type of CameraExtractor
   * @return Returns type as string
   */
  std::string GetTypeString() const override {
    std::string type = "CHECKERBOARD-";
    if (corner_detector_ == CornerDetectorType::OPENCV) {
      type += "OPENCV";
    } else if (corner_detector_ == CornerDetectorType::SADDLEPOINT) {
      type += "SADDLEPOINT";
    } else if (corner_detector_ == CornerDetectorType::MONKEYSADDLEPOINT) {
      type += "MONKEYSADDLEPOINT";
    }
    return type;
  };

private:
  void GetKeypoints() override;

  void DetectCornersOpenCV();

  void DetectCornersLibCBDetect();

  void PlotLibCBDetectCorners(const cbdetect::Corner& corners);

  CornerDetectorType corner_detector_{CornerDetectorType::OPENCV};
};

} // namespace vicon_calibration
