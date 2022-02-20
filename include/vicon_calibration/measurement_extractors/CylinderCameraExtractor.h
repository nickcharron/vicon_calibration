#pragma once

#include <vicon_calibration/measurement_extractors/CameraExtractor.h>

namespace vicon_calibration {
class CylinderCameraExtractor : public CameraExtractor {
public:
  // Inherit base class constructors
  using CameraExtractor::CameraExtractor;

  ~CylinderCameraExtractor() override = default;

  /**
   * @brief Get the type of CameraExtractor
   * @return Returns type as string
   */
  std::string GetTypeString() const override { return "CYLINDER"; };

private:
  void GetKeypoints() override;

  void CheckError();

  void GetEstimatedArea();

  void DrawContourAxis(std::shared_ptr<cv::Mat>& img_pointer,
                       const cv::Point& p, const cv::Point& q,
                       const cv::Scalar& colour, const float scale);

  void DisplayImagePair(const cv::Mat& img1, const cv::Mat& img2,
                        const std::string& display_name,
                        const std::string& output_text,
                        bool allow_override);

  // params:
  cv::Scalar color_threshold_min_{0, 95, 0}; // Min BGR to threshold img for tgt
  cv::Scalar color_threshold_max_{20, 255, 20}; // Max BGR to thresh img for tgt
  double area_error_allowance_{0.3}; // allowed area difference in percent

  // member variables:
  std::vector<cv::Point> target_contour_;
  double area_expected_;
  double area_detected_;
};

} // namespace vicon_calibration
