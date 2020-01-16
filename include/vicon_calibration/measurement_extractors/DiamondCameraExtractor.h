#pragma once

#include "vicon_calibration/measurement_extractors/CameraExtractor.h"

namespace vicon_calibration {
class DiamondCameraExtractor : public CameraExtractor {
public:
  // Inherit base class constructors
  using CameraExtractor::CameraExtractor;

  ~DiamondCameraExtractor() override = default;

  /**
   * @brief Get the type of CameraExtractor
   * @return Returns type as one of CameraExtractor types specified in the enum
   * CameraExtractorType
   */
  CameraExtractorType GetType() const override {
    return CameraExtractorType::DIAMOND;
  };

  void ExtractKeypoints(Eigen::Matrix4d &T_CAMERA_TARGET_EST,
                        cv::Mat &img_in) override;

private:
  void CheckInputs();

  void DisplayImage(cv::Mat &img, std::string display_name,
                    std::string output_text);

  void UndistortImage();

  bool CropImage();

  Eigen::Vector2d TargetPointToPixel(Eigen::Vector4d point);

  void GetMeasurementPoints();

  // unchanged:
  std::shared_ptr<cv::Mat> image_in_;
  std::shared_ptr<cv::Mat> image_undistorted_;
  std::shared_ptr<cv::Mat> image_cropped_;
  std::shared_ptr<cv::Mat> image_annotated_;
  Eigen::MatrixXd T_CAMERA_TARGET_EST_ = Eigen::MatrixXd(4,4);
  double axis_plot_scale_{0.3}; // scale for plotting projected axes on an img

  // removed
  // cv::Scalar color_threshold_min_{0, 95, 0};
  // cv::Scalar color_threshold_max_{20, 255, 20};
  // std::pair<cv::Point, double> target_pose_measured_;
  // std::pair<cv::Point, double> target_pose_estimated_;
  // double dist_acceptance_criteria_{300};
  // double rot_acceptance_criteria_{0.5};
};

} // namespace vicon_calibration
