#pragma once

#include "vicon_calibration/measurement_extractors/CameraExtractor.h"

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

private:
  void CheckInputs();

  void DisplayImage(cv::Mat &img, std::string display_name,
                    std::string output_text);

  void UndistortImage();

  bool CropImage();

  Eigen::Vector2d TargetPointToPixel(Eigen::Vector4d point);

  void GetMeasurementPoints();

  void GetMeasuredPose();

  void DrawContourAxis(std::shared_ptr<cv::Mat> &img_pointer, cv::Point p,
                       cv::Point q, cv::Scalar colour, const float scale);

  void GetEstimatedPose();

  void CheckError();

  std::shared_ptr<cv::Mat> image_in_;
  std::shared_ptr<cv::Mat> image_undistorted_;
  std::shared_ptr<cv::Mat> image_cropped_;
  std::shared_ptr<cv::Mat> image_annotated_;
  Eigen::MatrixXd T_CAMERA_TARGET_EST_ = Eigen::MatrixXd(4,4);
  double axis_plot_scale_{0.3}; // scale for plotting projected axes on an img
  cv::Scalar color_threshold_min_{0, 95, 0}; // Min BGR to threshold img for tgt
  cv::Scalar color_threshold_max_{20, 255, 20}; // Max BGR to thresh img for tgt
  std::vector<cv::Point> target_contour_;
  std::pair<cv::Point, double> target_pose_measured_;  // center/angle of
                                                       // measured target
  std::pair<cv::Point, double> target_pose_estimated_; // center/angle of est.
                                                       // (projected) target
  double dist_acceptance_criteria_{300}; // accept measurements if meas. vs.
                                         // est. centers are less than this
                                         // (in pixels)
  double rot_acceptance_criteria_{0.5};  // accept measurements if meas. vs.
                                         // est. angles are less than this
                                         // (in rad)
};

} // namespace vicon_calibration
