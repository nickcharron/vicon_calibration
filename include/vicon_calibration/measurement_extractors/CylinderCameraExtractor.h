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

private:
  void GetKeypoints() override;

  void GetMeasuredPose();

  void GetEstimatedPose();

  void CheckError();

  void DrawContourAxis(std::shared_ptr<cv::Mat> &img_pointer, cv::Point &p,
                       cv::Point &q, const cv::Scalar &colour,
                       const float scale);

  // params:
  cv::Scalar color_threshold_min_{0, 95, 0}; // Min BGR to threshold img for tgt
  cv::Scalar color_threshold_max_{20, 255, 20}; // Max BGR to thresh img for tgt
  double dist_acceptance_criteria_{300}; // accept measurements if meas. vs.
                                         // est. centers are less than this
                                         // (in pixels)
  double rot_acceptance_criteria_{0.5};  // accept measurements if meas. vs.
                                         // est. angles are less than this (rad)

  // member variables:
  double dist_err_;
  double rot_err_;
  std::vector<cv::Point> target_contour_;
  std::pair<cv::Point, double> target_pose_measured_;  // center/angle of
                                                       // measured target
  std::pair<cv::Point, double> target_pose_estimated_; // center/angle of est.
                                                       // (projected) target
};

} // namespace vicon_calibration
