#pragma once

#include "vicon_calibration/params.h"
#include <Eigen/Geometry>
#include <beam_calibration/CameraModel.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace vicon_calibration {

/**
 * @brief class for extracting cylinder measurements from images
 */
class CamCylExtractor {
public:
  // Constructor
  CamCylExtractor();

  // Destructor
  ~CamCylExtractor() = default;

  void SetTargetParams(vicon_calibration::CylinderTgtParams &params);

  void
  SetImageProcessingParams(vicon_calibration::ImageProcessingParams &params);

  void SetCameraParams(vicon_calibration::CameraParams &params);

  /**
   * @brief Extracts cylinder from an image and calculate measurement
   * @param T_CAMERA_TARGET_EST transform from estimated target to camera frame
   * @param image image to extract cylinder from
   */
  void ExtractMeasurement(Eigen::Affine3d T_CAMERA_TARGET_EST, cv::Mat &image);

  /**
   * @brief Returns the results of cylinder extraction
   * @return a pair of Eigen vector 3d (u,v,angle) and boolean to indicate if
   *         the measurement is valid or not
   */
  std::shared_ptr<std::vector<cv::Point>> GetMeasurements();

  bool GetMeasurementsValid();

  /**
   * @brief Gets calculated distance and rotation errors after extraction
   * @return a pair of distance error and rotation error
   */
  std::pair<double, double> GetErrors();

private:

  void CheckInputs();

  void UndistortImage();

  bool CropImage();

  void GetMeasurementPoints();

  void GetMeasuredPose();

  void GetEstimatedPose();

  void CheckError();

  void DrawContourAxis(std::shared_ptr<cv::Mat> &img_pointer, cv::Point p,
                       cv::Point q, cv::Scalar colour, const float scale);

  /**
   * @brief Transforms and projects a homogeneous cylinder point to 2d pixel
   * @param point point to project
   * @return Vector2d pixel that has been projected in camera frame
   */
  Eigen::Vector2d TargetPointToPixel(Eigen::Vector4d point);

  void DisplayResult();

  Eigen::Affine3d T_CAMERA_TARGET_EST_;
  vicon_calibration::ImageProcessingParams image_processing_params_;
  vicon_calibration::CylinderTgtParams target_params_;
  vicon_calibration::CameraParams camera_params_;
  bool target_params_set_{false}, camera_params_set_{false},
      measurement_valid_{true}, measurement_complete_{false};
  double dist_err_, rot_err_;
  std::shared_ptr<cv::Mat> image_in_, image_undistorted_, image_cropped_,
      image_annotated_;
  std::shared_ptr<beam_calibration::CameraModel> camera_model_;
  std::shared_ptr<std::vector<cv::Point>> target_contour_;
  std::pair<cv::Point, double> target_pose_measured_, target_pose_estimated_;
};

} // end namespace vicon_calibration
