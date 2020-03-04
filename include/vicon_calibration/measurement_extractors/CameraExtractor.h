#pragma once

#include "vicon_calibration/params.h"
#include <Eigen/Geometry>
#include <beam_calibration/CameraModel.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vicon_calibration {

/**
 * @brief Enum class for different types of camera extractors we can use
 */
enum class CameraExtractorType { CYLINDER = 0, DIAMOND };

/**
 * @brief Abstract class for CameraExtractor
 */
class CameraExtractor {
public:
  CameraExtractor(); // ADD INITIALIZATION OF POINTERS

  virtual ~CameraExtractor() = default;

  // alias for clarity
  using Ptr = std::shared_ptr<CameraExtractor>;

  /**
   * @brief Get the type of CameraExtractor
   * @return Returns type as one of CameraExtractor types specified in the enum
   * CameraExtractorType
   */
  virtual CameraExtractorType GetType() const = 0;

  void SetCameraParams(
      std::shared_ptr<vicon_calibration::CameraParams> &camera_params);

  void SetTargetParams(
      std::shared_ptr<vicon_calibration::TargetParams> &target_params);

  void SetShowMeasurements(const bool &show_measurements);

  bool GetShowMeasurements();

  void ProcessMeasurement(const Eigen::Matrix4d &T_CAMERA_TARGET_EST,
                          const cv::Mat &img_in);

  bool GetMeasurementValid();

  pcl::PointCloud<pcl::PointXY>::Ptr GetMeasurement();

protected:
  // this is what we will need to override in the derived class
  virtual void GetKeypoints() = 0;

  void CheckInputs();

  Eigen::Vector2d TargetPointToPixel(const Eigen::Vector4d &point);

  void CropImage();

  void UndistortImage();

  void DisplayImage(const cv::Mat &img, const std::string &display_name,
                    const std::string &output_text, const bool &allow_override);

  // params:
  double axis_plot_scale_{0.3}; // scale for plotting projected axes on an img
  bool crop_image_{true};
  bool show_measurements_{false};
  std::shared_ptr<vicon_calibration::CameraParams> camera_params_;
  std::shared_ptr<vicon_calibration::TargetParams> target_params_;

  // member variables:
  Eigen::MatrixXd T_CAMERA_TARGET_EST_ = Eigen::MatrixXd(4, 4);
  pcl::PointCloud<pcl::PointXY>::Ptr keypoints_measured_;
  std::shared_ptr<cv::Mat> image_in_;
  std::shared_ptr<cv::Mat> image_undistorted_;
  std::shared_ptr<cv::Mat> image_cropped_;
  std::shared_ptr<cv::Mat> image_annotated_;
  bool measurement_complete_{false};
  bool measurement_valid_{false};
  bool target_params_set_{false};
  bool camera_params_set_{false};

};

} // namespace vicon_calibration
