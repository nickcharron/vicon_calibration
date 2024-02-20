#pragma once

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vicon_calibration/Params.h>
#include <vicon_calibration/camera_models/CameraModel.h>

namespace vicon_calibration {

/**
 * @brief Enum class for different types of camera extractors we can use
 */
enum class CameraExtractorType { CYLINDER = 0, CHECKERBOARD };

/**
 * @brief Abstract class for CameraExtractor
 */
class CameraExtractor {
public:
  CameraExtractor(); // ADD INITIALIZATION OF POINTERS

  virtual ~CameraExtractor() = default;

  /**
   * @brief Factory method to create camera extractor at runtime
   */
  static std::shared_ptr<CameraExtractor> Create(const std::string& type);

  // alias for clarity
  using Ptr = std::shared_ptr<CameraExtractor>;

  /**
   * @brief Get the type of CameraExtractor
   * @return Returns type as string
   */
  virtual std::string GetTypeString() const = 0;

  void SetCameraParams(
      std::shared_ptr<vicon_calibration::CameraParams>& camera_params);

  void SetTargetParams(
      std::shared_ptr<vicon_calibration::TargetParams>& target_params);

  void SetShowMeasurements(bool show_measurements);

  bool GetShowMeasurements();

  void ProcessMeasurement(const Eigen::Matrix4d& T_Camera_Target_Est,
                          const cv::Mat& img_in);

  bool GetMeasurementValid();

  pcl::PointCloud<pcl::PointXY>::Ptr GetMeasurement();

protected:
  // this is what we will need to override in the derived class
  virtual void GetKeypoints() = 0;

  void CheckInputs();

  void TargetPointToPixel(const Eigen::Vector4d& point, Eigen::Vector2d& pixel,
                          bool& projection_valid);

  // this projects all template cloud points into the image plane, gets the min
  // and max coordinates then adds a buffer based on the parameter "crop_image"
  // in the target params. The crop_image is a percent to increase the bounding
  // box by
  void CropImage();

  void DisplayImage(const std::string& display_name,
                    const std::string& output_text, bool allow_override);

  // params:
  double axis_plot_scale_{0.3}; // scale for plotting projected axes on an img
  bool show_measurements_{false};
  std::shared_ptr<vicon_calibration::CameraParams> camera_params_;
  std::shared_ptr<vicon_calibration::TargetParams> target_params_;

  // member variables:
  Eigen::Matrix4d T_Camera_Target_Est_ = Eigen::Matrix4d::Identity();
  pcl::PointCloud<pcl::PointXY>::Ptr keypoints_measured_;
  std::shared_ptr<cv::Mat> image_in_;
  std::shared_ptr<cv::Mat> image_cropped_;
  std::shared_ptr<cv::Mat> image_annotated_;
  bool measurement_complete_{false};
  bool measurement_valid_{false};
  bool target_params_set_{false};
  bool camera_params_set_{false};

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace vicon_calibration
