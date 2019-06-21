#pragma once

#include <Eigen/Geometry>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include <beam_calibration/PinholeCamera.h>

namespace vicon_calibration {

/**
 * @brief class for extracting cylinder measurements from images
 */
class CamCylExtractor {
public:
  CamCylExtractor();
  ~CamCylExtractor() = default;

  void ConfigureCameraModel(std::string intrinsic_file);

  void ExtractCylinder(Eigen::Affine3d T_CAMERA_TARGET_EST, cv::Mat image,
                       int measurement_num);

  void SetCylinderDimension(double radius, double height);

  void SetThreshold(double cropping_offset);

  std::pair<Eigen::Vector3d, bool> GetMeasurementInfo();

  void LoadIntrinsics(std::string json_file);

private:
  std::vector<Eigen::Vector2d>
  GetBoundingBox(Eigen::Affine3d T_CAMERA_TARGET_EST);

  cv::Mat CropImage(cv::Mat image, Eigen::Vector2d min_vector,
                    Eigen::Vector2d max_vector);

  void PopulateCylinderPoints();

  bool CheckPixelWithinRange(Eigen::Vector2d pixel);

  cv::Mat ColorPixelsOnImage(cv::Mat &img);

  double radius_{0.0635};
  double height_{0.5};
  double threshold_{0.05};

  std::vector<Eigen::Vector4d> cylinder_points_;
  std::vector<Eigen::Vector2d> projected_pixels_;
  std::shared_ptr<beam_calibration::CameraModel> camera_model_;

  Eigen::Matrix3d K_;

  bool measurement_valid_;
  bool measurement_complete_;
  Eigen::Vector3d measurement_;
};

} // end namespace vicon_calibration
