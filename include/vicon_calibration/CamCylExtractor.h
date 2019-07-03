#pragma once

#include <Eigen/Geometry>

#include <vector>

#include <beam_calibration/CameraModel.h>

namespace vicon_calibration {

  const int edgeThresh=1;
  const int ratio=3;
  const int kernel_size=3;

/**
 * @brief class for extracting cylinder measurements from images
 */
class CamCylExtractor {
public:
  // Constructor
  CamCylExtractor();

  // Destructor
  ~CamCylExtractor() = default;

  /**
   * @brief Configures camera model object
   * @param intrinsic_file intrinsic calibration file path for camera model
   */
  void ConfigureCameraModel(std::string intrinsic_file);

  /**
   * @brief Extracts cylinder from an image and calculate measurement
   * @param T_CAMERA_TARGET_EST transform from estimated target to camera frame
   * @param image image to extract cylinder from
   * @param measurement_num measurement number identifier to display images
   */
  void ExtractCylinder(Eigen::Affine3d T_CAMERA_TARGET_EST, cv::Mat image,
                       int measurement_num);

  void SetCylinderDimension(double radius, double height);

  void SetThreshold(double threshold);

  std::pair<Eigen::Vector3d, bool> GetMeasurementInfo();

  void LoadIntrinsics(std::string json_file);

private:
  void PopulateCylinderPoints();

  std::vector<Eigen::Vector2d>
  GetBoundingBox(Eigen::Affine3d T_CAMERA_TARGET_EST);

  cv::Mat CropImage(cv::Mat image, Eigen::Vector2d min_vector,
                    Eigen::Vector2d max_vector);

  std::vector<cv::Vec4i> DetectLines(cv::Mat &img, int measurement_num);

  double Median(cv::Mat channel);

  cv::Mat ColorPixelsOnImage(cv::Mat &img, std::vector<Eigen::Vector2d> points);

  double radius_{0.0635};
  double height_{0.5};
  double threshold_{0.05};

  std::vector<Eigen::Vector4d> cylinder_points_;
  std::vector<Eigen::Vector2d> projected_pixels_;
  std::vector<Eigen::Vector2d> center_line_points_;
  std::shared_ptr<beam_calibration::CameraModel> camera_model_;
  Eigen::VectorXd undistorted_intrinsics_{Eigen::VectorXd(4)};
  Eigen::VectorXd distorted_intrinsics_{Eigen::VectorXd(4)};

  Eigen::Matrix3d K_;

  bool measurement_valid_;
  bool measurement_complete_;
  Eigen::Vector3d measurement_;

  double percent_{0.33};
};

} // end namespace vicon_calibration
