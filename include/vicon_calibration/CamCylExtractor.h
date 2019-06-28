#pragma once

#include <Eigen/Geometry>
#include <pcl/visualization/pcl_visualizer.h>

#include <vector>

#include <beam_calibration/CameraModel.h>

namespace vicon_calibration {

  const int edgeThresh=1;
  const int max_lowThreshold=100;
  const int ratio=3;
  const int kernel_size=3;
  std::string window_name;

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
  std::vector<Eigen::Vector2d>
  GetBoundingBox(Eigen::Affine3d T_CAMERA_TARGET_EST);

  cv::Mat CropImage(cv::Mat image, Eigen::Vector2d min_vector,
                    Eigen::Vector2d max_vector);

  void DetectEdges(cv::Mat &img, int measurement_num);

  void PopulateCylinderPoints();

  static void CannyThreshold(int, void*);

  cv::Mat ColorPixelsOnImage(cv::Mat &img);

  double radius_{0.0635};
  double height_{0.5};
  double threshold_{0.05};

  std::vector<Eigen::Vector4d> cylinder_points_;
  std::vector<Eigen::Vector2d> projected_pixels_;
  std::shared_ptr<beam_calibration::CameraModel> camera_model_;

  Eigen::Matrix3d K_;

  pcl::visualization::PCLVisualizer::Ptr pcl_viewer_;

  bool measurement_valid_;
  bool measurement_complete_;
  Eigen::Vector3d measurement_;

  static cv::Mat dst_;
  static cv::Mat detected_edges_;
  static cv::Mat srg_gray_;
  static cv::Mat src_;

  static int lowThreshold;
};

} // end namespace vicon_calibration
