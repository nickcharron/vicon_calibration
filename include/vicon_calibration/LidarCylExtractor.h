#pragma once

#include "vicon_calibration/utils.hpp"

#include <beam_filtering/CropBox.h>
#include <beam_utils/math.hpp>

#include <Eigen/Geometry>

#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <thread>
#include <vector>

namespace vicon_calibration {

const Eigen::Vector4d INVALID_MEASUREMENT(-100, -100, -100, -100);
/**
 * @brief class for extracting cylinder measurements from lidar scan
 */
class LidarCylExtractor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LidarCylExtractor();

  /**
   * @brief Constructor
   * @param template_cloud template pointcloud of the cylinder target. This
   * cloud is in its own frame which is centered at the bottom of the cylinder
   * with the z-axis aligned with the cylinder axis
   * @param scan scan pointcloud
   */
  LidarCylExtractor(PointCloud::Ptr &template_cloud, PointCloud::Ptr &scan);

  ~LidarCylExtractor() = default;

  void SetTemplateCloud(PointCloud::Ptr &template_cloud) {
    template_cloud_ = template_cloud;
  }

  void SetScan(PointCloud::Ptr &scan) { scan_ = scan; }

  /**
   * @brief Transforms the scan to lidar frame and store the transform
   * @param T_LIDAR_SCAN transform from scan to lidar
   */
  void SetScanTransform(Eigen::Affine3d &T_LIDAR_SCAN);

  /**
   * @brief Set height of the cylinder target
   * @param height height dimension to set the private variable to
   */
  void SetHeight(double height) { height_ = height; }

  /**
   * @brief Set radius of the cylinder target
   * @param radius radius dimension to set the private variable to
   */
  void SetRadius(double radius) { radius_ = radius; }

  /**
   * @brief Set threshold used for cropping the scan
   * @param threshold threshold value to set the private variable to
   */
  void SetThreshold(double threshold) { threshold_ = threshold; }

  /**
   * @brief Set the flag for displaying pointclouds
   * @param show_transformation boolean to set the private variable to
   */
  void SetShowTransformation(bool show_transformation);

  /**
   * @brief Get template cloud
   * @return boost::shared_ptr to template cloud
   */
  PointCloud::Ptr GetTemplateCloud() { return template_cloud_; }

  /**
   * @brief Get scan
   * @return boost::shared_ptr to scan
   */
  PointCloud::Ptr GetScan() { return scan_; }

  /**
   * @brief Return measurement_ and measurement_valid_ if extracting measurement
   * has been completed
   * @return a pair of 4x1 vector of x,y translation and rotation about x and y
   * axes [tx, ty, ra, ry]^T and bool indicating if the measurement is valid
   */
  std::pair<Eigen::Vector4d, bool> GetMeasurementInfo();

  /**
   * @brief Extract cylinder target from the aggregated cloud, then calculate
   * transform from cloud to target
   * @param T_SCAN_TARGET_EST transform from estimated target to scan
   * @param measurement_num measurement number, used for adding clouds to viewer
   * @return 4x1 vector of x,y translation and rotation about x and y axes [tx,
   * ty, ra, ry]^T
   */
  void ExtractCylinder(Eigen::Affine3d &T_SCAN_TARGET_EST,
                       int measurement_num = 0);

  /**
   * @brief Extracts measurements from the input transform
   * @param T_SCAN_TARGET transform from target to scan
   * @return 4x1 vector of x,y translation and rotation about x and y axes [tx,
   * ty, ra, ry]^T
   */
  Eigen::Vector4d ExtractRelevantMeasurements(Eigen::Affine3d &T_SCAN_TARGET);

  /**
   * @brief Set ICP registration parameters
   * @param t_eps Transformation epsilon
   * @param fit_eps Eucliedean Fitness Epsilon
   * @param max_corr Maximum correspondence distance
   * @param max_iter Maximum iteration
   */
  void SetICPParameters(double t_eps, double fit_eps, double max_corr,
                        int max_iter);

  /**
   * @brief Sets measurement acceptance criteria for auto acceptance/rejection
   * @param dist_err_criteria maximum distance between optimized and estimated
   * measurements
   * @param rot_err_criteria maximum rotation between optimized and estimated
   * measurements
   */
  void SetMeasurementAcceptanceCriteria(double dist_err_criteria,
                                        double rot_err_criteria) {
    dist_err_criteria_ = dist_err_criteria;
    rot_err_criteria_ = rot_err_criteria;
  }

private:
  /**
   * @brief Crop the aggregated cloud to extract cylinder target part
   * @param T_SCAN_TARGET_EST transform from estimated target to scan
   * @return cropped cloud
   */
  PointCloud::Ptr CropPointCloud(Eigen::Affine3d &T_SCAN_TARGET_EST);

  // Functions for displaying point clouds
  /**
   * @brief Add a coloured cloud to viewer
   * @param cloud pointcloud to add
   * @param cloud_name name of the point cloud
   */
  void AddColouredPointCloudToViewer(PointCloudColor::Ptr cloud,
                                     std::string cloud_name);

  /**
   * @brief Add a cloud to viewer
   * @param cloud pointcloud to add
   * @param cloud_name name of the point cloud
   */
  void AddPointCloudToViewer(PointCloud::Ptr cloud, std::string cloud_name);

  /**
   * @brief Colour a pointcloud
   * @param cloud pointcloud to colour
   * @param r,g,b rgb values of the colour
   * @return coloured pointcloud
   */
  PointCloudColor::Ptr ColourPointCloud(PointCloud::Ptr &cloud, int r, int g,
                                        int b);

  /**
   * @brief Show colored template clouds and cropped cloud
   */
  void ShowFinalTransformation();

  /**
   * @brief Show colored cropped cloud and scan when ICP registration fails
   */
  void ShowFailedMeasurement();

  /**
   * @brief Modifies icp registration configuration
   */
  void ModifyICPConfig();

  /**
   * @brief Keyboard event callback to allow the user to accept or reject final
   * transform measurement
   */
  static void ConfirmMeasurementKeyboardCallback(
      const pcl::visualization::KeyboardEvent &event, void *viewer_void);

  // Variables for extracting cylinder
  PointCloud::Ptr template_cloud_;
  PointCloud::Ptr scan_;
  Eigen::Affine3d T_LIDAR_SCAN_;
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;
  double height_{0.5};
  double radius_{0.0635};
  double threshold_{0.01}; // Threshold for cropping the the aggregated cloud
  double t_eps_{1e-8};
  double fit_eps_{1e-2};
  double max_corr_{1};
  int max_iter_{100};

  pcl::visualization::PCLVisualizer::Ptr pcl_viewer_;
  bool show_measurements_{false};

  beam_filtering::CropBox cropper_;

  // Measurement info
  static bool measurement_valid_;  // For displaying resulted clouds
  static bool measurement_failed_; // For displaying clouds when icp diverges
  bool measurement_complete_{false};
  Eigen::Vector4d measurement_;

  // Measurement acceptance criteria
  double dist_err_criteria_{0.05};
  double rot_err_criteria_{0.523599};
};

} // end namespace vicon_calibration
