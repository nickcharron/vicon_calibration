#pragma once

#include "vicon_calibration/utils.hpp"

#include <beam_utils/math.hpp>

#include <Eigen/Geometry>

#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>

#include <thread>
#include <vector>

namespace vicon_calibration {
/**
 * @brief class for extracting cylinder measurements from lidar scan
 */
class LidarCylExtractor {
public:
  LidarCylExtractor() = default;

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
  void SetScanTransform(Eigen::Affine3d T_LIDAR_SCAN);

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

  PointCloud::Ptr GetTemplateCloud() { return template_cloud_; }

  PointCloud::Ptr GetAggregatedCloud() { return scan_; }

  /**
   * @brief Extract cylinder target from the aggregated cloud, then calculate
   * transform from cloud to target
   * @param T_SCAN_TARGET_EST transform from estimated target to scan
   * @param measurement_num measurement number, used for adding clouds to viewer
   * @return 4x1 vector of x,y translation and rotation about x and y axes [tx,
   * ty, ra, ry]^T
   */
  Eigen::Vector4d ExtractCylinder(Eigen::Affine3d T_SCAN_TARGET_EST,
                                  int measurement_num = 0);

  /**
   * @brief Extracts measurements from the input transform
   * @param T_SCAN_TARGET transform from target to scan
   * @return 4x1 vector of x,y translation and rotation about x and y axes [tx,
   * ty, ra, ry]^T
   */
  Eigen::Vector4d ExtractRelevantMeasurements(Eigen::Affine3d T_SCAN_TARGET);

private:
  /**
   * @brief Crop the aggregated cloud to extract cylinder target part
   * @param T_SCAN_TARGET_EST transform from estimated target to scan
   * @return cropped cloud
   */
  PointCloud::Ptr CropPointCloud(Eigen::Affine3d T_SCAN_TARGET_EST);

  // Functions for testing
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

  PointCloud::Ptr template_cloud_;
  PointCloud::Ptr scan_;
  Eigen::Affine3d T_LIDAR_SCAN_;
  double height_{0.5};
  double radius_{0.0635};
  double threshold_{0.015}; // Threshold for cropping the the aggregated cloud

  pcl::visualization::PCLVisualizer pcl_viewer_;
  bool show_transformation_{false};
};

} // end namespace vicon_calibration
