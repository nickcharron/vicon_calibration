#pragma once

#include "vicon_calibration/utils.hpp"

#include <beam_utils/math.hpp>

#include <pcl/visualization/pcl_visualizer.h>

namespace vicon_calibration {

/**
 * @brief class for extracting cylinder measurements from lidar scan
 */
class LidarCylExtractor {
public:
  LidarCylExtractor() = default;

  /**
   * @brief Constructor
   * @param template_cloud template pointcloud of the cylinder target
   * @param scan aggregated pointcloud (assume the cloud is already in
   * lidar frame)
   */
  LidarCylExtractor(PointCloud::Ptr &template_cloud,
                    PointCloud::Ptr &scan);

  ~LidarCylExtractor() = default;

  void SetTemplateCloud(PointCloud::Ptr &template_cloud) {
    template_cloud_ = template_cloud;
  }

  void SetScan(PointCloud::Ptr &scan) {
    scan_ = scan;
  }

  /**
   * @brief Transforms the aggregated cloud to lidar frame and store the
   * transform
   */
  void SetScanTransform(beam::Affine3 TA_LIDAR_VICON);

  void SetHeight(double height) { height_ = height; }

  void SetRadius(double radius) { radius_ = radius; }

  void SetThreshold(double threshold) { threshold_ = threshold; }

  void SetShowTransformation(bool show_transformation);

  PointCloud::Ptr GetTemplateCloud() { return template_cloud_; }

  PointCloud::Ptr GetAggregatedCloud() { return scan_; }

  /**
   * @brief Extract cylinder target from the aggregated cloud, then calculate
   * transform from cloud to target
   * @param TA_LIDAR_TARGET transform from target to lidar
   * @param measurement_num measurement number, used for adding clouds to viewer
   * @return transform from the aggrated cloud to measured target
   */
  beam::Vec4 ExtractCylinder(beam::Affine3 TA_LIDAR_TARGET,
                             int measurement_num = 0);

  /**
   * @brief Calculate the transform measurement from aggregated cloud to target
   * @param TA_LIDAR_TARGET transform from target to lidar
   * target
   * @param TA_TARGET_ESTIMATED transform from estimated target to target (ICP)
   * @return transform from the aggrated cloud to measured target
   */
  beam::Vec4 CalculateMeasurement(beam::Affine3 TA_LIDAR_ESTIMATED);

  void ShowFinalTransformation();

private:
  /**
   * @brief Crop the aggregated cloud to extract cylinder target part
   * @param TA_LIDAR_TARGET transform from target to lidar
   * target
   * @return cropped cloud
   */
  PointCloud::Ptr CropPointCloud(beam::Affine3 TA_LIDAR_TARGET);

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
  PointCloudColor::Ptr ColourPointCloud(PointCloud::Ptr &cloud, int r,
                                         int g, int b);

  PointCloud::Ptr template_cloud_;
  PointCloud::Ptr scan_;
  beam::Affine3 TA_LIDAR_VICON_;
  beam::Affine3 TA_LIDAR_ESTIMATED_; // Only used to output results for testing
  double height_{0.5};
  double radius_{0.0635};
  double threshold_{0.015}; // Threshold for cropping the the aggregated cloud

  pcl::visualization::PCLVisualizer pcl_viewer_;
  bool show_transformation_{false};
};

} // end namespace vicon_calibration
