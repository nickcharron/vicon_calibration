#pragma once

#include <beam_utils/math.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace vicon_calibration {

using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudXYZRGB = pcl::PointCloud<pcl::PointXYZRGB>;

/**
 * @brief class for extracting cylinder measurements from lidar scan
 */
class LidarCylExtractor {
public:
  LidarCylExtractor() = default;

  /**
   * @brief Constructor
   * @param template_cloud template pointcloud of the cylinder target
   * @param agg_cloud aggregated pointcloud (assume the cloud is already in
   * lidar frame)
   */
  LidarCylExtractor(PointCloudXYZ::Ptr &template_cloud,
                    PointCloudXYZ::Ptr &agg_cloud);

  ~LidarCylExtractor() = default;

  void SetTemplateCloud(PointCloudXYZ::Ptr &template_cloud) {
    template_cloud_ = template_cloud;
  }

  void SetAggregatedCloud(PointCloudXYZ::Ptr &agg_cloud) {
    agg_cloud_ = agg_cloud;
  }

  /**
   * @brief Transforms the aggregated cloud to lidar frame and store the
   * transform
   */
  void SetAggregatedCloudTransform(beam::Affine3 TA_LIDAR_VICON);

  void SetHeight(double height) { height_ = height; }

  void SetRadius(double radius) { radius_ = radius; }

  void SetThreshold(double threshold) { threshold_ = threshold; }

  void SetShowTransformation(bool show_transformation);

  PointCloudXYZ::Ptr GetTemplateCloud() { return template_cloud_; }

  PointCloudXYZ::Ptr GetAggregatedCloud() { return agg_cloud_; }

  /**
   * @brief Extract cylinder target from the aggregated cloud, then calculate
   * transform from cloud to target
   * @param TA_LIDAR_TARGET transform from target to lidar
   * @param measurement_num measurement number, used for adding clouds to viewer
   * @return transform from the aggrated cloud to measured target
   */
  beam::Vec4 ExtractCylinder(beam::Affine3 TA_LIDAR_TARGET,
                             int measurement_num = 0);

  void ShowFinalTransformation();

private:
  /**
   * @brief Crop the aggregated cloud to extract cylinder target part
   * @param TA_LIDAR_TARGET transform from target to lidar
   * target
   * @return cropped cloud
   */
  PointCloudXYZ::Ptr CropPointCloud(beam::Affine3 TA_LIDAR_TARGET);

  /**
   * @brief Calculate the transform measurement from aggregated cloud to target
   * @param TA_LIDAR_TARGET transform from target to lidar
   * target
   * @param TA_TARGET_ESTIMATED transform from estimated target to target (ICP)
   * @return transform from the aggrated cloud to measured target
   */
  beam::Vec4 CalculateMeasurement(beam::Affine3 TA_LIDAR_TARGET,
                                  beam::Affine3 TA_TARGET_ESTIMATED);

  // Functions for testing
  /**
   * @brief Add a coloured cloud to viewer
   * @param cloud pointcloud to add
   * @param cloud_name name of the point cloud
   */
  void AddColouredPointCloudToViewer(PointCloudXYZRGB::Ptr cloud,
                                     std::string cloud_name);

  /**
   * @brief Add a cloud to viewer
   * @param cloud pointcloud to add
   * @param cloud_name name of the point cloud
   */
  void AddPointCloudToViewer(PointCloudXYZ::Ptr cloud, std::string cloud_name);

  /**
   * @brief Colour a pointcloud
   * @param cloud pointcloud to colour
   * @param r,g,b rgb values of the colour
   * @return coloured pointcloud
   */
  PointCloudXYZRGB::Ptr ColourPointCloud(PointCloudXYZ::Ptr &cloud, int r,
                                         int g, int b);

  PointCloudXYZ::Ptr template_cloud_;
  PointCloudXYZ::Ptr agg_cloud_;
  beam::Affine3 TA_LIDAR_VICON_;
  beam::Affine3 TA_LIDAR_ESTIMATED_;
  double height_{0.5};
  double radius_{0.0635};
  double threshold_{0.015}; // Threshold for cropping the the aggregated cloud

  pcl::visualization::PCLVisualizer pcl_viewer_;
  bool show_transformation_{false};
};

} // end namespace vicon_calibration
