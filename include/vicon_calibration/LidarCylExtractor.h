#pragma once

#include "vicon_calibration/params.h"
#include "vicon_calibration/utils.h"
#include <beam_filtering/CropBox.h>

#include <Eigen/Geometry>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <thread>
#include <vector>

namespace vicon_calibration {

/**
 * @brief class for extracting cylinder measurements from lidar scan
 */
class LidarCylExtractor {
public:
  
  /**
   * @brief constructor
   */
  LidarCylExtractor();

  /**
   * @brief default constructor
   */
  ~LidarCylExtractor() = default;

  /**
   * @brief Returns the full path to a file named file_name saved in config
   * subfolder
   * @param file_name
   * @return file_location
   */
  std::string GetJSONFileNameConfig(std::string file_name);

  /**
   * @brief sets the target parameters. For now, only cylinder targets have
   * been implemented
   * @param target_params
   */
  void SetTargetParams(vicon_calibration::CylinderTgtParams &target_params);

  /**
   * @brief sets the registration parameters. For now, only icp based
   * registration has been implemented
   * @param registration_params
   */
  void SetRegistrationParams(
      vicon_calibration::RegistrationParams &registration_params);

  /**
   * @brief sets the registration parameters. For now, only icp based
   * registration has been implemented
   * @param registration_params
   */
  void SetScan(PointCloud::Ptr &scan) { scan_ = scan; }

  /**
   * @brief Get scan
   * @return boost::shared_ptr to scan
   */
  PointCloud::Ptr GetScan() { return scan_; }

  /**
   * @brief If the scan is not in the lidar frame, setting this will transform
   * the scan into the lidar frame. E.g., maybe your lidar is rotating and you
   * want to register an aggregate scan
   * @param T_LIDAR_SCAN transforms points from scan frame to lidar frame
   */
  void SetScanTransform(Eigen::Affine3d &T_LIDAR_SCAN);

  /**
   * @brief Get template cloud
   * @return boost::shared_ptr to template cloud
   */
  PointCloud::Ptr GetTemplateCloud() { return template_cloud_; }

  /**
   * @brief Return measurement_ and measurement_valid_ if extracting measurement
   * has been completed
   * @return a pair of 4x1 vector of x,y translation and rotation about x and y
   * axes [tx, ty, ra, ry]^T and bool indicating if the measurement is valid
   */
  std::pair<Eigen::Affine3d, bool> GetMeasurementInfo();

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

private:
  /**
   * @brief set the template cloud from the file name specified in the target
   * parameters
   */
  void SetTemplateCloud();

  /**
   * @brief set the parameters used to crop each scan
   */
  void SetCropboxParams();

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
  void SetICPConfig();

  /**
   * @brief Keyboard event callback to allow the user to accept or reject final
   * transform measurement
   */
  static void ConfirmMeasurementKeyboardCallback(
      const pcl::visualization::KeyboardEvent &event, void *viewer_void);

  // params
  vicon_calibration::RegistrationParams registration_params_;
  vicon_calibration::CylinderTgtParams target_params_;

  // Objects for extracting cylinder
  PointCloud::Ptr template_cloud_;
  PointCloud::Ptr scan_;
  Eigen::Affine3d T_LIDAR_SCAN_;
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;
  pcl::visualization::PCLVisualizer::Ptr pcl_viewer_;
  beam_filtering::CropBox cropper_;

  // Measurement info
  static bool measurement_valid_;  // For displaying resulted clouds
  static bool measurement_failed_; // For displaying clouds when icp diverges
  bool measurement_complete_{false};
  Eigen::Affine3d measurement_;
  Eigen::Affine3d INVALID_MEASUREMENT;
};

} // end namespace vicon_calibration
