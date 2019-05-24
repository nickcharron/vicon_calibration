#include "vicon_calibration/LidarCylExtractor.h"

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

LidarCylExtractor::LidarCylExtractor(PointCloud::Ptr &template_cloud,
                                     PointCloud::Ptr &scan)
    : template_cloud_(template_cloud), scan_(scan) {}

void LidarCylExtractor::SetScanTransform(
    Eigen::Affine3d T_LIDAR_SCAN) {
  if (!beam::IsTransformationMatrix(T_LIDAR_SCAN.matrix())) {
    throw std::runtime_error{
        "Passed in scan transform (scan to lidar) is invalid"};
  }
  T_LIDAR_SCAN_ = T_LIDAR_SCAN;
  pcl::transformPointCloud(*scan_, *scan_, T_LIDAR_SCAN_);
}

void LidarCylExtractor::SetShowTransformation(bool show_transformation) {
  if (show_transformation) {
    pcl_viewer_.setBackgroundColor(0, 0, 0);
    pcl_viewer_.addCoordinateSystem(1.0);
    pcl_viewer_.initCameraParameters();
  }
  show_transformation_ = show_transformation;
}

Eigen::Vector4d LidarCylExtractor::ExtractCylinder(Eigen::Affine3d T_SCAN_TARGET_EST,
                                              int measurement_num) {
  if (template_cloud_ == nullptr) {
    throw std::runtime_error{"Template cloud is empty"};
  }

  if (!beam::IsTransformationMatrix(T_SCAN_TARGET_EST.matrix())) {
    throw std::runtime_error{"Passed in target to lidar transform is invalid"};
  }

  Eigen::Affine3d T_TARGET_EST_TARGET_OPT;
  // Crop the scan before performing ICP registration
  auto cropped_cloud = CropPointCloud(T_SCAN_TARGET_EST);

  // Perform ICP Registration
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cropped_cloud);
  icp.setInputTarget(template_cloud_);
  PointCloud::Ptr final_cloud(new PointCloud);
  icp.align(*final_cloud);

  if (!icp.hasConverged()) {
    throw std::runtime_error{
        "Couldn't register cylinder target to template cloud"};
  }

  // Get x,y,r,p data
  T_TARGET_EST_TARGET_OPT.matrix() = icp.getFinalTransformation().cast<double>();

  // Calculate transform from cloud to target
  Eigen::Affine3d T_SCAN_TARGET_OPT = T_SCAN_TARGET_EST * T_TARGET_EST_TARGET_OPT;
  auto final_transform_vector = ExtractRelevantMeasurements(T_SCAN_TARGET_OPT);

  if (show_transformation_) {
    // Display clouds for testing
    // transform template cloud from target to lidar
    auto estimated_template_cloud = ColourPointCloud(template_cloud_, 255, 0, 0);
    pcl::transformPointCloud(*estimated_template_cloud, *estimated_template_cloud,
                             T_SCAN_TARGET_EST);
    AddPointCloudToViewer(estimated_template_cloud,
                          "estimated template cloud " + std::to_string(measurement_num));

    auto measured_template_cloud = ColourPointCloud(template_cloud_, 0, 255, 0);
    pcl::transformPointCloud(*measured_template_cloud, *measured_template_cloud,
                             T_SCAN_TARGET_OPT);
    AddPointCloudToViewer(measured_template_cloud,
                          "measured template cloud " + std::to_string(measurement_num));

    AddPointCloudToViewer(cropped_cloud, "cropped scan " + std::to_string(measurement_num));
  }

  return final_transform_vector;
}

PointCloud::Ptr
LidarCylExtractor::CropPointCloud(Eigen::Affine3d T_SCAN_TARGET_EST) {
  if (scan_ == nullptr) {
    throw std::runtime_error{"Scan is empty"};
  }
  if (threshold_ == 0)
    std::cout << "WARNING: Using threshold of 0 for cropping" << std::endl;

  PointCloud::Ptr cropped_cloud(new PointCloud);
  PointCloud::Ptr transformed_scan(new PointCloud);
  double radius_squared = (radius_ + threshold_) * (radius_ + threshold_);

  // Transform the scan to target frame for cropping
  pcl::transformPointCloud(*scan_, *transformed_scan,
                           T_SCAN_TARGET_EST.inverse());

  // Crop the transformed scan. Reject any points that has z
  // exceeding the height of the cylinder target or the radius bigger than the
  // radius of the cylinder target
  for (PointCloud::iterator it = transformed_scan->begin();
       it != transformed_scan->end(); ++it) {
    if (it->z > height_ + threshold_)
      continue;
    if ((it->x * it->x + it->y * it->y) > radius_squared)
      continue;

    cropped_cloud->push_back(*it);
  }

  return cropped_cloud;
}

Eigen::Vector4d
LidarCylExtractor::ExtractRelevantMeasurements(Eigen::Affine3d T_SCAN_TARGET) {

  // Extract x,y,r,p values
  auto translation_vector = T_SCAN_TARGET.translation();
  auto rpy_vector = beam::RToLieAlgebra(T_SCAN_TARGET.rotation());
  Eigen::Vector4d measurement(translation_vector(0), translation_vector(1),
                         rpy_vector(0), rpy_vector(1));

  return measurement;
}

void LidarCylExtractor::ShowFinalTransformation() {
  while (!pcl_viewer_.wasStopped()) {
    pcl_viewer_.spinOnce(100);
    std::this_thread::sleep_for(100ms);
  }
}

void LidarCylExtractor::AddColouredPointCloudToViewer(
    PointCloudColor::Ptr cloud, std::string cloud_name) {
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
      cloud);
  pcl_viewer_.addPointCloud<pcl::PointXYZRGB>(cloud, rgb, cloud_name);
  pcl_viewer_.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name);
}

void LidarCylExtractor::AddPointCloudToViewer(PointCloud::Ptr cloud,
                                              std::string cloud_name) {
  pcl_viewer_.addPointCloud<pcl::PointXYZ>(cloud, cloud_name);
  pcl_viewer_.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

PointCloudColor::Ptr
LidarCylExtractor::ColourPointCloud(PointCloud::Ptr &cloud, int r, int g,
                                    int b) {
  PointCloudColor::Ptr coloured_cloud(new PointCloudColor);
  uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                  static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
  pcl::PointXYZRGB point;
  for (PointCloud::iterator it = cloud->begin(); it != cloud->end(); ++it) {
    point.x = it->x;
    point.y = it->y;
    point.z = it->z;
    point.rgb = *reinterpret_cast<float *>(&rgb);
    coloured_cloud->push_back(point);
  }
  return coloured_cloud;
}

} // end namespace vicon_calibration
