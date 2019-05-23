#include "vicon_calibration/LidarCylExtractor.h"

#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>

#include <thread>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

LidarCylExtractor::LidarCylExtractor(PointCloud::Ptr &template_cloud,
                                     PointCloud::Ptr &scan)
    : template_cloud_(template_cloud), scan_(scan) {}

void LidarCylExtractor::SetScanTransform(
    beam::Affine3 TA_LIDAR_VICON) {
  if (!beam::IsTransformationMatrix(TA_LIDAR_VICON.matrix())) {
    throw std::runtime_error{
        "Passed in aggregated cloud transform (vicon to lidar) is invalid"};
  }
  TA_LIDAR_VICON_ = TA_LIDAR_VICON;
  pcl::transformPointCloud(*scan_, *scan_, TA_LIDAR_VICON_);
}

void LidarCylExtractor::SetShowTransformation(bool show_transformation) {
  if (show_transformation) {
    pcl_viewer_.setBackgroundColor(0, 0, 0);
    pcl_viewer_.addCoordinateSystem(1.0);
    pcl_viewer_.initCameraParameters();
  }
  show_transformation_ = show_transformation;
}

beam::Vec4 LidarCylExtractor::ExtractCylinder(beam::Affine3 TA_LIDAR_TARGET,
                                              int measurement_num) {
  if (template_cloud_ == nullptr) {
    throw std::runtime_error{"Template cloud is empty"};
  }

  if (!beam::IsTransformationMatrix(TA_LIDAR_TARGET.matrix())) {
    throw std::runtime_error{"Passed in target to lidar transform is invalid"};
  }

  beam::Affine3 TA_TARGET_ESTIMATED;
  // Crop the aggregated cloud before performing ICP registration
  auto cropped_cloud = CropPointCloud(TA_LIDAR_TARGET);

  // Perform ICP Registration
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cropped_cloud);
  icp.setInputTarget(template_cloud_);
  PointCloud final_cloud;
  icp.align(final_cloud);

  if (!icp.hasConverged()) {
    throw std::runtime_error{
        "Couldn't register cylinder target to template cloud"};
  }

  // Get x,y,r,p data
  TA_TARGET_ESTIMATED.matrix() = icp.getFinalTransformation().cast<double>();

  // Calculate transform from cloud to target
  beam::Affine3 TA_LIDAR_ESTIMATED = TA_LIDAR_TARGET * TA_TARGET_ESTIMATED;
  auto final_transform_vector = CalculateMeasurement(TA_LIDAR_ESTIMATED);

  if (show_transformation_) {
    // Display clouds for testing
    // transform template cloud from target to lidar
    PointCloud::Ptr transformed_template_cloud(new PointCloud);
    pcl::transformPointCloud(*template_cloud_, *transformed_template_cloud,
                             TA_LIDAR_TARGET.inverse());
    AddPointCloudToViewer(transformed_template_cloud,
                          "template cloud " + std::to_string(measurement_num));

    // convert the final transform back to affine
    beam::Vec3 translation_vector(final_transform_vector(0),
                                  final_transform_vector(1), 0);
    beam::Vec3 rpy_vector(final_transform_vector(2), final_transform_vector(3),
                          0);

    auto rotation_matrix = beam::LieAlgebraToR(rpy_vector);

    beam::Mat4 transformation_matrix;
    transformation_matrix.setIdentity();
    transformation_matrix.block<3, 3>(0, 0) = rotation_matrix;
    transformation_matrix.block<3, 1>(0, 3) = translation_vector;

    // Colour cropped cloud and transform it from target estimated to lidar
    auto coloured_cropped_cloud = ColourPointCloud(cropped_cloud, 255, 0, 0);
    pcl::transformPointCloud(*coloured_cropped_cloud, *coloured_cropped_cloud,
                             TA_LIDAR_ESTIMATED_.inverse());
    AddColouredPointCloudToViewer(coloured_cropped_cloud,
                                  "coloured cropped cloud " +
                                  std::to_string(measurement_num));
  }

  return final_transform_vector;
}

PointCloud::Ptr
LidarCylExtractor::CropPointCloud(beam::Affine3 TA_LIDAR_TARGET) {
  if (scan_ == nullptr) {
    throw std::runtime_error{"Aggregated cloud is empty"};
  }
  if (threshold_ == 0)
    std::cout << "WARNING: Using threshold of 0 for cropping" << std::endl;

  PointCloud::Ptr cropped_cloud(new PointCloud);
  PointCloud::Ptr transformed_scan(new PointCloud);
  double radius_squared = (radius_ + threshold_) * (radius_ + threshold_);

  // Transform the aggregated cloud to target frame for cropping
  pcl::transformPointCloud(*scan_, *transformed_scan,
                           TA_LIDAR_TARGET.inverse());
  // Crop the transformed aggregated cloud. Reject any points that has z
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

beam::Vec4
LidarCylExtractor::CalculateMeasurement(beam::Affine3 TA_LIDAR_ESTIMATED) {

  // Extract x,y,r,p values
  auto translation_vector = TA_LIDAR_ESTIMATED.translation();
  auto rpy_vector = beam::RToLieAlgebra(TA_LIDAR_ESTIMATED.rotation());
  beam::Vec4 measurement(translation_vector(0), translation_vector(1),
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
