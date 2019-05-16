#include "vicon_calibration/LidarCylExtractor.h"

#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>

#include <thread>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

LidarCylExtractor::LidarCylExtractor(PointCloudXYZ::Ptr &template_cloud,
                                     PointCloudXYZ::Ptr &agg_cloud)
    : template_cloud_(template_cloud), agg_cloud_(agg_cloud) {}

void LidarCylExtractor::SetAggregatedCloudTransform(
    beam::Affine3 TA_LIDAR_VICON) {
  TA_LIDAR_VICON_ = TA_LIDAR_VICON;
  pcl::transformPointCloud(*agg_cloud_, *agg_cloud_, TA_LIDAR_VICON_);
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
  beam::Affine3 TA_TARGET_ESTIMATED;
  // Crop the aggregated cloud before performing ICP registration
  auto cropped_cloud = CropPointCloud(TA_LIDAR_TARGET);

  // Perform ICP Registration
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cropped_cloud);
  icp.setInputTarget(template_cloud_);
  PointCloudXYZ final_cloud;
  icp.align(final_cloud);

  if (!icp.hasConverged()) {
    PCL_ERROR("Couldn't extract the cylinder target.");
  }

  // Get x,y,r,p data
  TA_TARGET_ESTIMATED.matrix() = icp.getFinalTransformation().cast<double>();
  auto final_transform_vector =
      CalculateMeasurement(TA_LIDAR_TARGET, TA_TARGET_ESTIMATED);

  if (show_transformation_) {
    // Print out transforms and display clouds for testing
    std::cout << "ICP REGISTERED TRANSFORM: TARGET " << measurement_num
              << " ESTIMATED TO TARGET" << std::endl
              << TA_TARGET_ESTIMATED.matrix() << std::endl;

    // transform template cloud from target to lidar
    PointCloudXYZ::Ptr transformed_template_cloud(new PointCloudXYZ);
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

    TA_LIDAR_ESTIMATED_.matrix() = transformation_matrix;
    std::cout << "CHECK: TARGET " << measurement_num
              << " ESTIMATED TO LIDAR" << std::endl
              << TA_LIDAR_ESTIMATED_.matrix() << std::endl;

    // Colour cropped cloud and transform it from target estimated to lidar
    auto coloured_cropped_cloud = ColourPointCloud(cropped_cloud, 255, 0, 0);
    pcl::transformPointCloud(*coloured_cropped_cloud, *coloured_cropped_cloud,
                             TA_LIDAR_ESTIMATED_.inverse());
    AddColouredPointCloudToViewer(coloured_cropped_cloud,
                                  "coloured cropped cloud " +
                                      std::to_string(measurement_num));
    beam::Affine3 TA_VICON_ESTIMATED =
        TA_LIDAR_VICON_.inverse() * TA_LIDAR_ESTIMATED_;
    std::cout << "CHECK: TARGET " << measurement_num << " ESTIMATED TO VICON"
              << std::endl << TA_VICON_ESTIMATED.matrix() << std::endl;
  }

  return final_transform_vector;
}

PointCloudXYZ::Ptr
LidarCylExtractor::CropPointCloud(beam::Affine3 TA_LIDAR_TARGET) {
  if (threshold_ == 0)
    std::cout << "WARNING: Using threshold of 0 for cropping" << std::endl;

  PointCloudXYZ::Ptr cropped_cloud(new PointCloudXYZ);
  PointCloudXYZ::Ptr transformed_agg_cloud(new PointCloudXYZ);
  double radius__squared = (radius_ + threshold_) * (radius_ + threshold_);

  // Transform the aggregated cloud to target frame for cropping
  pcl::transformPointCloud(*agg_cloud_, *transformed_agg_cloud,
                           TA_LIDAR_TARGET.inverse());
  // Crop the transformed aggregated cloud. Reject any points that has z
  // exceeding the height of the cylinder target or the radius bigger than the
  // radius of the cylinder target
  for (PointCloudXYZ::iterator it = transformed_agg_cloud->begin();
       it != transformed_agg_cloud->end(); ++it) {
    if (it->z > height_ + threshold_)
      continue;
    if ((it->x * it->x + it->y * it->y) > radius__squared)
      continue;

    cropped_cloud->push_back(*it);
  }

  return cropped_cloud;
}

beam::Vec4
LidarCylExtractor::CalculateMeasurement(beam::Affine3 TA_LIDAR_TARGET,
                                        beam::Affine3 TA_TARGET_ESTIMATED) {
  // Calculate transform from cloud to target
  beam::Affine3 TA_LIDAR_ESTIMATED = TA_LIDAR_TARGET * TA_TARGET_ESTIMATED;
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
    PointCloudXYZRGB::Ptr cloud, std::string cloud_name) {
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
      cloud);
  pcl_viewer_.addPointCloud<pcl::PointXYZRGB>(cloud, rgb, cloud_name);
  pcl_viewer_.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name);
}

void LidarCylExtractor::AddPointCloudToViewer(PointCloudXYZ::Ptr cloud,
                                              std::string cloud_name) {
  pcl_viewer_.addPointCloud<pcl::PointXYZ>(cloud, cloud_name);
  pcl_viewer_.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

PointCloudXYZRGB::Ptr
LidarCylExtractor::ColourPointCloud(PointCloudXYZ::Ptr &cloud, int r, int g,
                                    int b) {
  PointCloudXYZRGB::Ptr coloured_cloud(new PointCloudXYZRGB);
  uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                  static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
  pcl::PointXYZRGB point;
  for (PointCloudXYZ::iterator it = cloud->begin(); it != cloud->end(); ++it) {
    point.x = it->x;
    point.y = it->y;
    point.z = it->z;
    point.rgb = *reinterpret_cast<float *>(&rgb);
    coloured_cloud->push_back(point);
  }
  return coloured_cloud;
}

} // end namespace vicon_calibration
