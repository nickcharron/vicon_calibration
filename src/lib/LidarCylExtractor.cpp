#include "vicon_calibration/LidarCylExtractor.h"

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

LidarCylExtractor::LidarCylExtractor(PointCloud::Ptr &template_cloud,
                                     PointCloud::Ptr &scan)
    : template_cloud_(template_cloud), scan_(scan) {}

void LidarCylExtractor::SetScanTransform(Eigen::Affine3d T_LIDAR_SCAN) {
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

Eigen::Vector4d
LidarCylExtractor::ExtractCylinder(Eigen::Affine3d T_SCAN_TARGET_EST,
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
  T_TARGET_EST_TARGET_OPT.matrix() =
      icp.getFinalTransformation().cast<double>();

  // Calculate transform from cloud to target
  Eigen::Affine3d T_SCAN_TARGET_OPT =
      T_SCAN_TARGET_EST * T_TARGET_EST_TARGET_OPT;
  auto final_transform_vector = ExtractRelevantMeasurements(T_SCAN_TARGET_OPT);

  if (show_transformation_) {
    // Display clouds for testing
    // transform template cloud from target to lidar
    auto estimated_template_cloud =
        ColourPointCloud(template_cloud_, 255, 0, 0);
    pcl::transformPointCloud(*estimated_template_cloud,
                             *estimated_template_cloud, T_SCAN_TARGET_EST);
    AddColouredPointCloudToViewer(estimated_template_cloud,
                                  "estimated template cloud " +
                                      std::to_string(measurement_num));

    auto measured_template_cloud = ColourPointCloud(template_cloud_, 0, 255, 0);
    pcl::transformPointCloud(*measured_template_cloud, *measured_template_cloud,
                             T_SCAN_TARGET_OPT);
    AddColouredPointCloudToViewer(measured_template_cloud,
                                  "measured template cloud " +
                                      std::to_string(measurement_num));

    AddPointCloudToViewer(cropped_cloud,
                          "cropped scan " + std::to_string(measurement_num));
  }

  return final_transform_vector;
}

void LidarCylExtractor::ShowCroppedCloud(Eigen::Affine3d T_SCAN_TARGET_EST) {
  /*PointCloud::Ptr input (new PointCloud);
  PointCloud::Ptr input2 (new PointCloud);

  // Test the PointCloud<PointT> method
  pcl::CropBox<pcl::PointXYZ> cropBoxFilter;
  input->push_back (pcl::PointXYZ (1.0, 0.0, 0.0));
  input->push_back (pcl::PointXYZ (1.9, 0.9, 0.9));
  input->push_back (pcl::PointXYZ (1.9, 0.9, -0.9));
  input->push_back (pcl::PointXYZ (1.9, -0.9, 0.9));
  input->push_back (pcl::PointXYZ (0.1, 0.9, 0.9));
  input->push_back (pcl::PointXYZ (1.9, -0.9, -0.9));
  input->push_back (pcl::PointXYZ (0.1, -0.9, 0.9));
  input->push_back (pcl::PointXYZ (0.1, 0.9, -0.9));
  input->push_back (pcl::PointXYZ (0.1, -0.9, -0.9));

  cropBoxFilter.setInputCloud (input);

  Eigen::Vector4f min_pt (-1.0, -1.0, -1.0, 0);
  Eigen::Vector4f max_pt (1.0, 1.0, 1.0, 0);
  // Cropbox slighlty bigger then bounding box of points
  cropBoxFilter.setMin (min_pt);
  cropBoxFilter.setMax (max_pt);

  cropBoxFilter.setTranslation(Eigen::Vector3f(1, 0, 0));

  PointCloud::Ptr cloud_out1(new PointCloud);

  cropBoxFilter.filter (*cloud_out1);
  AddPointCloudToViewer(cloud_out1, "test cloud 1");
  std::cout << cloud_out1->size() << std::endl;

  for (PointCloud::iterator it = cloud_out1->begin();
       it != cloud_out1->end(); ++it) {
    std::cout << "x: " << it->x << " y: " << it->y << " z: " << it->z <<
  std::endl;
  }

  input2->push_back (pcl::PointXYZ (0.0, 0.0, 0.0));
  input2->push_back (pcl::PointXYZ (0.9, 0.9, 0.9));
  input2->push_back (pcl::PointXYZ (0.9, 0.9, -0.9));
  input2->push_back (pcl::PointXYZ (0.9, -0.9, 0.9));
  input2->push_back (pcl::PointXYZ (-0.9, 0.9, 0.9));
  input2->push_back (pcl::PointXYZ (0.9, -0.9, -0.9));
  input2->push_back (pcl::PointXYZ (-0.9, -0.9, 0.9));
  input2->push_back (pcl::PointXYZ (-0.9, 0.9, -0.9));
  input2->push_back (pcl::PointXYZ (-0.9, -0.9, -0.9));

  cropBoxFilter.setInputCloud(input2);

  //cropBoxFilter.setTranslation(Eigen::Vector3f(0, 0, 0));

  PointCloud::Ptr cloud_out2(new PointCloud);
  std::vector<int> indices2;

  cropBoxFilter.filter (*cloud_out2);
  auto coloured = ColourPointCloud(cloud_out2, 255, 0, 0);
  AddColouredPointCloudToViewer(coloured, "test cloud 2");
  std::cout << cloud_out2->size() << std::endl;*/
  CropPointCloud(T_SCAN_TARGET_EST);
}

PointCloud::Ptr
LidarCylExtractor::CropPointCloud(Eigen::Affine3d T_SCAN_TARGET_EST) {
  if (scan_ == nullptr) {
    throw std::runtime_error{"Scan is empty"};
  }
  if (threshold_ == 0)
    std::cout << "WARNING: Using threshold of 0 for cropping" << std::endl;

  Eigen::Vector4f min_vector(-radius_ - threshold_, -radius_ - threshold_,
                             -threshold_, 0);
  Eigen::Vector4f max_vector(radius_ + threshold_, radius_ + threshold_,
                             height_ + threshold_, 0);

  Eigen::Vector3d translation = T_SCAN_TARGET_EST.translation();
  Eigen::Vector3d rotation = T_SCAN_TARGET_EST.rotation().eulerAngles(0, 1, 2);

  pcl::CropBox<pcl::PointXYZ> cropper;

  cropper.setMin(min_vector);
  cropper.setMax(max_vector);

  cropper.setTranslation(translation.cast<float>());
  cropper.setRotation(rotation.cast<float>());

  cropper.setInputCloud(scan_);

  PointCloud::Ptr cropped_cloud(new PointCloud);
  cropper.filter(*cropped_cloud);
  std::cout << "size " << cropped_cloud->size() << std::endl;

  AddPointCloudToViewer(cropped_cloud, "cropped cloud");
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
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

void LidarCylExtractor::AddPointCloudToViewer(PointCloud::Ptr cloud,
                                              std::string cloud_name) {
  pcl_viewer_.addPointCloud<pcl::PointXYZ>(cloud, cloud_name);
  pcl_viewer_.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

PointCloudColor::Ptr LidarCylExtractor::ColourPointCloud(PointCloud::Ptr &cloud,
                                                         int r, int g, int b) {
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
