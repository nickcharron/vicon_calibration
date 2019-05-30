#include "vicon_calibration/LidarCylExtractor.h"

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

bool LidarCylExtractor::accept_measurement_;

LidarCylExtractor::LidarCylExtractor(PointCloud::Ptr &template_cloud,
                                     PointCloud::Ptr &scan)
    : template_cloud_(template_cloud), scan_(scan) {}

void LidarCylExtractor::SetScanTransform(Eigen::Affine3d &T_LIDAR_SCAN) {
  if(!beam::IsTransformationMatrix(T_LIDAR_SCAN.matrix())) {
    throw std::runtime_error{
        "Passed in scan transform (scan to lidar) is invalid"};
  }
  T_LIDAR_SCAN_ = T_LIDAR_SCAN;
  pcl::transformPointCloud(*scan_, *scan_, T_LIDAR_SCAN_);
}

void LidarCylExtractor::SetShowTransformation(bool show_transformation) {
  if (show_transformation) {
    pcl_viewer_ = pcl::visualization::PCLVisualizer::Ptr(
        new pcl::visualization::PCLVisualizer("Cloud viewer"));
    pcl_viewer_->setBackgroundColor(0, 0, 0);
    pcl_viewer_->addCoordinateSystem(1.0);
    pcl_viewer_->initCameraParameters();
    pcl_viewer_->registerKeyboardCallback(ConfirmMeasurementKeyboardCallback,
                                          (void*)pcl_viewer_.get());
  }
  show_transformation_ = show_transformation;
}

Eigen::Vector4d
LidarCylExtractor::ExtractCylinder(Eigen::Affine3d &T_SCAN_TARGET_EST,
                                   bool &accept_measurement,
                                   int measurement_num) {
  if (template_cloud_ == nullptr) {
    throw std::runtime_error{"Template cloud is empty"};
  }

  if(!beam::IsTransformationMatrix(T_SCAN_TARGET_EST.matrix())) {
    throw std::runtime_error{"Passed in target to lidar transform is invalid"};
  }

  // Crop the scan before performing ICP registration
  auto cropped_cloud = CropPointCloud(T_SCAN_TARGET_EST);

  // Perform ICP Registration
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cropped_cloud);
  icp.setInputTarget(template_cloud_);
  PointCloud::Ptr final_cloud(new PointCloud);
  icp.align(*final_cloud, T_SCAN_TARGET_EST.inverse().matrix().cast<float>());

  if (!icp.hasConverged()) {
    throw std::runtime_error{
        "Couldn't register cylinder target to template cloud"};
  }

  Eigen::Affine3d T_SCAN_TARGET_OPT;
  T_SCAN_TARGET_OPT.matrix() =
      icp.getFinalTransformation().inverse().cast<double>();

  // Get x,y,r,p data
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

    ShowFinalTransformation();
    pcl_viewer_->resetStoppedFlag();

    accept_measurement = accept_measurement_;
  } else {
    accept_measurement = true;
  }

  return final_transform_vector;
}

PointCloud::Ptr
LidarCylExtractor::CropPointCloud(Eigen::Affine3d &T_SCAN_TARGET_EST) {
  if (scan_ == nullptr) {
    throw std::runtime_error{"Scan is empty"};
  }

  if (radius_ == 0 || height_ == 0) {
    throw std::runtime_error{"Can't crop a cylinder with radius of " +
                             std::to_string(radius_) + " and height of " +
                             std::to_string(height_)};
  }

  if (threshold_ == 0) {
    std::cout << "WARNING: Using threshold of 0 for cropping" << std::endl;
  }

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

  return cropped_cloud;
}

Eigen::Vector4d
LidarCylExtractor::ExtractRelevantMeasurements(Eigen::Affine3d &T_SCAN_TARGET) {

  // Extract x,y,r,p values
  auto translation_vector = T_SCAN_TARGET.translation();
  auto rpy_vector = beam::RToLieAlgebra(T_SCAN_TARGET.rotation());
  Eigen::Vector4d measurement(translation_vector(0), translation_vector(1),
                              rpy_vector(0), rpy_vector(1));

  return measurement;
}

void LidarCylExtractor::AddColouredPointCloudToViewer(
    PointCloudColor::Ptr cloud, std::string cloud_name) {
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
      cloud);
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, cloud_name);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

void LidarCylExtractor::AddPointCloudToViewer(PointCloud::Ptr cloud,
                                              std::string cloud_name) {
  pcl_viewer_->addPointCloud<pcl::PointXYZ>(cloud, cloud_name);
  pcl_viewer_->setPointCloudRenderingProperties(
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

void LidarCylExtractor::ShowFinalTransformation() {
  std::cout << "-------------------------------" << std::endl;
  std::cout << "Legend:" << std::endl;
  std::cout << "  White -> scan" << std::endl;
  std::cout << "  red   -> target initial guess" << std::endl;
  std::cout << "  green -> target aligned" << std::endl;
  std::cout << "Accept measurement? [y/n]" << std::endl;
  while (!pcl_viewer_->wasStopped()) {
    pcl_viewer_->spinOnce(100);
    std::this_thread::sleep_for(100ms);
  }
}

void LidarCylExtractor::ConfirmMeasurementKeyboardCallback(
    const pcl::visualization::KeyboardEvent &event, void *viewer_void) {
  pcl::visualization::PCLVisualizer *viewer =
      static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);

  if (event.getKeySym() == "y" && event.keyDown()) {
    std::cout << "Accepting measurement" << std::endl;
    accept_measurement_ = true;
    viewer->removeAllPointClouds();
    viewer->close();

  } else if (event.getKeySym() == "n" && event.keyDown()) {
    std::cout << "Rejecting measurement" << std::endl;
    accept_measurement_ = false;
    viewer->removeAllPointClouds();
    viewer->close();
  }
}

} // end namespace vicon_calibration
