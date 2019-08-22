#include "vicon_calibration/LidarCylExtractor.h"
#include "vicon_calibration/utils.h"
#include <beam_utils/math.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

namespace vicon_calibration {

using namespace std::literals::chrono_literals;

bool LidarCylExtractor::measurement_valid_;
bool LidarCylExtractor::measurement_failed_;

LidarCylExtractor::LidarCylExtractor() {
  INVALID_MEASUREMENT.matrix().setIdentity();
  INVALID_MEASUREMENT.matrix()(0, 3) = -100;
  INVALID_MEASUREMENT.matrix()(1, 3) = -100;
  INVALID_MEASUREMENT.matrix()(2, 3) = -100;
  template_cloud_ = boost::make_shared<PointCloud>();
  scan_ = boost::make_shared<PointCloud>();
  pcl_viewer_ = boost::make_shared<pcl::visualization::PCLVisualizer>();
}

std::string LidarCylExtractor::GetJSONFileNameConfig(std::string file_name) {
  std::string file_location = __FILE__;
  file_location.erase(file_location.end() - 27, file_location.end());
  file_location += "config/";
  file_location += file_name;
  return file_location;
}

void LidarCylExtractor::SetTargetParams(
    vicon_calibration::CylinderTgtParams &target_params) {
  target_params_ = target_params;
  SetTemplateCloud();
  SetCropboxParams();
}

void LidarCylExtractor::SetCropboxParams() {
  if (target_params_.radius == 0 || target_params_.height == 0) {
    throw std::runtime_error{"Can't crop scan, invalid target params."};
  }

  if (target_params_.crop_threshold == 0) {
    std::cout << "WARNING: Using threshold of 0 for cropping" << std::endl;
  }

  Eigen::Vector3f min_vector(
      -target_params_.crop_threshold,
      -target_params_.radius - target_params_.crop_threshold,
      -target_params_.radius - target_params_.crop_threshold);
  Eigen::Vector3f max_vector(
      target_params_.height + target_params_.crop_threshold,
      target_params_.radius + target_params_.crop_threshold,
      target_params_.radius + target_params_.crop_threshold);

  cropper_.SetMinVector(min_vector);
  cropper_.SetMaxVector(max_vector);
}

void LidarCylExtractor::SetRegistrationParams(
    vicon_calibration::RegistrationParams &registration_params) {
  registration_params_ = registration_params;
  SetICPConfig();
}

void LidarCylExtractor::SetTemplateCloud() {
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(target_params_.template_cloud,
                                          *template_cloud_) == -1) {
    LOG_ERROR("Couldn't read template file: %s\n",
              target_params_.template_cloud.c_str());
  }
}

void LidarCylExtractor::SetICPConfig() {
  icp_.setTransformationEpsilon(registration_params_.transform_epsilon);
  icp_.setEuclideanFitnessEpsilon(registration_params_.euclidean_epsilon);
  icp_.setMaximumIterations(registration_params_.max_iterations);
  icp_.setMaxCorrespondenceDistance(
      registration_params_.max_correspondance_distance);
}

void LidarCylExtractor::SetScanTransform(Eigen::Affine3d &T_LIDAR_SCAN) {
  if (!utils::IsTransformationMatrix(T_LIDAR_SCAN.matrix())) {
    throw std::runtime_error{
        "Passed in scan transform (scan to lidar) is invalid"};
  }
  T_LIDAR_SCAN_ = T_LIDAR_SCAN;
  pcl::transformPointCloud(*scan_, *scan_, T_LIDAR_SCAN_);
}

std::pair<Eigen::Affine3d, bool> LidarCylExtractor::GetMeasurementInfo() {
  if (measurement_complete_) {
    measurement_complete_ = false;
    return std::make_pair(measurement_, measurement_valid_);
  } else {
    throw std::runtime_error{"Measurement not complete. Please run/rerun "
                             "LidarCylExtractor::ExtractCylinder() before "
                             "getting the measurement information."};
  }
}

PointCloud::Ptr
LidarCylExtractor::CropPointCloud(Eigen::Affine3d &T_SCAN_TARGET_EST) {
  if (scan_ == nullptr || scan_->size() == 0) {
    throw std::runtime_error{"Scan is empty"};
  }
  PointCloud::Ptr cropped_cloud(new PointCloud);
  Eigen::Affine3f T_TARGET_EST_SCAN = T_SCAN_TARGET_EST.inverse().cast<float>();
  cropper_.SetTransform(T_TARGET_EST_SCAN);
  cropper_.Filter(*scan_, *cropped_cloud);
  return cropped_cloud;
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
  std::cout << "  Blue  -> target initial guess" << std::endl;
  std::cout << "  Green -> target aligned" << std::endl;
  std::cout << "Accept measurement? [y/n]" << std::endl;
  while (!pcl_viewer_->wasStopped()) {
    pcl_viewer_->spinOnce(10);
    std::this_thread::sleep_for(10ms);
  }
}

void LidarCylExtractor::ShowFailedMeasurement() {
  std::cout << "-------------------------------" << std::endl;
  std::cout << "Legend:" << std::endl;
  std::cout << "  Red   -> cropped scan" << std::endl;
  std::cout << "  White -> original scan" << std::endl;
  std::cout << "Press [c] to continue with other measurements" << std::endl;
  while (!pcl_viewer_->wasStopped()) {
    pcl_viewer_->spinOnce(10);
    std::this_thread::sleep_for(10ms);
  }
}

void LidarCylExtractor::ConfirmMeasurementKeyboardCallback(
    const pcl::visualization::KeyboardEvent &event, void *viewer_void) {
  pcl::visualization::PCLVisualizer *viewer =
      static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);

  if (measurement_failed_) {
    if (event.getKeySym() == "c" && event.keyDown()) {
      std::cout << "Continuing with taking measurements" << std::endl;
      measurement_failed_ = false;
      viewer->removeAllPointClouds();
      viewer->close();
    }
  } else {
    if (event.getKeySym() == "y" && event.keyDown()) {
      std::cout << "Accepting measurement" << std::endl;
      measurement_valid_ = true;
      viewer->removeAllPointClouds();
      viewer->close();

    } else if (event.getKeySym() == "n" && event.keyDown()) {
      std::cout << "Rejecting measurement" << std::endl;
      measurement_valid_ = false;
      viewer->removeAllPointClouds();
      viewer->close();
    }
  }
}

void LidarCylExtractor::ExtractCylinder(Eigen::Affine3d &T_SCAN_TARGET_EST,
                                        int measurement_num) {

  if (template_cloud_ == nullptr || template_cloud_->size() == 0) {
    throw std::runtime_error{"Template cloud is empty"};
  }

  if (!utils::IsTransformationMatrix(T_SCAN_TARGET_EST.matrix())) {
    throw std::runtime_error{
        "Estimated transform from target to lidar is invalid"};
  }

  // Crop the scan before performing ICP registration
  auto cropped_cloud = CropPointCloud(T_SCAN_TARGET_EST);

  // Perform ICP Registration
  PointCloud::Ptr final_cloud(new PointCloud);
  icp_.setInputSource(cropped_cloud);
  icp_.setInputTarget(template_cloud_);
  icp_.align(*final_cloud, T_SCAN_TARGET_EST.inverse().matrix().cast<float>());

  if (!icp_.hasConverged()) {
    if (registration_params_.show_transform) {
      measurement_failed_ = true;
      std::cout << "ICP failed. Displaying cropped scan." << std::endl;
      auto coloured_cropped_cloud = ColourPointCloud(cropped_cloud, 255, 0, 0);
      AddColouredPointCloudToViewer(coloured_cropped_cloud,
                                    "coloured cropped cloud " +
                                        std::to_string(measurement_num));
      AddPointCloudToViewer(scan_, "scan " + std::to_string(measurement_num));
      ShowFailedMeasurement();
      pcl_viewer_->resetStoppedFlag();
    }
    measurement_valid_ = false;
    measurement_ = INVALID_MEASUREMENT;
    measurement_complete_ = true;
    return;
  }

  Eigen::Affine3d T_SCAN_TARGET_OPT;
  T_SCAN_TARGET_OPT.matrix() =
      icp_.getFinalTransformation().inverse().cast<double>();
  measurement_ = T_SCAN_TARGET_OPT;

  if (registration_params_.show_transform) {
    // Display clouds for testing
    // transform template cloud from target to lidar
    auto estimated_template_cloud =
        ColourPointCloud(template_cloud_, 0, 0, 255);
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
  }

  Eigen::Vector2d dist_diff(
      measurement_.matrix()(0, 3) - T_SCAN_TARGET_EST.matrix()(0, 3),
      measurement_.matrix()(1, 3) - T_SCAN_TARGET_EST.matrix()(1, 3));
  Eigen::Vector3d rpy_measured, rpy_estimated;
  rpy_measured = measurement_.rotation().eulerAngles(0, 1, 2);
  rpy_estimated = T_SCAN_TARGET_EST.rotation().eulerAngles(0, 1, 2);
  Eigen::Vector2d rot_diff(rpy_measured(0) - rpy_estimated(0),
                           rpy_measured(1) - rpy_estimated(1));

  double dist_err = std::round(dist_diff.norm() * 10000) / 10000;
  double rot_err = std::round(rot_diff.norm() * 10000) / 10000;
  if (dist_err >= registration_params_.dist_acceptance_criteria ||
      rot_err >= registration_params_.rot_acceptance_criteria) {
    measurement_valid_ = false;
  } else {
    measurement_valid_ = true;
  }
  measurement_complete_ = true;
}

} // end namespace vicon_calibration
