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

  Eigen::Vector3f min_vector(
      -registration_params_.crop_threshold_x,
      -target_params_.radius - registration_params_.crop_threshold_y,
      -target_params_.radius - registration_params_.crop_threshold_z);
  Eigen::Vector3f max_vector(
      target_params_.height + registration_params_.crop_threshold_x,
      target_params_.radius + registration_params_.crop_threshold_y,
      target_params_.radius + registration_params_.crop_threshold_z);

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
      registration_params_.max_correspondence_distance);
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
    PointCloudColor::Ptr cloud, std::string cloud_name, Eigen::Affine3d &T) {
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
      cloud);
  pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, cloud_name);
  pcl_viewer_->addCoordinateSystem(1, T.cast<float>(), cloud_name + "frame");
  pcl::PointXYZ point;
  point.x = T.translation()(0);
  point.y = T.translation()(1);
  point.z = T.translation()(2);
  pcl_viewer_->addText3D(cloud_name + " ", point, 0.05, 0.05, 0.05);
  pcl_viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_name);
}

void LidarCylExtractor::AddPointCloudToViewer(PointCloud::Ptr cloud,
                                              std::string cloud_name,
                                              Eigen::Affine3d &T) {
  pcl_viewer_->addPointCloud<pcl::PointXYZ>(cloud, cloud_name);
  pcl_viewer_->addCoordinateSystem(1, T.cast<float>(), cloud_name + "frame");
  pcl::PointXYZ point;
  point.x = T.translation()(0);
  point.y = T.translation()(1);
  point.z = T.translation()(2);
  pcl_viewer_->addText3D(cloud_name + " ", point, 0.05, 0.05, 0.05);
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
  std::cout << "\nViewer Legend:\n"
            << "  White -> scan\n"
            << "  Blue  -> target initial guess\n"
            << "  Green -> target aligned\n"
            << "Accept measurement? [y/n]\n";
  while (!pcl_viewer_->wasStopped() && !close_viewer_) {
    pcl_viewer_->spinOnce(10);
    pcl_viewer_->registerKeyboardCallback(
        &LidarCylExtractor::ConfirmMeasurementKeyboardCallback, *this);
    std::this_thread::sleep_for(10ms);
  }
  close_viewer_ = false;
  pcl_viewer_->removeAllPointClouds();
  pcl_viewer_->removeAllCoordinateSystems();
  pcl_viewer_->removeAllShapes();
  pcl_viewer_->close();
  pcl_viewer_->resetStoppedFlag();
  if (measurement_failed_) {
    std::cout << "Continuing with taking measurements" << std::endl;
  } else if (measurement_valid_) {
    std::cout << "Accepting measurement" << std::endl;
  } else {
    std::cout << "Rejecting measurement" << std::endl;
  }
}

void LidarCylExtractor::ShowFailedMeasurement() {
  std::cout << "\nViewer Legend:\n"
            << "  Red   -> cropped scan\n"
            << "  White -> original scan\n"
            << "Press [c] to continue with other measurements\n";
  while (!pcl_viewer_->wasStopped() && !close_viewer_) {
    pcl_viewer_->spinOnce(10);
    pcl_viewer_->registerKeyboardCallback(
        &LidarCylExtractor::ConfirmMeasurementKeyboardCallback, *this);
    std::this_thread::sleep_for(10ms);
  }
  close_viewer_ = false;
  pcl_viewer_->removeAllPointClouds();
  pcl_viewer_->close();
  pcl_viewer_->resetStoppedFlag();
  if (measurement_failed_) {
    std::cout << "Continuing with taking measurements" << std::endl;
  } else if (measurement_valid_) {
    std::cout << "Accepting measurement" << std::endl;
  } else {
    std::cout << "Rejecting measurement" << std::endl;
  }
}

void LidarCylExtractor::ConfirmMeasurementKeyboardCallback(
    const pcl::visualization::KeyboardEvent &event, void *viewer_void) {

  if (measurement_failed_) {
    if (event.getKeySym() == "c" && event.keyDown()) {
      measurement_failed_ = false;
      close_viewer_ = true;
    }
  } else {
    if (event.getKeySym() == "y" && event.keyDown()) {
      measurement_valid_ = true;
      close_viewer_ = true;
    } else if (event.getKeySym() == "n" && event.keyDown()) {
      measurement_valid_ = false;
      close_viewer_ = true;
    }
  }
}

void LidarCylExtractor::ExtractCylinder(Eigen::Affine3d &T_SCAN_TARGET_EST,
                                        int measurement_num) {
  Eigen::Affine3d T_identity;
  T_identity.setIdentity();

  if (template_cloud_ == nullptr || template_cloud_->size() == 0) {
    throw std::runtime_error{"Template cloud is empty"};
  }

  if (!utils::IsTransformationMatrix(T_SCAN_TARGET_EST.matrix())) {
    throw std::runtime_error{
        "Estimated transform from target to lidar is invalid"};
  }

  // Crop the scan before performing ICP registration
  auto scan_cropped = CropPointCloud(T_SCAN_TARGET_EST);

  // Perform ICP Registration
  PointCloud::Ptr scan_registered(new PointCloud);
  icp_.setInputSource(scan_cropped);
  icp_.setInputTarget(template_cloud_);
  icp_.align(*scan_registered,
             T_SCAN_TARGET_EST.inverse().matrix().cast<float>());

  if (!icp_.hasConverged()) {
    if (registration_params_.show_transform) {
      measurement_failed_ = true;
      std::cout << "ICP failed. Displaying cropped scan." << std::endl;
      auto scan_cropped_coloured = ColourPointCloud(scan_cropped, 255, 0, 0);
      AddColouredPointCloudToViewer(scan_cropped_coloured,
                                    "coloured cropped cloud " +
                                        std::to_string(measurement_num),
                                    T_identity);
      AddPointCloudToViewer(scan_, "scan " + std::to_string(measurement_num),
                            T_identity);
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

  Eigen::Vector2d dist_diff(
      measurement_.matrix()(0, 3) - T_SCAN_TARGET_EST.matrix()(0, 3),
      measurement_.matrix()(1, 3) - T_SCAN_TARGET_EST.matrix()(1, 3));
  double dist_err = std::round(dist_diff.norm() * 10000) / 10000;

  // ------------------------------------------------
  // Ignoring rotation error due to unconstrained yaw
  // Eigen::Vector3d ypr_measured, ypr_estimated;
  // ypr_measured = measurement_.rotation().eulerAngles(2, 1, 0);
  // ypr_estimated = T_SCAN_TARGET_EST.rotation().eulerAngles(2, 1, 0);
  // Eigen::Vector2d rot_diff(ypr_measured(1) - ypr_estimated(1),
  //                          ypr_measured(2) - ypr_estimated(2)); // ignore yaw
  // double rot_err = std::round(rot_diff.norm() * 10000) / 10000;
  // if (dist_err >= registration_params_.dist_acceptance_criteria ||
  //     rot_err >= registration_params_.rot_acceptance_criteria)
  // ------------------------------------------------

  if (dist_err >= registration_params_.dist_acceptance_criteria) {
    measurement_valid_ = false;
    std::cout << "-----------------------------\n"
              << "Measurement Invalid\n"
              << "Distance error norm: " << dist_err << "\n"
              << "Distance acceptance criteria: "
              << registration_params_.dist_acceptance_criteria << "\n";
  } else {
    measurement_valid_ = true;
    std::cout << "-----------------------------\n"
              << "Measurement Valid\n"
              << "Distance error norm: " << dist_err << "\n"
              << "Distance acceptance criteria: "
              << registration_params_.dist_acceptance_criteria << "\n";
  }

  if (registration_params_.show_transform) {
    // Display clouds for testing
    // transform template cloud from target to lidar
    auto estimated_template_cloud =
        ColourPointCloud(template_cloud_, 0, 0, 255);
    pcl::transformPointCloud(*estimated_template_cloud,
                             *estimated_template_cloud, T_SCAN_TARGET_EST);
    AddColouredPointCloudToViewer(estimated_template_cloud,
                                  "estimated template cloud " +
                                      std::to_string(measurement_num),
                                  T_SCAN_TARGET_EST);

    auto measured_template_cloud = ColourPointCloud(template_cloud_, 0, 255, 0);
    pcl::transformPointCloud(*measured_template_cloud, *measured_template_cloud,
                             T_SCAN_TARGET_OPT);
    AddColouredPointCloudToViewer(measured_template_cloud,
                                  "measured template cloud " +
                                      std::to_string(measurement_num),
                                  T_SCAN_TARGET_OPT);

    AddPointCloudToViewer(scan_cropped,
                          "cropped scan " + std::to_string(measurement_num),
                          T_identity);

    ShowFinalTransformation();
  }
  measurement_complete_ = true;
}

} // end namespace vicon_calibration
